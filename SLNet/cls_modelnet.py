import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import argparse
import os
import logging
import datetime
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data import DataLoader
from utils import Logger, mkdir_p, progress_bar, save_model, save_args, cal_loss
from data.modelnet import ModelNet40
from torch.optim.lr_scheduler import CosineAnnealingLR
import sklearn.metrics as metrics
import numpy as np
import shutil

from encoder.Encoder import Encoder
from util import Classifier, log_experiment, configuration_exists

# Import AveragedModel for SWA (Exponential Moving Average)
from torch.optim.swa_utils import AveragedModel, update_bn

class Classification(nn.Module):
    def __init__(self,
                 n=1024,
                 embed=[3, 64, 'mlp', 'no', 0.3, 1000.0, 100.0],
                 res_dim_ratio=1.0,
                 bias=False,
                 use_xyz=True,
                 norm_mode=["anchor","yes"],
                 std_mode="BN1D",
                 dim_ratio=[2, 2, 2, 2],
                 num_blocks1=[2, 2, 2, 2],
                 transfer_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 block1_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 num_blocks2=[2, 2, 2, 2],
                 block2_mode=['mlp', 'mlp', 'mlp', 'mlp'],
                 k_neighbors=[32, 32, 32, 32],
                 sampling_mode=['fps', 'fps', 'fps', 'fps'],
                 sampling_ratio=[2, 2, 2, 2],
                 classifier_mode='mlp'):
        super(Classification, self).__init__()
        self.encoder = Encoder(
            n=n,
            embed=embed,
            res_dim_ratio=res_dim_ratio,
            bias=bias,
            use_xyz=use_xyz,
            norm_mode=norm_mode,
            std_mode=std_mode,
            dim_ratio=dim_ratio,
            num_blocks1=num_blocks1,
            transfer_mode=transfer_mode,
            block1_mode=block1_mode,
            num_blocks2=num_blocks2,
            block2_mode=block2_mode,
            k_neighbors=k_neighbors,
            sampling_mode=sampling_mode,
            sampling_ratio=sampling_ratio,
        )
        
        last_dim = embed[1]
        for d in dim_ratio:
            last_dim *= d
        
        self.classifier = Classifier(last_dim, classifier_mode, 40)
    
    def forward(self, xyz, feature):
        xyz_list, f_list = self.encoder(xyz=xyz, x=None, feature=feature)
        x = F.adaptive_max_pool1d(f_list[-1], 1).squeeze(dim=-1)
        return self.classifier(x)
             
def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def parse_args():
    parser = argparse.ArgumentParser('cls_modelnet_training')
    parser.add_argument('--n', type=int, default=1024, help='Point Number') 
    parser.add_argument('--embed', type=str, default='mlp', help='[mlp, tpe, gpe, pointhop]')
    parser.add_argument('--initial_dim', type=int, default=3, help='initial_dim')
    parser.add_argument('--embed_dim', type=int, default=64, help='embed_dim')
    parser.add_argument('--alpha_beta', type=str, default='no', help='alpha_beta')
    parser.add_argument('--sigma', type=float, default=0.3, help='sigma')
    parser.add_argument('--alpha', type=float, default=1000.0, help='alpha')
    parser.add_argument('--beta', type=float, default=100.0, help='beta')

    parser.add_argument('--res_dim_ratio', type=float, default=0.25,
                        help='Residual block dimension ratio.')
    parser.add_argument('--bias', type=bool, default=False, help='bias')
    parser.add_argument('--use_xyz', type=bool, default=True, help='Include xyz coordinates')
    parser.add_argument('--norm_mode', type=str, default='anchor-yes', help='[center, anchor]')
    parser.add_argument('--std_mode', type=str, default='BN1D', help='[1111, B111, BN11, BN1D]')
    
    parser.add_argument('--dim_ratio', type=str, default='2-2-2-1', help='dim_ratio')
    
    parser.add_argument('--num_blocks1', type=str, default='1-1-2-1', help='num_blocks1')
    parser.add_argument('--transfer_mode', type=str, default='mlp-mlp-mlp-mlp', help='transfer_mode')
    parser.add_argument('--block1_mode', type=str, default='mlp-mlp-mlp-mlp', help='block1_mode')
    
    parser.add_argument('--num_blocks2', type=str, default='1-1-2-1', help='num_blocks2')
    parser.add_argument('--block2_mode', type=str, default='mlp-mlp-mlp-mlp', help='block2_mode')

    parser.add_argument('--k_neighbors', type=str, default='24-24-24-24', help='k_neighbors')
    parser.add_argument('--sampling_mode', type=str, default='fps-fps-fps-fps', help='sampling_mode')
    parser.add_argument('--sampling_ratio', type=str, default='2-2-2-2', help='sampling_ratio')

    parser.add_argument('--classifier_mode', type=str, default='mlp', help='classifier_mode')

    parser.add_argument('-c', '--checkpoint', type=str, metavar='PATH',
                        help='path to save checkpoint (default: checkpoint)')
    parser.add_argument('--msg', type=str, help='message after checkpoint')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--epoch', default=300, type=int, help='number of epochs in training')
    parser.add_argument('--learning_rate', default=0.1, type=float, help='learning rate in training')
    parser.add_argument('--min_lr', default=0.005, type=float, help='minimum lr')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='weight decay')
    parser.add_argument('--seed', type=int, help='random seed')
    parser.add_argument('--workers', default=6, type=int, help='number of data loader workers')
    parser.add_argument('--ema', default='yes', type=str, help='set "yes" to enable EMA using AveragedModel')
    return parser.parse_args()

def main():
    args = parse_args()

    args.model = 'cls_modelnet' #norm_mode
    args.embed = [args.initial_dim, args.embed_dim, args.embed, args.alpha_beta, args.sigma, args.alpha, args.beta]
    args.dim_ratio = list(map(lambda x: int(x), args.dim_ratio.split('-')))
    args.norm_mode = list(map(lambda x: str(x), args.norm_mode.split('-')))

    args.num_blocks1 = list(map(lambda x: int(x), args.num_blocks1.split('-')))
    args.transfer_mode = list(map(lambda x: str(x), args.transfer_mode.split('-')))
    args.block1_mode = list(map(lambda x: str(x), args.block1_mode.split('-')))

    args.num_blocks2 = list(map(lambda x: int(x), args.num_blocks2.split('-')))
    args.block2_mode = list(map(lambda x: str(x), args.block2_mode.split('-')))

    args.k_neighbors = list(map(lambda x: int(x), args.k_neighbors.split('-')))
    args.sampling_mode = list(map(lambda x: str(x), args.sampling_mode.split('-')))
    args.sampling_ratio = list(map(lambda x: int(x), args.sampling_ratio.split('-')))

    if args.seed is None:
        args.seed = np.random.randint(1, 10000)
    os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.set_printoptions(10)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        os.environ['PYTHONHASHSEED'] = str(args.seed)
    time_str = datetime.datetime.now().strftime('-%Y%m%d%H%M%S')
    message = time_str if args.msg is None else "-" + args.msg
    args.checkpoint = 'checkpoints/' + args.model + '/' + message + '-' + str(args.seed)
    
    code_dir = f'{args.checkpoint}/code/'
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy('cls_modelnet.py', f'{code_dir}/cls_modelnet.py')
    shutil.copy('provider.py', f'{code_dir}/provider.py')
    shutil.copy('util.py', f'{code_dir}/util.py')
    shutil.copytree('encoder', f'{code_dir}/encoder')
    shutil.copytree('decoder', f'{code_dir}/encode')
    
    args.path = args.checkpoint
    if not os.path.isdir(args.checkpoint):
        mkdir_p(args.checkpoint)

    screen_logger = logging.getLogger("Model")
    screen_logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(os.path.join(args.checkpoint, "out.txt"))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    screen_logger.addHandler(file_handler)

    def printf(str):
        screen_logger.info(str)
        print(str)

    printf(f"args: {args}")
    printf('==> Building model..')
    net = Classification(n=args.n,
                         embed=args.embed,
                         res_dim_ratio=args.res_dim_ratio,
                         bias=args.bias,
                         use_xyz=args.use_xyz,           
                         norm_mode=args.norm_mode,
                         std_mode=args.std_mode,
                         dim_ratio=args.dim_ratio,
                         num_blocks1=args.num_blocks1,
                         transfer_mode=args.transfer_mode,
                         block1_mode=args.block1_mode,
                         num_blocks2=args.num_blocks2,
                         block2_mode=args.block2_mode,
                         k_neighbors=args.k_neighbors,
                         sampling_mode=args.sampling_mode,
                         sampling_ratio=args.sampling_ratio,
                         classifier_mode=args.classifier_mode)
    
    criterion = cal_loss
    net = net.to(device)
    args.param = trainable_params(net)
    printf(f'number of params: {args.param}')
    if device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    # Initialize AveragedModel for EMA if enabled
    ema_net = None
    if args.ema.lower() == "yes":
        ema_net = AveragedModel(net)
        ema_net = ema_net.to(device)
        printf("AveragedModel (SWA/EMA) is enabled.")

    best_test_acc = 0.0
    best_train_acc = 0.0
    best_test_acc_avg = 0.0
    best_train_acc_avg = 0.0
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0
    optimizer_dict = None

    if not os.path.isfile(os.path.join(args.checkpoint, "last_checkpoint.pth")):
        save_args(args)
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])
    else:
        printf(f"Resuming last checkpoint from {args.checkpoint}")
        checkpoint_path = os.path.join(args.checkpoint, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        net.load_state_dict(checkpoint['net'])
        if ema_net is not None and "ema_net" in checkpoint:
            ema_net.load_state_dict(checkpoint["ema_net"])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(args.checkpoint, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')
    train_loader = DataLoader(ModelNet40(partition='train', num_points=args.n), num_workers=args.workers,
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.n), num_workers=args.workers,
                             batch_size=args.batch_size // 2, shuffle=False, drop_last=False)

    optimizer = torch.optim.SGD(net.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.weight_decay)
    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr, last_epoch=start_epoch - 1)

    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(net, train_loader, optimizer, criterion, device, ema_net)
        test_out = validate(net, test_loader, criterion, device)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        # Save checkpoint including ema_net state
        save_model(
            net, epoch, path=args.checkpoint, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict(),
            ema_net=ema_net.state_dict() if ema_net is not None else None
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf("++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf("++++++++" * 5)

    results = {
        "task": "cls_modelnet",
        "OA_test": best_test_acc,
        "mAcc_test": best_test_acc_avg,
        "OA_train": best_train_acc,
        "mAcc_train": best_train_acc_avg,
        "num_param": args.param,
        "path": args.path,
        "n": args.n,
        "embed": args.embed,
        "initial_dim": args.initial_dim,
        "embed_dim": args.embed_dim,
        "res_dim_ratio": args.res_dim_ratio,
        "bias": args.bias,
        "use_xyz": args.use_xyz,
        "norm_mode": args.norm_mode,
        "std_mode": args.std_mode,
        "dim_ratio": args.dim_ratio,
        "num_blocks1": args.num_blocks1,
        "transfer_mode": args.transfer_mode,
        "block1_mode": args.block1_mode,
        "num_blocks2": args.num_blocks2,
        "block2_mode": args.block2_mode,
        "k_neighbors": args.k_neighbors,
        "sampling_mode": args.sampling_mode,
        "sampling_ratio": args.sampling_ratio,
        "classifier_mode": args.classifier_mode,
        "checkpoint": args.checkpoint,
        "msg": args.msg,
        "batch_size": args.batch_size,
        "epoch": f"{args.epoch}_ema:{args.ema}" if args.ema=='yes' else args.epoch,
        "learning_rate": args.learning_rate,
        "min_lr": args.min_lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "workers": args.workers,
    }
    log_experiment(results, excel_path='cls_modelnet_experiment.xlsx')

def train(net, trainloader, optimizer, criterion, device, ema_net=None):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_pred = []
    train_true = []
    time_cost = datetime.datetime.now()
    for batch_idx, (data, feature, label) in enumerate(trainloader):
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        optimizer.zero_grad()
        logits = net(data, feature)
        loss = criterion(logits, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1)
        optimizer.step()

        # Update the averaged model parameters
        if ema_net is not None:
            ema_net.update_parameters(net)

        train_loss += loss.item()
        preds = logits.max(dim=1)[1]
        train_true.append(label.cpu().numpy())
        train_pred.append(preds.detach().cpu().numpy())
        total += label.size(0)
        correct += preds.eq(label).sum().item()
        progress_bar(batch_idx, len(trainloader), 
                     'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    train_true = np.concatenate(train_true)
    train_pred = np.concatenate(train_pred)
    return {
        "loss": float("%.3f" % (train_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(train_true, train_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(train_true, train_pred))),
        "time": time_cost
    }

def validate(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    test_true = []
    test_pred = []
    time_cost = datetime.datetime.now()
    with torch.no_grad():
        for batch_idx, (data, feature, label) in enumerate(testloader):
            data, label = data.to(device), label.to(device).squeeze()
            data = data.permute(0, 2, 1)
            logits = net(data, feature)
            loss = criterion(logits, label)
            test_loss += loss.item()
            preds = logits.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())
            total += label.size(0)
            correct += preds.eq(label).sum().item()
            progress_bar(batch_idx, len(testloader), 
                         'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                         % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
    time_cost = int((datetime.datetime.now() - time_cost).total_seconds())
    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    return {
        "loss": float("%.3f" % (test_loss / (batch_idx + 1))),
        "acc": float("%.3f" % (100. * metrics.accuracy_score(test_true, test_pred))),
        "acc_avg": float("%.3f" % (100. * metrics.balanced_accuracy_score(test_true, test_pred))),
        "time": time_cost
    }

if __name__ == '__main__':
    main()
    #pass




if 0:#__name__ == '__main__':
    def all_params(model):
        return sum(p.numel() for p in model.parameters())
    def trainable_params(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    data = torch.rand(2, 3, 1024)
    feature = None
    print("===> testing Model ...")
    for embd in [[3,16,'mlp'],[3,18,'fourier'],[3,16,'gaussian'],[3,16,'harmonic']]:
        print(f"\n\nembd: {embd};")
        model = Classification(embed=embd)
        print(f'number of params: {trainable_params(model)}')
        feature = torch.rand(2, 16, 1024) if embd[-1] == 'pointhop' else None
        out = model(xyz=data, feature=feature)
        print(out.shape)