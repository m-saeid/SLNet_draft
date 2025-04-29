from __future__ import print_function
import os
import argparse
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from data.shapenet import PartNormalDataset
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from util import to_categorical, compute_overall_iou, log_experiment, IOStream
from tqdm import tqdm
from collections import defaultdict
from torch.autograd import Variable
import random
from decoder.Decoder import Decoder
import datetime
import shutil


classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(f'checkpoints/partseg_shapenet/' + args.msg):
        os.makedirs(f'checkpoints/partseg_shapenet/' + args.msg)

    args.path = f'checkpoints/partseg_shapenet/{args.msg}'
    code_dir = f'checkpoints/partseg_shapenet/{args.msg}/code/'
    os.makedirs(code_dir, exist_ok=True)
    shutil.copy('partseg_shapenet.py', f'{code_dir}/partseg_shapenet.py')
    shutil.copy('provider.py', f'{code_dir}/provider.py')
    shutil.copy('util.py', f'{code_dir}/util.py')
    shutil.copytree('encoder', f'{code_dir}/encoder')
    shutil.copytree('decoder', f'{code_dir}/decoder')


def weight_init(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.Conv1d):
        torch.nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif isinstance(m, torch.nn.BatchNorm1d):
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)


def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(args, io):

    # ============= Model ===================
    device = torch.device("cuda" if args.cuda else "cpu")

    # model = models.__dict__[args.model](num_part).to(device)
    model = Decoder(n=args.n,
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
    
                    # Decoder:
                    de_dims=args.de_dims,
                    de_blocks=args.de_blocks,

                    de_fp_fuse=args.de_fp_fuse,
                    de_fp_block=args.de_fp_block,

                    gmp_dim=args.gmp_dim,
                    gmp_dim_mode=args.gmp_dim_mode,

                    cls_dim=args.cls_dim,
                    cls_map_mode=args.cls_map_mode,
                    gmp_map_end_mode=args.gmp_map_end_mode,

                    num_cls=args.num_cls,
                    classifier_mode=args.classifier_mode)
    
    num_part = args.num_cls
    io.cprint(str(model))

    model.apply(weight_init)
    model = nn.DataParallel(model)

    args.param = trainable_params(model)
    print(f'number of params: {args.param}')

    print("Let's use", torch.cuda.device_count(), "GPUs!")

    '''Resume or not'''
    if args.resume:
        state_dict = torch.load("checkpoints/partseg_shapenet/%s/best_insiou_model.pth" % args.msg,
                                map_location=torch.device('cpu'))['model']
        for k in state_dict.keys():
            if 'module' not in k:
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k in state_dict:
                    new_state_dict['module.' + k] = state_dict[k]
                state_dict = new_state_dict
            break
        model.load_state_dict(state_dict)

        print("Resume training model...")
        print(torch.load("checkpoints/partseg_shapenet/%s/best_insiou_model.pth" % args.msg).keys())
    else:
        print("Training from scratch...")

    # =========== Dataloader =================
    train_data = PartNormalDataset(npoints=2048, split='trainval', normalize=False)
    print("The number of training data is:%d", len(train_data))

    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              drop_last=True)

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # ============= Optimizer ================
    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=0)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

    if args.scheduler == 'cos':
        print("Use CosLR")
        scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr if args.use_sgd else args.lr / 100)
    else:
        print("Use StepLR")
        scheduler = StepLR(opt, step_size=args.step, gamma=0.5)

    # ============= Training =================
    best_acc = 0
    best_class_iou = 0
    best_instance_iou = 0
    num_part = 50
    num_classes = 16

    for epoch in range(args.epochs):

        train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io)

        test_metrics, total_per_cat_iou = test_epoch(test_loader, model, epoch, num_part, num_classes, io)

        # 1. when get the best accuracy, save the model:
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
            io.cprint('Max Acc:%.5f' % best_acc)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_acc': best_acc}
            torch.save(state, 'checkpoints/partseg_shapenet/%s/best_acc_model.pth' % args.msg)

        # 2. when get the best instance_iou, save the model:
        if test_metrics['shape_avg_iou'] > best_instance_iou:
            best_instance_iou = test_metrics['shape_avg_iou']
            io.cprint('Max instance iou:%.5f' % best_instance_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_instance_iou': best_instance_iou}
            torch.save(state, 'checkpoints/partseg_shapenet/%s/best_insiou_model.pth' % args.msg)

        # 3. when get the best class_iou, save the model:
        # first we need to calculate the average per-class iou
        class_iou = 0
        for cat_idx in range(16):
            class_iou += total_per_cat_iou[cat_idx]
        avg_class_iou = class_iou / 16
        if avg_class_iou > best_class_iou:
            best_class_iou = avg_class_iou
            # print the iou of each class:
            for cat_idx in range(16):
                io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))
            io.cprint('Max class iou:%.5f' % best_class_iou)
            state = {
                'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
                'optimizer': opt.state_dict(), 'epoch': epoch, 'test_class_iou': best_class_iou}
            torch.save(state, 'checkpoints/partseg_shapenet/%s/best_clsiou_model.pth' % args.msg)

    # report best acc, ins_iou, cls_iou
    io.cprint('Final Max Acc:%.5f' % best_acc)
    io.cprint('Final Max instance iou:%.5f' % best_instance_iou)
    io.cprint('Final Max class iou:%.5f' % best_class_iou)
    # save last model
    state = {
        'model': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer': opt.state_dict(), 'epoch': args.epochs - 1, 'test_iou': best_instance_iou}
    torch.save(state, 'checkpoints/partseg_shapenet/%s/model_ep%d.pth' % (args.msg, args.epochs))

    results = {
        "task" : "partseg_shapenet",

        "Final Max Acc" : best_acc,
        "Final Max instance iou" : best_instance_iou,
        "Final Max class iou" : best_class_iou,

        "num_param" : args.param,

        "path" : args.path,

        "n" : args.n,
        "embed" : args.embed,
        "initial_dim" : args.initial_dim,
        "embed_dim" : args.embed_dim,
        "res_dim_ratio" : args.res_dim_ratio,
        "bias" : args.bias,
        "use_xyz" : args.use_xyz,
        "norm_mode" : args.norm_mode,
        "std_mode" : args.std_mode,
        "dim_ratio" : args.dim_ratio,
        "num_blocks1" : args.num_blocks1,
        "transfer_mode" : args.transfer_mode,
        "block1_mode" : args.block1_mode,
        "num_blocks2" : args.num_blocks2,
        "block2_mode" : args.block2_mode,
        "k_neighbors" : args.k_neighbors,
        "sampling_mode" : args.sampling_mode,
        "sampling_ratio" : args.sampling_ratio,
        "de_dims" : args.de_dims,
        "de_blocks" : args.de_blocks,
        "de_fp_fuse" : args.de_fp_fuse,
        "de_fp_block" : args.de_fp_block,
        "gmp_dim" : args.gmp_dim,
        "gmp_dim_mode" : args.gmp_dim_mode,
        "cls_dim" : args.cls_dim,
        "cls_map_mode" : args.cls_map_mode,
        "gmp_map_end_mode" : args.gmp_map_end_mode,
        "num_cls" : args.num_cls,

        "classifier_mode" : args.classifier_mode,

        "model" : args.model,
        "msg" : args.msg,
        "batch_size" : args.batch_size,
        "test_batch_size" : args.test_batch_size,
        "epochs" : args.epochs,
        "use_sgd" : args.use_sgd,
        "scheduler" : args.scheduler,
        "step" : args.step,
        "lr" : args.lr,
        "momentum" : args.momentum,
        "no_cuda" : args.no_cuda,
        "manual_seed" : args.manual_seed,
        "eval" : args.eval,
        "num_points" : args.num_points,
        "workers" : args.workers,
        "resume" : args.resume,
        "model_type" : args.model_type
    }

    log_experiment(results, excel_path='partseg_shapenet_experiment.xlsx')

def train_epoch(train_loader, model, opt, scheduler, epoch, num_part, num_classes, io):
    train_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    metrics = defaultdict(lambda: list())
    model.train()

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        # target: b,n
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))[0]  # seg_pred: b,n,50
        loss = F.nll_loss(seg_pred.contiguous().view(-1, num_part), target.view(-1, 1)[:, 0])

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # list of of current batch_iou:[iou1,iou2,...,iou#b_size]
        # total iou of current batch in each process:
        batch_shapeious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # Loss backward
        loss = torch.mean(loss)
        opt.zero_grad()
        loss.backward()
        opt.step()

        # accuracy
        seg_pred = seg_pred.contiguous().view(-1, num_part)  # b*n,50
        target = target.view(-1, 1)[:, 0]   # b*n
        pred_choice = seg_pred.contiguous().data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.contiguous().data).sum()  # torch.int64: total number of correct-predict pts

        # sum
        shape_ious += batch_shapeious.item()  # count the sum of ious in each iteration
        count += batch_size   # count the total number of samples in each iteration
        train_loss += loss.item() * batch_size
        accuracy.append(correct.item()/(batch_size * num_point))   # append the accuracy of each iteration

        # Note: We do not need to calculate per_class iou during training

    if args.scheduler == 'cos':
        scheduler.step()
    elif args.scheduler == 'step':
        if opt.param_groups[0]['lr'] > 0.9e-5:
            scheduler.step()
        if opt.param_groups[0]['lr'] < 0.9e-5:
            for param_group in opt.param_groups:
                param_group['lr'] = 0.9e-5
    io.cprint('Learning rate: %f' % opt.param_groups[0]['lr'])

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Train %d, loss: %f, train acc: %f, train ins_iou: %f' % (epoch+1, train_loss * 1.0 / count,
                                                                       metrics['accuracy'], metrics['shape_avg_iou'])
    io.cprint(outstr)


def test_epoch(test_loader, model, epoch, num_part, num_classes, io):
    test_loss = 0.0
    count = 0.0
    accuracy = []
    shape_ious = 0.0
    final_total_per_cat_iou = np.zeros(16).astype(np.float32)
    final_total_per_cat_seen = np.zeros(16).astype(np.int32)
    metrics = defaultdict(lambda: list())
    model.eval()

    # label_size: b, means each sample has one corresponding class
    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), \
                                          Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze(1).cuda(non_blocking=True), \
                                          target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
        seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        # per category iou at each batch_size:

        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx], denotes current sample belongs to which cat
            final_total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]  # add the iou belongs to this cat
            final_total_per_cat_seen[cur_gt_label] += 1  # count the number of this cat is chosen

        # total iou of current batch in each process:
        batch_ious = seg_pred.new_tensor([np.sum(batch_shapeious)], dtype=torch.float64)  # same device with seg_pred!!!

        # prepare seg_pred and target for later calculating loss and acc:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        # Loss
        loss = F.nll_loss(seg_pred.contiguous(), target.contiguous())

        # accuracy:
        pred_choice = seg_pred.data.max(1)[1]  # b*n
        correct = pred_choice.eq(target.data).sum()  # torch.int64: total number of correct-predict pts

        loss = torch.mean(loss)
        shape_ious += batch_ious.item()  # count the sum of ious in each iteration
        count += batch_size  # count the total number of samples in each iteration
        test_loss += loss.item() * batch_size
        accuracy.append(correct.item() / (batch_size * num_point))  # append the accuracy of each iteration

    for cat_idx in range(16):
        if final_total_per_cat_seen[cat_idx] > 0:  # indicating this cat is included during previous iou appending
            final_total_per_cat_iou[cat_idx] = final_total_per_cat_iou[cat_idx] / final_total_per_cat_seen[cat_idx]  # avg class iou across all samples

    metrics['accuracy'] = np.mean(accuracy)
    metrics['shape_avg_iou'] = shape_ious * 1.0 / count

    outstr = 'Test %d, loss: %f, test acc: %f  test ins_iou: %f' % (epoch + 1, test_loss * 1.0 / count,
                                                                    metrics['accuracy'], metrics['shape_avg_iou'])

    io.cprint(outstr)

    return metrics, final_total_per_cat_iou


def test(args, io):
    # Dataloader
    test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
    print("The number of test data is:%d", len(test_data))

    test_loader = DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=args.workers,
                             drop_last=False)

    # Try to load models
    num_part = 50
    device = torch.device("cuda" if args.cuda else "cpu")

    # model = models.__dict__[args.model](num_part).to(device)
    model = Decoder(n=args.n,
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

                    # Decoder:
                    de_dims=args.de_dims,
                    de_blocks=args.de_blocks,

                    de_fp_fuse=args.de_fp_fuse,
                    de_fp_block=args.de_fp_block,

                    gmp_dim=args.gmp_dim,
                    gmp_dim_mode=args.gmp_dim_mode,

                    cls_dim=args.cls_dim,
                    cls_map_mode=args.cls_map_mode,
                    gmp_map_end_mode=args.gmp_map_end_mode,

                    num_cls=args.num_cls,
                    classifier_mode=args.classifier_mode
                    )
    io.cprint(str(model))

    from collections import OrderedDict
    state_dict = torch.load("checkpoints/partseg_shapenet/%s/best_%s_model.pth" % (args.msg, args.model_type),
                            map_location=torch.device('cpu'))['model']

    new_state_dict = OrderedDict()
    for layer in state_dict:
        new_state_dict[layer.replace('module.', '')] = state_dict[layer]
    model.load_state_dict(new_state_dict)

    model.eval()
    num_part = 50
    num_classes = 16
    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    total_per_cat_iou = np.zeros((16)).astype(np.float32)
    total_per_cat_seen = np.zeros((16)).astype(np.int32)

    for batch_id, (points, label, target, norm_plt) in tqdm(enumerate(test_loader), total=len(test_loader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(
            non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = model(points, norm_plt, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        shape_ious += batch_shapeious  # iou +=, equals to .append

        # per category iou at each batch_size:
        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx]
            total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
            total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['shape_avg_iou'] = np.mean(shape_ious)
    for cat_idx in range(16):
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    class_iou = 0
    for cat_idx in range(16):
        class_iou += total_per_cat_iou[cat_idx]
        io.cprint(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    avg_class_iou = class_iou / 16
    outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    io.cprint(outstr)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Part Segmentation_shapenet')

    # Encoder
    parser.add_argument('--n', type=int, default=2048, help='Point Number') 
    parser.add_argument('--embed', type=str, default='mlp', help='[mlp, tpe, gpe, pointhop]')
    parser.add_argument('--initial_dim', type=int, default=6, help='embed_dim')
    parser.add_argument('--embed_dim', type=int, default=16, help='embed_dim')
    parser.add_argument('--res_dim_ratio', type=float, default=0.25,
                        help='In residual blocks, the tensor dimension is changed by MLP in a certain ratio and returns to the original dimension.')
    parser.add_argument('--bias', type=bool, default=False, help='bias')
    parser.add_argument('--use_xyz', type=bool, default=True, help='Connecting initial 3D points to features')
    parser.add_argument('--norm_mode', type=str, default='anchor', help='norm_mode')
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

    # Decoder
    parser.add_argument('--de_dims', type=str, default='512-256-128-128', help='sampling_ratio')
    parser.add_argument('--de_blocks', type=str, default='2-2-2-2', help='sampling_ratio')
    
    parser.add_argument('--de_fp_fuse', type=str, default='mlp-mlp-mlp-mlp', help='de_fp_fuse')
    parser.add_argument('--de_fp_block', type=str, default='mlp-mlp-mlp-mlp', help='de_fp_block')
    
    parser.add_argument('--gmp_dim', type=int, default=64, help='gmp_dim')
    parser.add_argument('--gmp_dim_mode', type=str, default='mlp', help='gmp_dim_mode')

    parser.add_argument('--cls_dim', type=int, default=64, help='cls_dim')
    parser.add_argument('--cls_map_mode', type=str, default='mlp', help='cls_map_mode')
    parser.add_argument('--gmp_map_end_mode', type=str, default='mlp', help='gmp_map_end_mode')

    parser.add_argument('--num_cls', type=int, default=50, help='num_cls')
    parser.add_argument('--classifier_mode', type=str, default='mlp', help='classifier_mode')


    parser.add_argument('--model', type=str, default='seg_shapenet')
    parser.add_argument('--msg', type=str, metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=350, metavar='N',
                        help='number of episode to train')
    parser.add_argument('--use_sgd', type=bool, default=False,
                        help='Use SGD')
    parser.add_argument('--scheduler', type=str, default='step',
                        help='lr scheduler')
    parser.add_argument('--step', type=int, default=40,
                        help='lr decay step')
    parser.add_argument('--lr', type=float, default=0.003, metavar='LR',
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--manual_seed', type=int, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    #parser.add_argument('--num_points', type=int, default=2048,
    #                    help='num of points to use')
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--resume', type=bool, default=False,
                        help='Resume training or not')
    parser.add_argument('--model_type', type=str, default='insiou',
                        help='choose to test the best insiou/clsiou/acc model (options: insiou, clsiou, acc)')

    args = parser.parse_args()


    # Encoder
    args.embed = [args.initial_dim, args.embed_dim, args.embed]
    args.dim_ratio = list(map(lambda x:int(x), args.dim_ratio.split('-'))) # 
    args.num_blocks1 = list(map(lambda x:int(x), args.num_blocks1.split('-')))
    args.transfer_mode = list(map(lambda x:str(x), args.transfer_mode.split('-')))
    args.block1_mode = list(map(lambda x:str(x), args.block1_mode.split('-')))
    args.num_blocks2 = list(map(lambda x:int(x), args.num_blocks2.split('-')))
    args.block2_mode = list(map(lambda x:str(x), args.block2_mode.split('-')))
    args.k_neighbors = list(map(lambda x:int(x), args.k_neighbors.split('-')))
    args.sampling_mode = list(map(lambda x:str(x), args.sampling_mode.split('-')))
    args.sampling_ratio = list(map(lambda x:int(x), args.sampling_ratio.split('-')))

    # Decoder
    args.de_dims = list(map(lambda x:int(x), args.de_dims.split('-')))
    args.de_blocks = list(map(lambda x:int(x), args.de_blocks.split('-')))

    args.de_fp_fuse = list(map(lambda x:str(x), args.de_fp_fuse.split('-')))
    args.de_fp_block = list(map(lambda x:str(x), args.de_fp_block.split('-')))


    time_str = str(datetime.datetime.now().strftime('-%Y%m%d%H%M%S'))
    if args.msg is None:
        args.msg = time_str + '-' + str(args.manual_seed)
    else:
        args.msg = args.msg + "-" + str(args.manual_seed)

    # args.msg = args.model # +"_"+args.msg

    _init_()

    if not args.eval:
        io = IOStream(f'checkpoints/partseg_shapenet/' + args.msg + '/%s_train.log' % (args.msg))
    else:
        io = IOStream(f'checkpoints/partseg_shapenet/' + args.msg + '/%s_test.log' % (args.msg))
    io.cprint(str(args))

    if args.manual_seed is not None:
        random.seed(args.manual_seed)
        np.random.seed(args.manual_seed)
        torch.manual_seed(args.manual_seed)

    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.cuda:
        io.cprint('Using GPU')
        if args.manual_seed is not None:
            torch.cuda.manual_seed(args.manual_seed)
            torch.cuda.manual_seed_all(args.manual_seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)