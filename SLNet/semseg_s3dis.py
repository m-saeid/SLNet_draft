"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
from data.s3dis import S3DISDataset
import torch
import datetime
import logging
from pathlib import Path
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from util import log_experiment

import torch.nn.functional as F

from decoder.Decoder import Decoder
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'table', 'chair', 'sofa', 'bookcase',
           'board', 'clutter']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True



class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss



def trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(f'checkpoints/{args.model}')

    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)

    # experiment_dir = Path(f'checkpoints/{args.model}/log/')
    #experiment_dir.mkdir(exist_ok=True)
    #experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    
    args.path = experiment_dir

    checkpoints_dir = experiment_dir + '/checkpoints/'
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir + '/logs/'
    log_dir.mkdir(exist_ok=True)

    code_dir = experiment_dir + '/codes/'
    code_dir.mkdir(exist_ok=True)
    # os.makedirs(code_dir, exist_ok=True)
    shutil.copy('semseg_s3dis.py', f'{code_dir}/semseg_s3dis.py')
    shutil.copy('provider.py', f'{code_dir}/provider.py')
    shutil.copy('util.py', f'{code_dir}/util.py')
    shutil.copytree('encoder', f'{code_dir}/encoder')
    shutil.copytree('decoder', f'{code_dir}/encoder')


    '''    
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath(f'checkpoints/{args.model}/')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('checkpoints/{args.model}/logs/')
    log_dir.mkdir(exist_ok=True)
    '''

    '''LOG'''
    #args = parse_args()add_argument
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = "/home/anil/Desktop/saeid/code/papers/Pointnet_Pointnet2_pytorch-master/data/stanford_indoor3d" # 'data/stanford_indoor3d/'
    NUM_CLASSES = 13
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    print("start loading training data ...")
    TRAIN_DATASET = S3DISDataset(split='train', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)
    print("start loading test data ...")
    TEST_DATASET = S3DISDataset(split='test', data_root=root, num_point=NUM_POINT, test_area=args.test_area, block_size=1.0, sample_rate=1.0, transform=None)

    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=10,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=10,
                                                 pin_memory=True, drop_last=True)
    weights = torch.Tensor(TRAIN_DATASET.labelweights).cuda()

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    #MODEL = importlib.import_module(args.model)
    # shutil.copy('models/%s.py' % args.model, str(experiment_dir)) ###################################
    # shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    #classifier = MODEL.get_model(NUM_CLASSES).cuda() ###################
    criterion = get_loss().cuda() # MODEL.get_loss().cuda()
    #classifier.apply(inplace_relu) ######################################
    '''
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
    
    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    '''


    start_epoch = 0
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


    classifier = Decoder(
    task=args.task,
    n=args.n,
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


    classifier.apply(weight_init)
    classifier = nn.DataParallel(classifier)
    classifier.apply(inplace_relu) ###########################
    print("Let's use", torch.cuda.device_count(), "GPUs!")

    args.param = trainable_params(classifier)
    print(f'number of params: {args.param}')

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    ### TRAIN ###
    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        ### TRAIN ###
        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()   # points; 16,4096,9     target: 16,4096

            points = points.data.numpy()           # 16,4096,9  
            points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])  # 16,4096,9  
            points = torch.Tensor(points)                                       # 16,4096,9 
            points, target = points.float().cuda(), target.long().cuda()    # points; 16,4096,9     target: 16,4096
            points = points.transpose(2, 1)                                 # 16,9,4096

            # seg_pred, trans_feat = classifier(points)           # input; points(16,9,4096)  -   output; seg_pred(16,4096,13), trans_feat(2,512,16)
            seg_pred, trans_feat = classifier(points,None,None,'s3dis')
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES) # B*N, num_cls

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy() # B*N, 
            target = target.view(-1, 1)[:, 0]                         # B*N
            loss = criterion(seg_pred, target, trans_feat, weights)   # inputs; seg_pred,(B*N,num_cls) target(B*N), trans_feat(B,512,16), weights(16)   output(loss())
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy() # B*N, num_cls > B*N
            correct = np.sum(pred_choice == batch_label)        # ()
            total_correct += correct                            # ()
            total_seen += (BATCH_SIZE * NUM_POINT)              # int
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points,None,None,'s3dis')
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


    results = {
    "task" : "semseg_s3dis",

    "Best mIoU" : best_iou,

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
    "batch_size" : args.batch_size,
    "epoch" : args.epochs,
    "learning_rate" : args.learning_rate,
    "gpu" : args.gpu,
    "optimizer" : args.optimizer,
    "log_dir" : args.log_dir,
    "decay_rate" : args.decay_rate,
    "npoint" : args.npoint,
    "step_size" : args.step_size,
    "lr_decay" : args.lr_decay,
    "test_area" : args.test_area,
    }


    log_experiment(results, excel_path='partseg_shapenet_experiment.xlsx')


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser(description='3D Shape Semantic Segmentation_S3DIS')

    parser.add_argument('--task', type=str, default='semseg_s3dis', help='model')

    parser.add_argument('--model', type=str, default='semseg_s3dis', help='model')

    # Encoder
    parser.add_argument('--n', type=int, default=4096, help='Point Number') 
    parser.add_argument('--embed', type=str, default='mlp', help='[mlp, tpe, gpe, pointhop]')
    parser.add_argument('--initial_dim', type=int, default=9, help='embed_dim')
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

    parser.add_argument('--num_cls', type=int, default=13, help='num_cls')
    parser.add_argument('--classifier_mode', type=str, default='mlp', help='classifier_mode')


    #parser.add_argument('--model', type=str, default='pointnet2_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]') # 16 > 2
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')


    args = parser.parse_args()

    # Encoder
    args.embed = [args.initial_dim, args.embed_dim, args.embed]
    args.dim_ratio = list(map(lambda x:int(x), args.dim_ratio.split('-')))
    args.num_blocks1 = list(map(lambda x:int(x), args.num_blocks1.split('-')))
    args.num_blocks2 = list(map(lambda x:int(x), args.num_blocks2.split('-')))
    args.k_neighbors = list(map(lambda x:int(x), args.k_neighbors.split('-')))
    args.sampling_mode = list(map(lambda x:str(x), args.sampling_mode.split('-')))
    args.sampling_ratio = list(map(lambda x:int(x), args.sampling_ratio.split('-')))

    # Decoder
    args.de_dims = list(map(lambda x:int(x), args.de_dims.split('-')))
    args.de_blocks = list(map(lambda x:int(x), args.de_blocks.split('-')))
    args.de_fp_fuse = list(map(lambda x:str(x), args.de_fp_fuse.split('-')))
    args.de_fp_block = list(map(lambda x:str(x), args.de_fp_block.split('-')))

    main(args)
