import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder.encodings import *

import pandas as pd
import os

def log_experiment(results, excel_path='experiment_results.xlsx'):
    df_new = pd.DataFrame([results])
    # If file exists, append the new results to it
    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        df_combined.to_excel(excel_path, index=False)
    else:
        df_new.to_excel(excel_path, index=False)

def configuration_exists(args, excel_path='cls_modelnet_experiment.xlsx'):
    """Check if the current configuration already exists in the results."""
    if os.path.exists(excel_path):
        df = pd.read_excel(excel_path)
        # Check if the configuration exists
        # You can customize this check based on the parameters you consider unique
        
        for i in range(len(df)):
            if (str(df['n'][i]) == str(args.n)) and \
                (str(df['embed'][i]) == str(args.embed)) and \
                (str(df['initial_dim'][i]) == str(args.initial_dim)) and \
                (str(df['embed_dim'][i]) == str(args.embed_dim)) and \
                (str(df['res_dim_ratio'][i]) == str(args.res_dim_ratio)) and \
                (str(df['norm_mode'][i]) == str(args.norm_mode)) and \
                (str(df['std_mode'][i]) == str(args.std_mode)) and \
                (str(df['dim_ratio'][i]) == str(args.dim_ratio)) and \
                (str(df['num_blocks1'][i]) == str(args.num_blocks1)) and \
                (str(df['transfer_mode'][i]) == str(args.transfer_mode)) and \
                (str(df['block1_mode'][i]) == str(args.block1_mode)) and \
                (str(df['num_blocks2'][i]) == str(args.num_blocks2)) and \
                (str(df['block2_mode'][i]) == str(args.block2_mode)) and \
                (str(df['k_neighbors'][i]) == str(args.k_neighbors)) and \
                (str(df['sampling_mode'][i]) == str(args.sampling_mode)) and \
                (str(df['sampling_ratio'][i]) == str(args.sampling_ratio)) and \
                (str(df['batch_size'][i]) == str(args.batch_size)) and \
                (str(df['epoch'][i]) == str(args.epoch)) and \
                (str(df['learning_rate'][i]) == str(args.learning_rate)) and \
                (str(df['min_lr'][i]) == str(args.min_lr)) and \
                (str(df['weight_decay'][i]) == str(args.weight_decay)) and \
                (str(df['seed'][i]) == str(args.seed)) and \
                (str(df['classifier_mode'][i]) == str(args.classifier_mode)):
                return True
        return False
            
'''
        existing = df[
            (df['n'] == args.n) &
            (df['embed'] == args.embed) &
            (df['initial_dim'] == args.initial_dim) &
            (df['embed_dim'] == args.embed_dim) &
            (df['res_dim_ratio'] == args.res_dim_ratio) &
            # bias:0orFalse - use_xyz:1orTrue
            (df['norm_mode'] == args.norm_mode) &
            (df['std_mode'] == args.std_mode) &

            (df['dim_ratio'] == args.dim_ratio) &
            (df['num_blocks1'] == args.num_blocks1) &
            (df['transfer_mode'] == args.transfer_mode) &
            (df['block1_mode'] == args.block1_mode) &
            (df['num_blocks2'] == args.num_blocks2) &
            (df['block2_mode'] == args.block2_mode) &
            (df['k_neighbors'] == args.k_neighbors) &
            (df['sampling_mode'] == args.sampling_mode) &
            (df['sampling_ratio'] == args.sampling_ratio) &

            (df['batch_size'] == args.batch_size) &
            (df['epoch'] == args.epoch) &
            (df['learning_rate'] == args.learning_rate) &
            (df['min_lr'] == args.min_lr) &
            (df['weight_decay'] == args.weight_decay) &
            (df['seed'] == args.seed) &

            (df['classifier_mode'] == args.classifier_mode)
        ]
        return not existing.empty
    return False
'''
################# Classification ################

class Classifier(nn.Module):
    def __init__(self, last_channel, classifier_mode, num_cls):
        super(Classifier, self).__init__()

        if last_channel >=2048:
            if classifier_mode == "mlp_very_large":
                l1 = nn.Linear(last_channel, 2048)
                l2 = nn.Linear(2048, 1024)
                l3 = nn.Linear(1024, 512)
                l4 = nn.Linear(512, 256)
                l5 = nn.Linear(256, 128)
                l6 = nn.Linear(128, 64)
                l7 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(2048),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l5,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l6,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l7
                )
            elif classifier_mode == "mlp_large":
                l1 = nn.Linear(last_channel, 1024)
                l2 = nn.Linear(1024, 512)
                l3 = nn.Linear(512, 256)
                l4 = nn.Linear(256, 128)
                l5 = nn.Linear(128, 64)
                l6 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l5,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l6
                    )
            elif classifier_mode == "mlp_medium":
                l1 = nn.Linear(last_channel, 1024)
                l2 = nn.Linear(1024, 512)
                l3 = nn.Linear(512, 128)
                l4 = nn.Linear(128, 64)
                l5 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l5
                    )
            elif classifier_mode == "mlp_small":
                l1 = nn.Linear(last_channel, 512)
                l2 = nn.Linear(512, 256)
                l3 = nn.Linear(256, 128)
                l4 = nn.Linear(128, 64)
                l5 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l5
                    )
            elif classifier_mode == "mlp_very_small":
                l1 = nn.Linear(last_channel, 512)
                l2 = nn.Linear(512, 128)
                l3 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3  
                    )
                
        
        if last_channel >=1024:
            if classifier_mode == "mlp_very_large":
                l1 = nn.Linear(last_channel, 1024)
                l2 = nn.Linear(1024, 512)
                l3 = nn.Linear(512, 256)
                l4 = nn.Linear(256, 128)
                l5 = nn.Linear(128, 64)
                l6 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l5,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l6
                    )
            elif classifier_mode == "mlp_large":
                l1 = nn.Linear(last_channel, 1024)
                l2 = nn.Linear(1024, 512)
                l3 = nn.Linear(512, 128)
                l4 = nn.Linear(128, 64)
                l5 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l5
                    )
            elif classifier_mode == "mlp_medium":
                l1 = nn.Linear(last_channel, 1024)
                l2 = nn.Linear(1024, 512)
                l3 = nn.Linear(512, 128)
                l4 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4
                    )
            elif classifier_mode == "mlp_small":
                l1 = nn.Linear(last_channel, 1024)
                l2 = nn.Linear(1024, 512)
                l3 = nn.Linear(512, 128)
                l4 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(1024),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4
                    )
            elif classifier_mode == "mlp_very_small":
                l1 = nn.Linear(last_channel, 512)
                l2 = nn.Linear(512, 128)
                l3 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3
                )

        elif last_channel >= 512:
            if classifier_mode == "mlp_very_large":
                l1 = nn.Linear(last_channel, 512)
                l2 = nn.Linear(512, 256)
                l3 = nn.Linear(256, 128)
                l4 = nn.Linear(128, 64)
                l5 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l5
                    )
            elif classifier_mode == "mlp_large":
                l1 = nn.Linear(last_channel, 512)
                l2 = nn.Linear(512, 256)
                l3 = nn.Linear(256, 128)
                l4 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4
                )
            elif classifier_mode == "mlp_medium":
                l1 = nn.Linear(last_channel, 512)
                l2 = nn.Linear(512, 128)
                l3 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3
                )
            elif classifier_mode == "mlp_small":
                l1 = nn.Linear(last_channel, 256)
                l2 = nn.Linear(256, 128)
                l3 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3
                )
            elif classifier_mode == "mlp_very_small":
                l1 = nn.Linear(last_channel, 128)
                l2 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2
                )


        elif last_channel >= 256:
            if classifier_mode == "mlp_very_very_large":
                l1 = nn.Linear(last_channel, 256)
                l2 = nn.Linear(256, 128)
                l3 = nn.Linear(128, 64)
                l4 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l4
                    )
            elif classifier_mode == "mlp_very_large":
                l1 = nn.Linear(last_channel, 256)
                l2 = nn.Linear(256, 128)
                l3 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3)
            elif classifier_mode == "mlp_large":
                l1 = nn.Linear(last_channel, 128)
                l2 = nn.Linear(128, 64)
                l3 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3)
            elif classifier_mode == "mlp_medium":
                l1 = nn.Linear(last_channel, 128)
                l2 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2)
            elif classifier_mode == "mlp_small":
                l1 = nn.Linear(last_channel, 64)
                l2 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2)
            elif classifier_mode == "mlp_very_small":
                l1 = nn.Linear(last_channel, num_cls)
                self.classifier = nn.Sequential(
                    l1)

        elif last_channel >= 128:
            if classifier_mode == "mlp_very_large":
                l1 = nn.Linear(last_channel, 128)
                l2 = nn.Linear(128, 64)
                l3 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l3
                    )
            elif classifier_mode == "mlp_medium":
                l1 = nn.Linear(last_channel, 128)
                l2 = nn.Linear(128, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(128),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2
                    )
            elif classifier_mode == "mlp_small":
                l1 = nn.Linear(last_channel, 64)
                l2 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2
                    )
            elif classifier_mode == "mlp_very_small":
                l1 = nn.Linear(last_channel, num_cls)
                self.classifier = nn.Sequential(
                    l1
                    )


        elif last_channel >= 64:
            if classifier_mode == "mlp_large":
                l1 = nn.Linear(last_channel, 64)
                l2 = nn.Linear(64, num_cls)
                self.classifier = nn.Sequential(
                    l1,
                    nn.BatchNorm1d(64),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    l2
                    )
            elif classifier_mode == "mlp_medium":
                l1 = nn.Linear(last_channel, num_cls)
                self.classifier = nn.Sequential(
                    l1
                    )
                
        else:
            raise Exception(f"classifier_mode!!! {classifier_mode} - last_channel: {last_channel}")
            
    def forward(self, x):
        return self.classifier(x)
    

########################### Segmentation #################################
import numpy as np
import torch
import torch.nn.functional as F


def cal_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous().view(-1) # gold is the groudtruth label in the dataloader

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)  # the number of feature_dim of the ouput, which is output channels

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')

    return loss


# create a file and write the text into it:
class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda(non_blocking=True)
    return new_y


def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred = pred.max(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

            F = np.sum(target_np[shape_idx] == part)

            if F != 0:
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]