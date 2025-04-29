import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

import sys
sys.path.append("/home/anil/Desktop/saeid/code/mine/SLNet/data")

from data.pointhop import *


def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s  --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    #download()
    #DATA_DIR = os.path.join(BASE_DIR, 'data')
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = "/home/anil/Desktop/saeid/datasets"
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        # print(f"h5_name: {h5_name}")
        f = h5py.File(h5_name,'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)

    return all_data, all_label


def pointhop_feature():
    train_data, train_label = load_data('train')
    valid_data, valid_label = load_data('train')

    feat_train = []
    feat_valid = []

    final_feature_train, feature_train, pca_params = pointhop_train(train_data, n_batch=20,
                                                                    n_newpoint=[1024,1024,1024,1024],   # [1024,128,128,64]
                                                                    n_sample=[4,4,4,4],             # [64,64,64,64],
                                                                    layer_num=[9,11,13,15], energy_percent=None)
    
    final_feature_valid, feature_valid = pointhop_pred(valid_data, n_batch=1, pca_params=pca_params,
                                                       n_newpoint=[1024,1024,1024,1024],                # [1024,128,128,64]
                                                       n_sample=[4,4,4,4],                          # [64,64,64,64],
                                                       layer_num=[9,11,13,15], idx_save=None, new_xyz_save=None)
    
    feature_train = extract(feature_train)
    feature_valid = extract(feature_valid)
    feat_train.append(feature_train)
    feat_valid.append(feature_valid)

    feat_train = np.concatenate(feat_train, axis=-1)
    feat_valid = np.concatenate(feat_valid, axis=-1)

    return feat_train, feat_valid


def random_point_dropout(pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    # for b in range(batch_pc.shape[0]):
    dropout_ratio = np.random.random()*max_dropout_ratio # 0~0.875    
    drop_idx = np.where(np.random.random((pc.shape[0]))<=dropout_ratio)[0]
    # print ('use random drop', len(drop_idx))

    if len(drop_idx)>0:
        pc[drop_idx,:] = pc[0,:] # set to the first point
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


class ModelNet40_pointhop(Dataset):
    def __init__(self, num_points, partition='train', feature=None, out_ch=16):
        self.data, self.label = load_data(partition)
        self.feature = feature
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        feature = self.feature[item][:self.num_points]
        label = self.label[item]
        
        if self.partition == 'train':
            # pointcloud = random_point_dropout(pointcloud) # open for dgcnn not for our idea  for all
            pointcloud = translate_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, feature, label
        

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40_pointhop(1024)
    #test = ModelNet40_pointhop(1024, 'test')
    for data, feature, label in train:
        print(data.shape)
        print(label.shape)
        break

    '''
    from torch.utils.data import DataLoader
    train_loader = DataLoader(ModelNet40_pointhop(partition='train', num_points=1024), num_workers=4,
                              batch_size=32, shuffle=True, drop_last=True)
    for batch_idx, (data, label) in enumerate(train_loader):
        print(f"batch_idx: {batch_idx}  | data shape: {data.shape} | ;lable shape: {label.shape}")

    train_set = ModelNet40_pointhop(partition='train', num_points=1024)
    test_set = ModelNet40_pointhop(partition='test', num_points=1024)
    print(f"train_set size {train_set.__len__()}")
    print(f"test_set size {test_set.__len__()}")
    '''