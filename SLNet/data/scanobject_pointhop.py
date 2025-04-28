import os
import sys
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset


import sys
sys.path.append("/home/anil/Desktop/saeid/code/mine/SLNet/data")
from data.pointhop import *

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

import os

def download():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'data')
    H5_DIR = os.path.join(DATA_DIR, 'h5_files')
    
    expected_file = os.path.join(H5_DIR, "main_split", "test_objectdataset_augmentedrot_scale75.h5")
    
    if os.path.exists(expected_file):
        print("Dataset already downloaded. Skipping download.")
        return
    
    if not os.path.exists(H5_DIR):
        os.makedirs(H5_DIR)
    
    www = 'https://github.com/ma-xu/pointMLP-pytorch/releases/download/dataset/h5_files.zip'
    zipfile = os.path.basename(www)
    
    os.system(f'wget {www} --no-check-certificate')
    os.system(f'unzip {zipfile} -d {H5_DIR}')
    os.system(f'rm {zipfile}')
    os.system(f'rm -rf "{os.path.join(H5_DIR, "main_files")}"')
    os.system(f'rm -rf "{os.path.join(H5_DIR, "__MACOSX")}"')

    print("Download and extraction complete.")


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2. / 3., high=3. / 2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def load_scanobjectnn_data(partition):
    # download()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []

    h5_name = BASE_DIR + '/data/h5_files/main_split/' + partition + '_objectdataset_augmentedrot_scale75.h5'
    f = h5py.File(h5_name, mode="r")
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    all_data.append(data)
    all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def pointhop_feature():
    train_data, train_label = load_scanobjectnn_data('training')
    valid_data, valid_label = load_scanobjectnn_data('test')

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


class ScanObjectNN_pointhop(Dataset):
    def __init__(self, num_points, feature=None, partition='training'):
        self.data, self.label = load_scanobjectnn_data(partition)
        self.feature = feature
        self.num_points = num_points
        self.partition = partition

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        feature = self.feature[item][:self.num_points]
        label = self.label[item]

        if self.partition == 'training':
            pointcloud = translate_pointcloud(pointcloud)
            # np.random.shuffle(pointcloud)
        return pointcloud, feature, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ScanObjectNN_pointhop(1024)
    test = ScanObjectNN_pointhop(1024, 'test')
    for data, label in train:
        print(data.shape)
        print(label)
        break