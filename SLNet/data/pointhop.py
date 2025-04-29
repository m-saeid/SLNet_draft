import os
# import h5py
from sklearn.decomposition import PCA, IncrementalPCA
import threading
import numpy as np
from sklearn.model_selection import train_test_split

from numpy import linalg as LA



############## PointHop ############## >

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = np.tile(np.arange(B).reshape(view_shape),repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def load_dir_pointhop(data_dir, name='train_files.txt'):
    with open(os.path.join(data_dir,name),'r') as f:
        lines = f.readlines()
    return [os.path.join(data_dir, line.rstrip().split('/')[-1]) for line in lines]

'''
def data_load_pointhop(num_point=None, data_dir='F:/Datasets/modelnet40_ply_hdf5_2048', train=True):

    if train:
        data_pth = load_dir_pointhop(data_dir, name='train_files.txt')
    else:
        data_pth = load_dir_pointhop(data_dir, name='test_files.txt')

    point_list = []
    label_list = []
    for pth in data_pth:
        data_file = h5py.File(pth, 'r')
        point = data_file['data'][:]
        label = data_file['label'][:]
        point_list.append(point)
        label_list.append(label)
    data = np.concatenate(point_list, axis=0)
    label = np.concatenate(label_list, axis=0)
    # data, idx = shuffle_data(data)
    # data, ind = shuffle_points(data)

    if not num_point:
        return data[:, :, :], label
    else:
        return data[:, :num_point, :], label
'''

def data_separate_pointhop(data, label):
    seed = 7
    np.random.seed(seed)
    train_data, valid_data, train_label, valid_label = train_test_split(data, label, test_size=0.1, random_state=seed)

    return train_data, train_label, valid_data, valid_label


def remove_mean(features, axis):
    '''
    Remove the dataset mean.
    :param features [num_samples,...]
    :param axis the axis to compute mean
    
    '''
    feature_mean = np.mean(features, axis=axis, keepdims=True)
    feature_remove_mean = features-feature_mean
    return feature_remove_mean, feature_mean


def remove_zero_patch(samples):
    std_var = (np.std(samples, axis=1)).reshape(-1, 1)
    ind_bool = (std_var == 0)
    ind = np.where(ind_bool==True)[0]
    # print('zero patch shape:',ind.shape)
    samples_new = np.delete(samples, ind, 0)
    return samples_new


def find_kernels_pca(sample_patches, num_kernels, energy_percent, n_batch):
    '''
    Do the PCA based on the provided samples.
    If num_kernels is not set, will use energy_percent.
    If neither is set, will preserve all kernels.
    :param samples: [num_samples, feature_dimension]
    :param num_kernels: num kernels to be preserved
    :param energy_percent: the percent of energy to be preserved
    :return: kernels, sample_mean
    '''
    # Remove patch mean
    sample_patches_centered, dc = remove_mean(sample_patches, axis=1)
    sample_patches_centered = remove_zero_patch(sample_patches_centered)
    # Remove feature mean (Set E(X)=0 for each dimension)
    training_data, feature_expectation = remove_mean(sample_patches_centered, axis=0)

    # pca = PCA(n_components=training_data.shape[1], svd_solver='full', whiten=True)
    batch_size = training_data.shape[0]//n_batch
    pca = IncrementalPCA(n_components=training_data.shape[1], whiten=True, batch_size=batch_size, copy=False)
    pca.fit(training_data)

    # Compute the number of kernels corresponding to preserved energy
    if energy_percent:
        energy = np.cumsum(pca.explained_variance_ratio_)
        num_components = np.sum(energy < energy_percent)+1
    else:
        num_components = num_kernels

    kernels = pca.components_[:num_components, :]
    mean = pca.mean_

    num_channels = sample_patches.shape[-1]
    largest_ev = [np.var(dc*np.sqrt(num_channels))]
    dc_kernel = 1/np.sqrt(num_channels)*np.ones((1, num_channels))/np.sqrt(largest_ev)
    kernels = np.concatenate((dc_kernel, kernels), axis=0)

    print("Num of kernels: %d" % num_components)
    print("Energy percent: %f" % np.cumsum(pca.explained_variance_ratio_)[num_components-1])
    return kernels, mean

def furthest_point_sample_pointhop(pts, K):
    """
    Input:
        pts: pointcloud data, [B, N, C]
        K: number of samples
    Return:
        (B, K, 3)
    """
    B, N, C = pts.shape
    centroids = np.zeros((B, K), dtype=int)
    distance = np.ones((B, N), dtype=int) * 1e10
    farthest = np.random.randint(0, N, (B,))
    batch_indices = np.arange(B)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = pts[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = np.sum((pts - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)
    return index_points(pts, centroids)

def calc_feature(pc_temp, pc_bin, pc_gather):
    value = np.multiply(pc_temp, pc_bin)
    value = np.sum(value, axis=2, keepdims=True)
    num = np.sum(pc_bin, axis=2, keepdims=True)
    final = np.squeeze(value/num)
    pc_gather.append(final)

def gather_fea(nn_idx, point_data, fea):
    """
    nn_idx:(B, n_sample, K)
    pts:(B, N, dim)
    :return: pc_n(B, K, dim_fea)
    """
    num_newpts = nn_idx.shape[2]
    assert point_data.shape[:-1] == fea.shape[:-1]
    pts_fea = np.concatenate([point_data, fea], axis=-1)
    num_dim = pts_fea.shape[2]

    pts_fea_expand = index_points(pts_fea, nn_idx)
    # print(pts_fea_expand.shape)
    pts_fea_expand = pts_fea_expand.transpose(0, 2, 1, 3)  # (B, K, n_sample, dim)
    pc_n = pts_fea_expand[..., :3]
    pc_temp = pts_fea_expand[..., 3:]

    pc_n_center = np.expand_dims(pc_n[:, :, 0, :], axis=2)
    pc_n_uncentered = pc_n - pc_n_center

    pc_idx = []
    pc_idx.append(pc_n_uncentered[:, :, :, 0] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 0] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 1] <= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] >= 0)
    pc_idx.append(pc_n_uncentered[:, :, :, 2] <= 0)

    pc_bin = []
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[0] * pc_idx[3] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[2] * pc_idx[5])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[4])*1.0, axis=3))
    pc_bin.append(np.expand_dims((pc_idx[1] * pc_idx[3] * pc_idx[5])*1.0, axis=3))

    pc_gather1 = []
    pc_gather2 = []
    pc_gather3 = []
    pc_gather4 = []
    pc_gather5 = []
    pc_gather6 = []
    pc_gather7 = []
    pc_gather8 = []
    threads = []
    t1 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[0], pc_gather1))
    threads.append(t1)
    t2 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[1], pc_gather2))
    threads.append(t2)
    t3 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[2], pc_gather3))
    threads.append(t3)
    t4 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[3], pc_gather4))
    threads.append(t4)
    t5 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[4], pc_gather5))
    threads.append(t5)
    t6 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[5], pc_gather6))
    threads.append(t6)
    t7 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[6], pc_gather7))
    threads.append(t7)
    t8 = threading.Thread(target=calc_feature, args=(pc_temp, pc_bin[7], pc_gather8))
    threads.append(t8)
    for t in threads:
        t.setDaemon(False)
        t.start()
    for t in threads:
        if t.is_alive():
            t.join()
    pc_gather = pc_gather1 + pc_gather2 + pc_gather3 + pc_gather4 + pc_gather5 + pc_gather6 + pc_gather7 + pc_gather8
    pc_fea = np.concatenate(pc_gather, axis=2)

    return pc_fea

def calc_distances(tmp, pts):
    '''

    :param tmp:(B, k, 3)/(B, 3)
    :param pts:(B, N, 3)
    :return:(B, N, k)/(B, N)
    '''
    if len(tmp.shape) == 2:
        tmp = np.expand_dims(tmp, axis=1)
    tmp_trans = np.transpose(tmp, [0,2,1])
    xy = np.matmul(pts, tmp_trans)
    pts_square = (pts**2).sum(axis=2, keepdims=True)
    tmp_square_trans = (tmp_trans**2).sum(axis=1, keepdims=True)
    return np.squeeze(pts_square + tmp_square_trans - 2 * xy)

def knn_query(new_pts, pts, n_sample, idx):
    '''
    new_pts:(B, K, 3)
    pts:(B, N, 3)
    n_sample:int
    :return: nn_idx (B, n_sample, K)
    '''
    distance_matrix = calc_distances(new_pts, pts)
    # nn_idx = np.argsort(distance_matrix, axis=1, kind='stable')[:, :n_sample, :]  # (B, n, K)
    nn_idx = np.argpartition(distance_matrix, (0, n_sample), axis=1)[:, :n_sample, :]
    idx.append(nn_idx)


def knn_pointhop(new_xyz, point_data, n_sample):
    idx1 = []
    idx2 = []
    idx3 = []
    idx4 = []
    idx5 = []
    idx6 = []
    idx7 = []
    idx8 = []
    threads = []
    batch_size = point_data.shape[0]//8
    t1 = threading.Thread(target=knn_query, args=(new_xyz[:batch_size], point_data[:batch_size], n_sample, idx1))
    threads.append(t1)
    t2 = threading.Thread(target=knn_query, args=(new_xyz[batch_size:2*batch_size], point_data[batch_size:2*batch_size], n_sample, idx2))
    threads.append(t2)
    t3 = threading.Thread(target=knn_query, args=(new_xyz[2*batch_size:3*batch_size], point_data[2*batch_size:3*batch_size], n_sample, idx3))
    threads.append(t3)
    t4 = threading.Thread(target=knn_query, args=(new_xyz[3*batch_size:4*batch_size], point_data[3*batch_size:4*batch_size], n_sample, idx4))
    threads.append(t4)
    t5 = threading.Thread(target=knn_query, args=(new_xyz[4*batch_size:5*batch_size], point_data[4*batch_size:5*batch_size], n_sample, idx5))
    threads.append(t5)
    t6 = threading.Thread(target=knn_query, args=(new_xyz[5*batch_size:6*batch_size], point_data[5*batch_size:6*batch_size], n_sample, idx6))
    threads.append(t6)
    t7 = threading.Thread(target=knn_query, args=(new_xyz[6*batch_size:7*batch_size], point_data[6*batch_size:7*batch_size], n_sample, idx7))
    threads.append(t7)
    t8 = threading.Thread(target=knn_query, args=(new_xyz[7*batch_size:], point_data[7*batch_size:], n_sample, idx8))
    threads.append(t8)

    for t in threads:
        t.setDaemon(False)
        t.start()
    for t in threads:
        if t.is_alive():
            t.join()
    idx = idx1 + idx2 + idx3 + idx4 + idx5 + idx6 + idx7 + idx8
    idx_tmp = np.concatenate(idx, axis=0)

    return idx_tmp

def extract(feat):
    '''
    Do feature extraction based on the provided feature.
    :param feat: [num_layer, num_samples, feature_dimension]
    # :param pooling: pooling method to be used
    :return: feature
    '''
    mean = []
    maxi = []
    l1 = []
    l2 = []

    for i in range(len(feat)):
        mean.append(feat[i].mean(axis=1, keepdims=False))
        maxi.append(feat[i].max(axis=1, keepdims=False))
        l1.append(np.linalg.norm(feat[i], ord=1, axis=1, keepdims=False))
        l2.append(np.linalg.norm(feat[i], ord=2, axis=1, keepdims=False))
    mean = np.concatenate(mean, axis=-1)
    maxi = np.concatenate(maxi, axis=-1)
    l1 = np.concatenate(l1, axis=-1)
    l2 = np.concatenate(l2, axis=-1)

    return [mean, maxi, l1, l2]

def query_and_gather(new_xyz, n_batch, batch_size, pts_coor, pts_fea, n_sample, pooling):
    idx = []
    grouped_feature = []
    for j in range(n_batch):
        if j != n_batch - 1:
            idx_tmp = knn_pointhop(new_xyz[j * batch_size:(j + 1) * batch_size],
                                      pts_coor[j * batch_size:(j + 1) * batch_size]
                                      , n_sample)
            grouped_feature_tmp = gather_fea(idx_tmp, pts_coor[j * batch_size:(j + 1) * batch_size],
                                                         pts_fea[j * batch_size:(j + 1) * batch_size])
        else:
            idx_tmp = knn_pointhop(new_xyz[j * batch_size:], pts_coor[j * batch_size:], n_sample)
            grouped_feature_tmp = gather_fea(idx_tmp, pts_coor[j * batch_size:],
                                                         pts_fea[j * batch_size:])
        if pooling is not None:
            grouped_feature_tmp = grouped_feature_tmp.reshape(grouped_feature_tmp.shape[0], grouped_feature_tmp.shape[1], 8, -1)
            grouped_feature_tmp = extract(grouped_feature_tmp, pooling, 2)
        idx.append(idx_tmp)
        grouped_feature.append(grouped_feature_tmp)
    idx = np.concatenate(idx, axis=0)
    grouped_feature = np.concatenate(grouped_feature, axis=0)
    return idx, grouped_feature


def furthest_point_sample(pts, K):
    """
    Input:
        pts: pointcloud data, [B, N, C]
        K: number of samples
    Return:
        (B, K, 3)
    """
    B, N, C = pts.shape
    centroids = np.zeros((B, K), dtype=int)
    distance = np.ones((B, N), dtype=int) * 1e10
    farthest = np.random.randint(0, N, (B,))
    batch_indices = np.arange(B)
    for i in range(K):
        centroids[:, i] = farthest
        centroid = pts[batch_indices, farthest, :].reshape(B, 1, 3)
        dist = np.sum((pts - centroid) ** 2, axis=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, axis=-1)
    return index_points(pts, centroids)


def pointhop_train(train_data, n_batch, n_newpoint, n_sample, layer_num, energy_percent):
             # 9840,16,3    args.num_batch_train         args.num_point            args.num_sample         args.num_filter               None
    '''
    Train based on the provided samples.
    :param train_data: [num_samples, num_point, feature_dimension]
    :param n_batch:                                       batch size
    :param n_newpoint: point numbers used in every stage: Point Number after down sampling
    :param n_sample: k nearest neighbors:                 KNN query number
    :param layer_num: num kernels to be preserved         Filter Number
    :param energy_percent: the percent of energy to be preserved
    
    :return:
    idx_save: knn index
    new_xyz_save: down sample index
    final_feature: 
    feature_train: 
    pca_params: pca kernel and mean
    '''

    num_data = train_data.shape[0]
    pca_params = {}
    # idx_save = {}
    # new_xyz_save = {}

    point_data = train_data
    batch_size = num_data//n_batch
    grouped_feature = None
    feature_train = []

    feature_data = train_data

    for i in range(len(n_newpoint)):
        print(i)
        point_num = point_data.shape[1]
        print('Start sampling-------------')
        if n_newpoint[i] == point_num:
            new_xyz = point_data
        else:
            new_xyz = furthest_point_sample_pointhop(point_data, n_newpoint[i])

        # new_xyz_save['Layer_{:d}'.format(i)] = new_xyz

        print('Start query and gathering-------------')
        # time_start = time.time()
        if not grouped_feature is None:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, grouped_feature, n_sample[i], None)
        else:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, feature_data, n_sample[i], None)

        # idx_save['Layer_%d' % (i)] = idx
        grouped_feature = grouped_feature.reshape(num_data*n_newpoint[i], -1)
        print('ok-------------')

        kernels, mean = find_kernels_pca(grouped_feature, layer_num[i], energy_percent, n_batch)
        # 3,24    24,          157440/78720/78720/39360,24    1,2,3,4        None          20
#num_filter+1,24  24,

        if i == 0:
            transformed = np.matmul(grouped_feature, np.transpose(kernels))
        else:
            bias = LA.norm(grouped_feature, axis=1)
            bias = np.max(bias)
            pca_params['Layer_{:d}/bias'.format(i)] = bias
            grouped_feature = grouped_feature + bias

            transformed = np.matmul(grouped_feature, np.transpose(kernels))
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias*e
        grouped_feature = transformed.reshape(num_data, n_newpoint[i], -1)
        print(grouped_feature.shape)
        feature_train.append(grouped_feature)
        pca_params['Layer_{:d}/kernel'.format(i)] = kernels
        pca_params['Layer_{:d}/pca_mean'.format(i)] = mean
        point_data = new_xyz
    final_feature = grouped_feature.max(axis=1, keepdims=False) # 9840,args.num_point[-1],args.num_filter+1 > 9840,args.num_filter+1

    # return idx_save, new_xyz_save, final_feature, feature_train, pca_params
    return final_feature, feature_train, pca_params




def pointhop_pred(test_data, n_batch, pca_params, n_newpoint, n_sample, layer_num, idx_save, new_xyz_save):
    '''
    Test based on the provided samples.
    :param test_data: [num_samples, num_point, feature_dimension]
    :param pca_params: pca kernel and mean
    :param n_newpoint: point numbers used in every stage
    :param n_sample: k nearest neighbors
    :param layer_num: num kernels to be preserved
    :param idx_save: knn index
    :param new_xyz_save: down sample index
    :return: final stage feature, feature, pca_params
    '''

    num_data = test_data.shape[0]
    point_data = test_data
    grouped_feature = None
    feature_test = []
    batch_size = num_data//n_batch

    feature_data = test_data

    for i in range(len(n_newpoint)):
        if not new_xyz_save:
            point_num = point_data.shape[1]
            if n_newpoint[i] == point_num:
                new_xyz = point_data
            else:
                new_xyz = furthest_point_sample(point_data, n_newpoint[i])
        else:
            print('---------------loading idx--------------')
            new_xyz = new_xyz_save['Layer_{:d}'.format(i)]

        if not grouped_feature is None:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, grouped_feature, n_sample[i], None)
        else:
            idx, grouped_feature = query_and_gather(new_xyz, n_batch, batch_size, point_data, feature_data, n_sample[i], None)

        grouped_feature = grouped_feature.reshape(num_data*n_newpoint[i], -1)

        kernels = pca_params['Layer_{:d}/kernel'.format(i)]
        mean = pca_params['Layer_{:d}/pca_mean'.format(i)]

        if i == 0:
            transformed = np.matmul(grouped_feature, np.transpose(kernels))
        else:
            bias = pca_params['Layer_{:d}/bias'.format(i)]
            grouped_feature = grouped_feature + bias
            transformed = np.matmul(grouped_feature, np.transpose(kernels))
            e = np.zeros((1, kernels.shape[0]))
            e[0, 0] = 1
            transformed -= bias*e
        grouped_feature = transformed.reshape(num_data, n_newpoint[i], -1)
        feature_test.append(grouped_feature)
        point_data = new_xyz
    final_feature = grouped_feature.max(axis=1, keepdims=False)
    return final_feature, feature_test



########### PointHup ########### <