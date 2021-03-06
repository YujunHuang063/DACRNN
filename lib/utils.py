import logging
import numpy as np
import os
import pickle
import scipy.sparse as sp
import sys
import tensorflow as tf
import pandas as pd

from scipy.sparse import linalg


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True, shuffle=False):
        """

        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        if shuffle:
            permutation = np.random.permutation(self.size)
            xs, ys = xs[permutation], ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler:
    """
    Standard the input
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        print('mean: ', mean)
        print('std: ', std)

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def add_simple_summary(writer, names, values, global_step):
    """
    Writes summary for a list of scalars.
    :param writer:
    :param names:
    :param values:
    :param global_step:
    :return:
    """
    for name, value in zip(names, values):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        writer.add_summary(summary, global_step)


def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_random_walk_matrix(adj_mx):
    adj_mx = sp.coo_matrix(adj_mx)
    d = np.array(adj_mx.sum(1))
    d_inv = np.power(d, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)
    random_walk_mx = d_mat_inv.dot(adj_mx).tocoo()
    return random_walk_mx


def calculate_reverse_random_walk_matrix(adj_mx):
    return calculate_random_walk_matrix(np.transpose(adj_mx))


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32)


def config_logging(log_dir, log_filename='info.log', level=logging.INFO):
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # Create the log directory if necessary.
    try:
        os.makedirs(log_dir)
    except OSError:
        pass
    file_handler = logging.FileHandler(os.path.join(log_dir, log_filename))
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level=level)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    console_handler.setLevel(level=level)
    logging.basicConfig(handlers=[file_handler, console_handler], level=level)


def get_logger(log_dir, name, log_filename='info.log', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    # Add file handler and stdout handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    log_path = os.path.join(log_dir, log_filename)
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    # Add console handler.
    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    # Add google cloud log handler
    logger.info('Log directory: %s', log_dir)
    return logger


def get_total_trainable_parameter_size():
    """
    Calculates the total number of trainable parameters in the current graph.
    :return:
    """
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        total_parameters += np.product([x.value for x in variable.get_shape()])
    return total_parameters


def load_dataset(dataset_dir, batch_size, input_dim, test_batch_size=None, **kwargs):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        ind_x = np.sum(np.reshape(cat_data['x']>100, [len(cat_data['x']), cat_data['x'].shape[1]*cat_data['x'].shape[2]*cat_data['x'].shape[3]]), 1)
        ind_y = np.sum(np.reshape(cat_data['y']>100, [len(cat_data['y']), cat_data['y'].shape[1]*cat_data['x'].shape[2]*cat_data['y'].shape[3]]), 1)
        mask = (ind_x*ind_y).astype(bool)
        ind = np.arange(len(cat_data['x']))
        ind = ind[~mask]
        if category=='train':
            x_max = np.max(cat_data['x'])
            y_max = np.max(cat_data['y'])
        data['x_' + category] = np.clip(cat_data['x'], 0, x_max)
        data['y_' + category] = np.clip(cat_data['y'], 0, x_max)
        
        print('max_data: ', np.max(cat_data['x']), np.max(cat_data['y']))
    print('train_max: ', np.max(np.clip(data['x_train'][..., 0], 0, x_max)), 'train_mean: ', np.mean(np.clip(data['x_train'][..., 0], 0, x_max)), ' ', data['x_train'][..., 0].mean())
    scaler = StandardScaler(mean=np.clip(data['x_train'][..., 0], 0, x_max).mean(), std=np.clip(data['x_train'][..., 0], 0, x_max).std())    
    #scaler = StandardScaler(mean=data['x_train'][..., 0].mean(), std=data['x_train'][..., 0].std())
    # Data format
    for category in ['train', 'val', 'test']:
        print('max: ', np.max(data['x_' + category][..., 0]))
        print('concat_max: ', np.max(data['x_' + category]))
        '''
        data['x_' + category][..., 0] = scaler.transform(data['x_' + category][..., 0])
        data['y_' + category][..., 0] = scaler.transform(data['y_' + category][..., 0])
        '''
        data['x_' + category][..., 0] = scaler.transform(np.clip(data['x_' + category][..., 0], 0, x_max))
        data['y_' + category][..., 0] = scaler.transform(np.clip(data['y_' + category][..., 0], 0, x_max))
        
    data['train_loader'] = DataLoader(np.clip(data['x_train'], 0, x_max), np.clip(data['y_train'], 0, x_max), batch_size, shuffle=True)
    data['val_loader'] = DataLoader(np.clip(data['x_val'], 0, x_max), np.clip(data['y_val'], 0, x_max), test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(np.clip(data['x_test'], 0, x_max), np.clip(data['y_test'], 0, x_max), test_batch_size, shuffle=False)
    '''
    data['train_loader'] = DataLoader(np.clip(data['x_train'][..., 0:input_dim], 0, x_max), np.clip(data['y_train'][..., 0:input_dim], 0, y_max), batch_size, shuffle=True)
    data['val_loader'] = DataLoader(np.clip(data['x_val'][..., 0:input_dim], 0, x_max), np.clip(data['y_val'][..., 0:input_dim], 0, y_max), test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(np.clip(data['x_test'][..., 0:input_dim], 0, x_max), np.clip(data['y_test'][..., 0:input_dim], 0, y_max), test_batch_size, shuffle=False)
    '''
    data['scaler'] = scaler

    return data

def load_dataset_from_csv(dataset_dir, batch_size, input_dim, test_batch_size=None, **kwargs):
    sz_tf = pd.read_csv(dataset_dir+'/sz_speed.csv')
    data1 =np.mat(sz_tf, dtype=np.float32)
    train_size = int(len(data1) * 0.8)
    train_data = data1[0:train_size]
    test_data = data1[train_size:len(data1)]
    
    trainX, trainY, testX, testY = [], [], [], []
    seq_len = 12
    pre_len = 12
    for i in range(len(train_data) - seq_len - pre_len):
        a = train_data[i: i + seq_len + pre_len]
        trainX.append(a[0 : seq_len])
        trainY.append(a[seq_len : seq_len + pre_len])
    for i in range(len(test_data) - seq_len -pre_len):
        b = test_data[i: i + seq_len + pre_len]
        testX.append(b[0 : seq_len])
        testY.append(b[seq_len : seq_len + pre_len])
    trainX1 = np.expand_dims(np.array(trainX), -1)
    trainY1 = np.expand_dims(np.array(trainY), -1)
    testX1 = np.expand_dims(np.array(testX), -1)
    testY1 = np.expand_dims(np.array(testY), -1)
    val_x = np.copy(testX1)
    val_y = np.copy(testY1)
    scaler = StandardScaler(mean=trainX1.mean(), std=trainX1.std())
    data = {}
    data['x_train'] = scaler.transform(trainX1)
    data['y_train'] = scaler.transform(trainY1)
    data['x_val'] = scaler.transform(val_x)
    data['y_val'] = scaler.transform(val_y)
    data['x_test'] = scaler.transform(testX1)
    data['y_test'] = scaler.transform(testY1)
    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, shuffle=True)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], test_batch_size, shuffle=False)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, shuffle=False)
    data['scaler'] = scaler
    return data
    
def load_graph_data_from_csv(data_dir):
    los_adj = pd.read_csv(data_dir+'/sz_adj.csv',header=None)
    adj = np.mat(los_adj)
    one = np.eye(adj.shape[0])
    adj = adj+one
    adj = adj.astype(np.float32)
    print('adj_shape: ', adj.shape)
    return adj

def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data
