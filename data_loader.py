import numpy as np
import scipy.sparse as sp
import torch, os
from configs import train_configs, data_configs
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from transfer_entropy import TE
from pprint import pprint
import collections
from tqdm import tqdm
import pickle
def normalize_adj(mx):
    # Row-normalize sparse matrix
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def split_by_window(data, n_features, n_pred):
    # data: pd.DataFrame, columns: [Time, KQI, KPI1, KPI2, ...]
    # window_size: int
    # return: splited samples, [n_nodes, window_size+1], 1 for node index
    sample_num = len(data)
    start = 0
    window_size = n_features + n_pred
    end = start + window_size
    samples = []
    while end <= sample_num:
        tmp = data.iloc[start:end, 1:].values.transpose() # [n_nodes, window_size]
        node_idx = np.arange(tmp.shape[0]) # asign a node index to each node
        tmp = np.concatenate((node_idx[:, np.newaxis], tmp), axis=1) # [n_nodes, window_size+1]
        samples.append(tmp) # except Time column, [n_nodes, window_size+1]
        start += 1
        end = start + window_size
    samples = np.array(samples) # [n_batch, n_nodes, window_size+1] 
    return samples

def construct_graph_by_data(data:pd.DataFrame):
    # data: pd.DataFrame, columns: [Time, KQI, KPI1, KPI2, ...]
    # return: pd.DataFrame, columns: [node1, node2]
    data = data.iloc[:, 1:] # remove Time column
    columns = data.columns.to_list()

    node_idx = np.arange(len(columns))
    node2idx = {node: idx for idx, node in zip(node_idx, columns)}
    idx2node = {idx: node for idx, node in zip(node_idx, columns)}
    delay = data_configs.transfer_entropy_delay

    c = data.iloc[:, 0].values # KQI, [data_len]
    S = data.iloc[:, 1:].values # KPIs, [data_len, n_kpi]

    n = S.shape[-1]
    edges = set()
    te = {} # record computed te
    # stage 1
    pbar = tqdm(total=n*n, desc='computing transfer entropy, stage 1')
    for i in range(n):
        for j in range(n):
            t_i2j = TE(S[:, i], S[:, j], delay=delay)
            te[f'{i+1}-{j+1}'] = t_i2j
            if t_i2j >= 1:
                edges.add((i+1, j+1))
            t_i2c = TE(S[:, i], c, delay=delay)
            te[f'{i+1}-0'] = t_i2c
            if t_i2c >= 1:
                edges.add((i+1, 0))
            pbar.update(1)
    # stage 2
    pbar = tqdm(total=n*n, desc='computing transfer entropy, stage 2')
    for i in range(n):
        for j in range(n):
            if te[f'{i+1}-0'] >= 1 and te[f'{j+1}-0'] >= 1 and te[f'{i+1}-{j+1}'] >=1:
                edges.pop((i+1, 0))
            pbar.update(1)
    # edges stored in set edges
    node1s = []
    node2s = []
    for node1, node2 in edges:
        node1s.append(node1)
        node2s.append(node2)
    output = pd.DataFrame({'NodeA': node1s, 'NodeB': node2s})
    output.sort_values(by=['NodeA', 'NodeB'], axis=0, inplace=True)
    output = output.values # [n_edges, 2]

    # sort by key
    idx2node = collections.OrderedDict(sorted(idx2node.items()))
    # sort by value
    node2idx = collections.OrderedDict(sorted(node2idx.items(), key=lambda item: item[1]))
    return output, node2idx, idx2node


def construct_graph():
    '''
    读入一组小区的多维KPI，按照时间分割为训练集、验证集和测试集
    input data: csv file, columns: [Time, KPI1, KPI2, ...]
    '''
    n_features = train_configs.n_features
    n_pred = train_configs.n_pred

    data_file = data_configs.data_file
    data = pd.read_csv(data_file)
    sample_num = len(data)
    train_num = int(sample_num * data_configs.train_ratio)
    val_num = int(sample_num * data_configs.val_ratio)

    # get graph
    adj, node2idx, idx2node = construct_graph_by_data(data)

    # normalize data
    data_values = data.iloc[:, 1:].values # [data_len, n_kpi]
    scaler = MinMaxScaler()
    normalize_data = scaler.fit_transform(data_values) # [data_len, n_kpi]
    data.iloc[:, 1:] = normalize_data

    train_data = data.iloc[0:train_num, :]
    val_data = data.iloc[train_num:(train_num+val_num), :]
    test_data = data.iloc[(train_num+val_num):, :]
    print(f'original data points - train: {len(train_data)}, val: {len(val_data)}, test: {len(test_data)}')
    print(f'train ratio: {data_configs.train_ratio}, val ratio: {data_configs.val_ratio}, test ratio: {data_configs.test_ratio}')
    
    # split by window
    train_samples = split_by_window(train_data, n_features, n_pred) # [n_batch, n_nodes, window_size+1]
    val_samples = split_by_window(val_data, n_features, n_pred)
    test_samples = split_by_window(test_data, n_features, n_pred)

    # save to file
    path = os.path.join(train_configs.path, train_configs.dataset)
    dataset_name = train_configs.dataset

    # saving processed
    f_train = open(os.path.join(path, f'{dataset_name}.train.nodes'), 'wb')
    pickle.dump(train_samples, f_train)
    f_train.close()
    f_val = open(os.path.join(path, f'{dataset_name}.val.nodes'), 'wb')
    pickle.dump(val_samples, f_val)
    f_val.close()
    f_test = open(os.path.join(path, f'{dataset_name}.test.nodes'), 'wb')
    pickle.dump(test_samples, f_test)
    f_test.close()
    f_adj = open(os.path.join(path, f'{dataset_name}.edges'), 'wb')
    pickle.dump(adj, f_adj)
    f_adj.close()

    # node dict
    f = open(os.path.join(path, 'node2idx.txt'), 'w')
    pprint(node2idx, stream=f)
    f.close()
    f = open(os.path.join(path, 'idx2node.txt'), 'w')
    pprint(idx2node, stream=f)
    f.close()
    print('data have all processed done!')

    # save scaler
    f = open(os.path.join(path, 'scaler.pkl'), 'wb')
    pickle.dump(scaler, f)
    f.close()


def load_data():
    # read train val and test dataset, containing features, labels, adj
    path = os.path.join(train_configs.path, train_configs.dataset)
    dataset_name = train_configs.dataset
    outputs = {}
    for mode in ['train', 'val', 'test']:
        node_file = os.path.join(path, f'{dataset_name}.{mode}.nodes')
        edge_file = os.path.join(path, f'{dataset_name}.edges')
        if os.path.exists(node_file) is False or os.path.exists(edge_file) is False:
            # process dataset
            construct_graph()

        print(f'Loading {dataset_name} {mode} dataset...')
        n_features = train_configs.n_features
        n_pred = train_configs.n_pred
        print(f'input_dim: {n_features}, output_dim: {n_pred}')
        scaler = pickle.load(open(os.path.join(path, 'scaler.pkl'), 'rb'))

        input_all = pickle.load(open(node_file, 'rb')) # [n_batch, n_nodes, window_size+1]
        # build graph
        idx = np.array(input_all[0][:, 0], dtype=np.int32)
        idx_map = {j: i for i, j in enumerate(idx)} 
        edges_unordered = pickle.load(open(edge_file, 'rb')) # [n_edges, 2]
        edges_unordered = edges_unordered.astype(np.int32)
        edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(idx.shape[0], idx.shape[0]), dtype=np.float32)

        # build symmetric adjacency matrix
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0])) # [n_node, n_node]
        adj = np.array(adj.todense(), dtype=np.int32)

        
        batch_features = []
        batch_labels = []
        batch_adjs = []
        for batch_idx in range(len(input_all)):
            input = input_all[batch_idx]
            ts = np.array(input[:, 1:], dtype=np.float32) # node index
            features = ts[:, 0:n_features]
            labels = ts[:, n_features:(n_features+n_pred)]
            batch_features.append(features)
            batch_labels.append(labels)
            batch_adjs.append(adj)

        outputs[mode] = {}
        outputs[mode]['adj'] = torch.LongTensor(batch_adjs) # [n_batch, n_nodes, n_nodes]
        outputs[mode]['features'] = torch.FloatTensor(batch_features) # [n_batch, n_nodes, n_feature]
        outputs[mode]['labels'] = torch.FloatTensor(batch_labels) # [n_batch, n_nodes, n_pred]
        outputs[mode]['scaler'] = scaler
    return outputs

def evaluate(preds, labels, scaler):
    # metrics of mse, rmse, mae, mape
    # preds: [n_sample, n_node, n_pred]
    # labels: [n_sample, n_node, n_pred]
    # scaler: MinMaxScaler
    preds = preds.transpose(0, 2, 1) # [n_sample, n_pred, n_node]
    labels = labels.transpose(0, 2, 1)
    preds = np.concatenate([preds[i] for i in range(preds.shape[0])], axis=0) # [n_sample * n_pred, n_node]
    labels = np.concatenate([labels[i] for i in range(labels.shape[0])], axis=0)

    preds = scaler.inverse_transform(preds)
    labels = scaler.inverse_transform(preds)
    metrics = {}
    metrics['mse'] = np.mean(np.square(preds - labels))
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = np.mean(np.abs(preds - labels))
    metrics['mape'] = np.mean(np.abs((preds - labels) / labels))
    # compute metric on each columns
    for i in range(preds.shape[1]):
        metrics[f'mse_{i}'] = np.mean(np.square(preds[:, i] - labels[:, i]))
        metrics[f'rmse_{i}'] = np.sqrt(metrics[f'mse_{i}'])
        metrics[f'mae_{i}'] = np.mean(np.abs(preds[:, i] - labels[:, i]))
        metrics[f'mape_{i}'] = np.mean(np.abs((preds[:, i] - labels[:, i]) / labels[:, i]))
    return metrics