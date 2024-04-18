import json
import os
import pickle
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.utils.data import TensorDataset  
from torch.utils.data import DataLoader
from scipy.spatial.distance import mahalanobis, cosine
from autotune.tuner import DBTuner
from autotune.utils.history_container import HistoryContainer
from autotune.utils.config_space.util import convert_configurations_to_array
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import r2_score

# 定义常量 
KNOBS_INFO = "scripts/experiment/gen_knobs/postgres_new.json"
MAX_MIN_SCALER = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
NUMERIC_METRICS = [  # counter
    # global
    "buffers_alloc",
    "buffers_backend",
    "buffers_backend_fsync",
    "buffers_checkpoint",
    "buffers_clean",
    "checkpoints_req",
    "checkpoints_timed",
    "checkpoint_sync_time",
    "checkpoint_write_time",
    "maxwritten_clean",
    "archived_count",
    "failed_count",
    # db
    "blk_read_time",
    "blks_hit",
    "blks_read",
    "blk_write_time",
    "conflicts",
    "deadlocks",
    "temp_bytes",
    "temp_files",
    "tup_deleted",
    "tup_fetched",
    "tup_inserted",
    "tup_returned",
    "tup_updated",
    "xact_commit",
    "xact_rollback",
    "confl_tablespace",
    "confl_lock",
    "confl_snapshot",
    "confl_bufferpin",
    "confl_deadlock",
    # table
    "analyze_count",
    "autoanalyze_count",
    "autovacuum_count",
    "heap_blks_hit",
    "heap_blks_read",
    "idx_blks_hit",
    "idx_blks_read",
    "idx_scan",
    "idx_tup_fetch",
    "n_dead_tup",
    "n_live_tup",
    "n_tup_del",
    "n_tup_hot_upd",
    "n_tup_ins",
    "n_tup_upd",
    "n_mod_since_analyze",
    "seq_scan",
    "seq_tup_read",
    "tidx_blks_hit",
    "tidx_blks_read",
    "toast_blks_hit",
    "toast_blks_read",
    "vacuum_count",
    # index
    "idx_blks_hit",
    "idx_blks_read",
    "idx_scan",
    "idx_tup_fetch",
    "idx_tup_read",
]
OPERATOR_INFO = 'distance/new_model/workload_feature.json'


    
def extract_from_json(path, save=False):
    history_list = []
    dist = {}
    if not os.path.isdir(path):
        history_list.append(path)
    else:
        history_list = [f for f in os.listdir(path) if "history_task" in f]

    # constants
    config_space = DBTuner.setup_configuration_space(KNOBS_INFO, -1)
    for history in history_list:
        tmp = {}
        history_container = HistoryContainer(0, config_space=config_space)
        history_container.load_history_from_json(
            os.path.join(path, history) if os.path.isdir(path) else path
        )
        
        X, y, internal_metrics, resource = extract_data(history_container)
        y = y.values.flatten()
        
        filename = history.replace('history_task_SMAC_', '').replace('.json', '')
        model = RandomForestRegressor(n_estimators=100)
        model.fit(X, y)
        
        # 存储模型（先存储，后读取）
        # with open('distance/new_model/' + filename + '.model','wb') as file:
        #     pickle.dump(model,file)
        #     print('成功存储' + filename + '模型')
        
        # 读取模型获取一致对比例来计算真实距离
        for exp in history_list:
            if exp == history:
                continue
            exp = exp.replace('history_task_SMAC_', '').replace('.json', '')
            with open('distance/new_model/' + exp + '.model','rb') as file:
                model = pickle.load(file)
                y_pred = model.predict(X)
                tmp[exp] = concordant_pairs(y, y_pred)
        dist[filename] = tmp
        tmp = {}
    with open('distance/new_model/dist.json', 'w') as file:
        json.dump(dist, file, indent=2)            
        
        
def extract_data(history_container):
    X = convert_configurations_to_array(history_container.configurations)
    y = history_container.get_transformed_perfs()
    resource = history_container.resource
    internal_metrics = history_container.internal_metrics

    # drop invalid data
    drop_indexes = set()
    for i in range(len(X)):
        if len(internal_metrics[i]) == 0:
            drop_indexes.add(i)
    X = pd.DataFrame([item for idx, item in enumerate(X) if idx not in drop_indexes])
    y = pd.DataFrame([item for idx, item in enumerate(y) if idx not in drop_indexes])
    y = y.apply(MAX_MIN_SCALER)
    resource_metric = resource[0].keys()
    resource = pd.DataFrame(
        [
            [item[k] for k in resource_metric]
            for idx, item in enumerate(resource)
            if idx not in drop_indexes
        ]
    )
    internal_metrics = pd.DataFrame(
        [item for idx, item in enumerate(internal_metrics) if idx not in drop_indexes]
    )

    X_col = [e.name for e in history_container.config_space.get_hyperparameters()]
    y_col = history_container.info["objs"]
    internal_metrics_col = NUMERIC_METRICS
    resource_col = [k for k in resource_metric]

    assert (
        len(X_col) == len(X.columns)
        and len(y_col) == len(y.columns)
        and len(resource_col) == len(resource.columns)
        and len(internal_metrics_col) == len(internal_metrics.columns)
    )

    X.columns = X_col
    y.columns = y_col
    internal_metrics.columns = internal_metrics_col
    resource.columns = resource_col

    return X, y, internal_metrics, resource

# 定义一致对
def concordant_pairs(y, y_pred):
    pairs = 0
    for i in range(len(y)):
        for j in range(len(y_pred)):
            if (y[i] >= y[j] and y_pred[i] >= y_pred[j]) or (y[i] <= y[j] and y_pred[i] <= y_pred[j]):
                pairs += 1
    return pairs / (len(y) * len(y_pred))

def keys(data):
    suid_keys = ['INSERT', 'SELECT', 'DELETE', 'UPDATE']
    opts_keys = []
    for data_keys in data.values():
        opts = data_keys['OPTS']
        opts_keys = set(opts.keys()) | set(opts_keys)
    return suid_keys, opts_keys

# 获取模型所需的数据，真实距离、欧式、马氏、余弦距离
def model_data(data):
    with open('distance/new_model/dist.json', 'r') as file:
        dist = json.load(file)
    suid_keys, opt_keys = keys(data)
    exp_eu = []
    exp_ma = []
    exp_co = []
    exp_real = []
    exp_feature = []
    for name_i,feature_i in data.items():
        name_i = name_i.replace('task_SMAC_', '')
        feature_i_v = np.array([feature_i['SUID'][x] for x in suid_keys] + [feature_i['OPTS'].get(x, 0) for x in opt_keys])
        for name_j, feature_j in data.items():
            name_j = name_j.replace('task_SMAC_', '')
            feature_j_v = np.array([feature_j['SUID'][x] for x in suid_keys] + [feature_j['OPTS'].get(x, 0) for x in opt_keys])
            feature_comb = np.hstack((feature_i_v, feature_j_v))
            if name_i == name_j:
                real_dist = 1
            else:
                real_dist = dist[name_i][name_j]
                
            # Euclidean
            eu_dist = np.linalg.norm(feature_i_v - feature_j_v)
            
            # Mahalanobis
            cov_matrix = np.cov(np.vstack((feature_i_v, feature_j_v)), rowvar=False)
            inverse = np.linalg.inv(cov_matrix + np.eye(cov_matrix.shape[0]) * 1e-5)
            ma_dist = mahalanobis(feature_i_v, feature_j_v, inverse)
             
            # Cosine
            co_dist = cosine(feature_i_v, feature_j_v)
            
            exp_eu.append(eu_dist)
            exp_ma.append(ma_dist)
            exp_co.append(co_dist)
            exp_real.append(real_dist)
            exp_feature.append(feature_comb)
    
    df = pd.DataFrame(exp_feature)
    exp_real = np.array(exp_real)
    df['Real_Distance'] = exp_real
    df['Euclidean'] = 1 / normalize_to_1_100(np.array(exp_eu))
    df['Mahalanobis'] = 1 / normalize_to_1_100(np.array(exp_ma))
    df['Cosine'] = 1 / normalize_to_1_100(np.array(exp_co))
    df.to_csv('distance/new_model/data_new.csv', index=False)

def normalize_to_1_100(data):
    min_val = np.min(data)
    max_val = np.max(data)
    normalized_data = 1 + ((data - min_val) * 99) / (max_val - min_val)
    return normalized_data

# 随机森林模型
def random_forest_model():
    df = pd.read_csv('distance/new_model/data_new.csv')
    X = df.drop(columns=['Real_Distance'], axis=1)
    y = df['Real_Distance']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model_rf = RandomForestRegressor(n_estimators=100, max_depth=10)
    model_rf.fit(X_train, y_train)
    y_pred = model_rf.predict(X_test)
    score_rf = r2_score(y_test, y_pred)
    print(f"随机森林模型评分: {score_rf:.2f}")

def mlp_model():
    df = pd.read_csv('distance/new_model/data_new.csv')
    X_data = df.drop(columns=['Real_Distance'], axis=1).values
    Y_data = df['Real_Distance'].values.reshape(-1, 1)
    X = torch.from_numpy(X_data).type(torch.float32)
    Y = torch.from_numpy(Y_data).type(torch.float32)
    train_x, test_x, train_y, test_y = train_test_split(X_data, Y_data)
    train_x = torch.from_numpy(train_x).type(torch.float32)
    test_x = torch.from_numpy(test_x).type(torch.float32)
    train_y = torch.from_numpy(train_y).type(torch.float32)
    test_y = torch.from_numpy(test_y).type(torch.float32)
    
    model, optim = get_model(len(X_data))
    loss_fn = nn.MSELoss()
    batch = 64
    epochs = 100
    train_ds = TensorDataset(train_x, train_y)
    train_dl = DataLoader(train_ds, batch_size=batch, shuffle=True)
    test_ds = TensorDataset(test_x, test_y)
    test_dl = DataLoader(test_ds, batch_size=batch)
    for epoch in range(epochs):
        for x, y in train_dl:
            y_pred = model(x)
            loss = loss_fn(y_pred,y)
            optim.zero_grad()
            loss.backward()
            optim.step()
        with torch.no_grad():
            epoch_loss = loss_fn(model(train_x), train_y)
            epoch_test_loss = loss_fn(model(test_x), test_y)
            print('epoch: ', epoch, 
                  'loss: ', round(epoch_loss.item(), 3),
            'test_loss: ', round(epoch_test_loss.item(), 3))
    
    # 在测试集上进行预测并评估模型
    with torch.no_grad():
        y_pred_mlp = model(test_x)
        score_mlp = r2_score(test_y, y_pred_mlp.numpy())
        print(f"pytorch MLP模型评分: {score_mlp:.2f}")

class MLP(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.liner_1 = nn.Linear(57, 256)
        self.liner_2 = nn.Linear(256, 512)
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=1)
        self.liner_3 = nn.Linear(512, 1)
    def forward(self, input):
        x = F.relu(self.liner_1(input))
        x = F.relu(self.liner_2(x))
        attention_output, _ = self.attention(x, x, x)
        x = x + attention_output
        x = self.liner_3(x)
        return x

def get_model(embed_dim):
    model = MLP(embed_dim)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    return model, opt
    
# 其他距离公式以及绘图 
def distance():
    df = pd.read_csv('distance/new_model/data_new.csv')
    y_real = df['Real_Distance']
    y_eu = df['Euclidean']
    y_ma = df['Mahalanobis']
    y_co = df['Cosine']
    y = {'Real_Distance' : y_real, 'Euclidean' : y_eu, 'Mahalanobis' : y_ma, 'Cosine' : y_co}

    # 计算 concordant pairs
    eu_accuracy = concordant_pairs(y_eu, y_real)
    ma_accuracy = concordant_pairs(y_ma, y_real)
    co_accuracy = concordant_pairs(y_co, y_real)
    
    # 打印准确度
    print('Euclidean距离公式准确度: ', eu_accuracy)
    print('Mahalanobis距离公式准确度: ', ma_accuracy)
    print('Cosine距离公式准确度: ', co_accuracy)

    # 绘图
    plt.figure(figsize=(10, 6))
    plt.plot(y_real, label='Real Distance', marker=None)
    for y_key, y_value in y.items():
        plt.plot(y_value, label=y_key, marker=None)
        plt.xlabel('Sample Index')
        plt.ylabel('Distance')
        plt.title('Distance Comparison')
        plt.legend()
        plt.savefig('distance/new_model/' + y_key + '.png')   
    
if __name__=='__main__':
    with open (OPERATOR_INFO, 'r') as file:
        data = json.load(file)    
    extract_from_json("DBTune_history",save=False)
    model_data(data)
    # distance()
    random_forest_model()
    mlp_model()