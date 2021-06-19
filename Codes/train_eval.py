# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
from importlib import import_module
from utils import build_dataset, get_time_dif   #build_iterator, 
from sklearn.model_selection import train_test_split,KFold
# global predict_all,labels_all

import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


import gc
#计算迭代次数
def train_step_counts(config,len_data):
    train_step = len_data*config.num_epochs//config.batch_size
    return train_step
def Kfold_train_reg(config,train_data,test_data):
    model_name = config.model_name
    x = import_module('models.' + model_name)
    start_time = time.time()                   #每个模型运行时开始计时
    k_fold = KFold(config.k_folds, shuffle=True, random_state=2020)   #K折交叉验证
    dev_best_loss = float('inf')
    Rmse_collection = []
    Pearson_collection = []
    R2_collection = []
    labels_all_collection = []
    predict_all_collection = []
    train_labels_all_collection = []
    train_predict_all_collection = []
    #last_improve = 0  # 记录上次验证集loss下降的batch数
    #K折交叉验证，相当于把数据分成K分，训练出K个模型，分别测试K次
    for fold_index, (train_index, eval_index) in enumerate(k_fold.split(train_data)):
        print('Kfold [{}/{}]'.format(fold_index + 1, config.k_folds))
        train_model = x.Model(config).to(config.device)
        train_single = train_data[train_index]
        dev_single = train_data[eval_index]
        test_Rmse, test_Pearson, test_R2, time_dif,test_labels_all1,test_predict_all1,train_labels_all, train_predict_all = train_reg(config,train_model,train_single,dev_single,
                            test_data,
              start_time,fold_index)
        del train_model, train_single, dev_single  #每训练一个模型做完测试之后删掉。这样才能训练下一个模型
        Rmse_collection.append(test_Rmse)           #将改模型的效果值存下来
        Pearson_collection.append(test_Pearson)
        R2_collection.append(test_R2)
        labels_all_collection.append(test_labels_all1)
        predict_all_collection.append(test_predict_all1)
        train_labels_all_collection.append(train_labels_all)
        train_predict_all_collection.append(train_predict_all)
        mmm=dict(zip(R2_collection,labels_all_collection))
        nnn=dict(zip(R2_collection,predict_all_collection))
        mmm0=dict(zip(R2_collection,train_labels_all_collection))
        nnn0=dict(zip(R2_collection,train_predict_all_collection))
        gc.collect()
    Rmse_collection.sort()
    Pearson_collection.sort()
    R2_collection.sort()
    mmm=sorted(list(mmm.items()),key=lambda x:x[0]);mmm1=dict(mmm)
    g=list(mmm1)[-1]
    nnn=sorted(list(nnn.items()),key=lambda x:x[0]);nnn1=dict(nnn)
    h=list(nnn1)[-1]
    mlabels_all_collection=mmm1[g]
    npredict_all_collection=nnn1[h]
    mmm0=sorted(list(mmm0.items()),key=lambda x:x[0]);mmm0=dict(mmm0)
    g0=list(mmm0)[-1]
    nnn0=sorted(list(nnn0.items()),key=lambda x:x[0]);nnn0=dict(nnn0)
    h0=list(nnn0)[-1]
    m0train_labels_all_collection=mmm0[g0]
    n0train_predict_all_collection=nnn0[h0]

    msg = 'Test_Rmse: {0:>6.2%}, Test_Pearson: {1:>6.2%},Test_R_2: {2:>6.2%}'
    print('Loading model test ……')
    print(msg.format(Rmse_collection[0], Pearson_collection[len(Pearson_collection)-1],
                     R2_collection[len(R2_collection)-1], mlabels_all_collection, npredict_all_collection))
    print("Time usage:", time_dif)
    return Rmse_collection[0],Pearson_collection[len(Pearson_collection)-1],R2_collection[len(R2_collection)-1],mlabels_all_collection, npredict_all_collection, m0train_labels_all_collection, n0train_predict_all_collection
def train_reg(config, model, train_data, dev_data,test_data,
          start_time,fold_index):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)#Adam优化器
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    model.train()
    test_data = torch.tensor(test_data)         #使用pytorch框架需要将数据类型转化为tensor
    dev_data = torch.tensor(dev_data)
    train_data = torch.tensor(train_data)
    test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)     #将训练集按batch_size个数据为一批放入模型训练
    dev_iter = DataLoader(dev_data, batch_size=config.batch_size, shuffle=True)
    train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        train_loss=0
        train_predict_all = np.array([], dtype=int)
        train_labels_all = np.array([], dtype=int)
        for _, (trains) in enumerate(train_iter):  #trains的形状为（64,5,1）每个样本的形状为（5,1），第5行为样本真实值
            labels = trains[:,-1].squeeze(1)    #取出第五行为标签
            trains = trains[:,:-1]              #剩余的（64,4,1）为训练集
            outputs = model(trains).squeeze(1)  #输入到模型得到一个值得输出
            model.zero_grad()  #模型对每一批数据做反向传播处理之前，要清空上一批求得的梯度值
            labels = labels.float()
            loss = F.mse_loss(outputs, labels)  #损失函数
            loss.backward()   #反向传播
            optimizer.step()   #对参数进行优化
            labels = labels.data.cpu()
            predic = outputs.data.cpu()
            train_loss+=loss
            train_labels_all = np.append(train_labels_all, labels)
            train_predict_all = np.append(train_predict_all, predic)
        train_Rmse = np.sqrt(metrics.mean_squared_error(train_labels_all, train_predict_all))
        train_Pearson = pearson_r(train_labels_all, train_predict_all)
        dev_Rmse, dev_Pearson,dev_loss,dev_R2 = evaluate(config, model, dev_iter)  #验证
        test_labels_all1=labels
        test_predict_all1=predic
        if dev_loss < dev_best_loss:
        # 根据验证集的损失去比较上一轮验证的损失，我们希望模型的损失越小越好，如果比上一次要好，我们就保存这一轮模型的参数
            dev_best_loss = dev_loss
            torch.save(model.state_dict(), config.save_path)
            improve = '*'
            last_improve = epoch
            test_Rmse,test_Pearson,test_R2,time_dif,test_labels_all1,test_predict_all1 = test(config, model, test_iter)
        else:
            improve = ''
        time_dif = get_time_dif(start_time)
        msg_train = 'Train Loss: {0:>5.2},Train_Rmse: {1:>6.2%},Train_Pearson: {2:>6.2%}, Val Loss: {3:>5.2},  Val_Rmse: {4:>6.2%}  Time: {6} {7}'
        # msg_evaluate = 'Val Loss: {1:>5.2},  Val Rmse: {2:>6.2%}, Val Pearson: {3:>6.2%}  Time: {4} {5}'
        print(msg_train.format(train_loss / len(train_iter), train_Rmse, train_Pearson, dev_loss, dev_Rmse,
                               dev_Pearson, time_dif, improve))
        if epoch - last_improve > config.require_improvement:
            # 验证集loss超过1000batch没下降，结束训练
            print("No optimization for a long time, auto-stopping...")                             
            break
    test_Rmse,test_Pearson,test_R2,time_dif,test_labels_all1,test_predict_all1 = test(config, model, test_iter)
    return test_Rmse,test_Pearson,test_R2,time_dif,test_labels_all1,test_predict_all1,train_labels_all, train_predict_all



def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path,map_location='cpu'))
    model.eval()
    start_time = time.time()   #做测试的开始时间
    test_Rmse,test_Pearson,test_loss,test_R2= evaluate(config, model, test_iter,test=True)
    msg = 'Test Loss: {0:>5.2},  Test_Rmse: {1:>6.2%}, Test_Pearson: {2:>6.2%},Test_R_2: {3:>6.2%}'
    print(msg.format(test_loss, test_Rmse,test_Pearson,test_R2))
    time_dif = get_time_dif(start_time)   #做测试的结束时间
    #print("Time usage:", time_dif)
    # return test_Rmse,test_Pearson,test_R2,time_dif
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    for texts in test_iter:
            labels = texts[:, -1].squeeze(1)
            texts = texts[:, :-1]
            outputs = model(texts).squeeze(1)
            labels = labels.float()
            labels = labels.data.cpu().numpy()
            predic = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    return test_Rmse,test_Pearson,test_R2,time_dif,labels_all,predict_all


def evaluate(config, model, data_iter,test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    # 评估我们只需要前向传播输出预测值，不需要反向传播，当然也不需要求得梯度
    with torch.no_grad():
        for texts in data_iter:
            labels = texts[:, -1].squeeze(1)
            texts = texts[:, :-1]
            outputs = model(texts).squeeze(1)
            labels = labels.float()
            loss = F.mse_loss(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu().numpy()
            predic = outputs.data.cpu().numpy()
            labels_all = np.append(labels_all, labels)
            predict_all = np.append(predict_all, predic)
    Pearson = pearson_r(labels_all, predict_all)
    R2 = metrics.r2_score(labels_all, predict_all)
    Rmse = np.sqrt(metrics.mean_squared_error(labels_all, predict_all))
    if test:
        return Rmse, Pearson, loss_total / len(data_iter),R2
    return Rmse, Pearson, loss_total / len(data_iter),R2,

#皮尔森指数
def pearson_r(x, y):
    x_bar, y_bar = np.mean(x), np.mean(y)
    cov_est = np.sum((x - x_bar) * (y - y_bar))
    std_x_est = np.sqrt(np.sum((x - x_bar)**2))
    std_y_est = np.sqrt(np.sum((y - y_bar)**2))
    if std_x_est * std_y_est == 0:
        if std_x_est == 0 and std_y_est!=0:
            return cov_est / std_y_est
        if std_x_est != 0 and std_y_est==0:
            return cov_est / std_x_est
        if std_x_est == 0 and std_y_est==0:
            return 0.0
    else:
        return cov_est / (std_x_est * std_y_est)