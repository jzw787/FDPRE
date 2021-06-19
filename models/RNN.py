# coding: UTF-8
import torch
import torch.nn as nn
#import torch.nn.functional as F
#import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'RNN'
        self.train_path = dataset + '/data/data.csv'                                # 训练集

        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果

        self.device = torch.device('cpu')
        #self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 10
        self.num_classes = 1                        # 类别数
        self.k_folds = 5            #5

        self.num_epochs = 28                                           # epoch数
        self.batch_size = 35                                      # mini-batch大小

        self.learning_rate = 0.1                                        # 学习率 0.1

        self.hidden_size = 256                                         # lstm隐藏层
        self.num_layers = 1                                           # lstm层数
        self.embed = 1




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout)
        self.fc1 = nn.Linear(config.hidden_size * 2, config.num_classes)

    def forward(self, x):
        out = x.type(torch.float32).clone().detach().requires_grad_(True)        #输出的x为[64，4,1]的矩阵
        out, _ = self.lstm(out)    #3维矩阵输入3维矩阵输出，lstm输出（output,（hn,cn）） output(sep_len,batch,num_diction*hidden_size)
        #output的形状为[batch，seq_len，hidden_size]取seq_len最后一个位置，形状为（64,512）去做线性变换
        out = self.fc1(out[:, -1, :])  # 句子最后时刻的 hidden state    out[seq_len，batch，hidden_size]    从隐藏层到输出层的变换
        return out
