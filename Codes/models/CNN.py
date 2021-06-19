# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
#import numpy as np


class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'CNN'
        self.train_path = dataset + '/data/data.csv'  # 训练集

        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'  # 模型训练结果

        self.device = torch.device('cpu')  # 设备
        self.k_folds = 5

        self.dropout = 0.5  # 随机失活
        self.require_improvement = 10

        self.num_classes = 1  # 类别数

        self.num_epochs = 20  # epoch数
        self.batch_size = 64  # mini-batch大小

        self.learning_rate = 0.1  # 学习率
        self.embed = 1

        self.filter_sizes = (2, 3, 4)                                   # 卷积核尺寸
        self.num_filters = 32                                          # 卷积核数量(channels数)


'''Convolutional Neural Networks for Sentence Classification'''


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = x.type(torch.float32).clone().detach().requires_grad_(True)
        out = out.unsqueeze(1)
        #CNN的输入格式为（64,1,4,1），第二维度为频道信号，虽然我们没有，但是要加上才能用cnn
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        #依次按3个卷积核得到三个特征矩阵，并拼接起来，形状为（64,96）
        out = self.dropout(out)
        out = self.fc(out)
        return out
