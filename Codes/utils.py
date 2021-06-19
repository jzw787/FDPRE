# coding: UTF-8
#import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
import numpy as np
#from sklearn import metrics
#import os
#from bs4 import BeautifulSoup
#import re
#import string
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.ensemble import RandomForestRegressor


def build_dataset(config):

    def load_dataset(path_data):
        contents = []
        raw_data = pd.read_csv(path_data)
        raw_data = raw_data.dropna().reset_index()
        standardScaler = StandardScaler()   #对特征做归一化
        standard_data = standardScaler.fit_transform(raw_data.loc[:,['a','b','c','d']])
#        standard_data = raw_data.loc[:, ['a', 'b', 'c', 'd']].values
        for index in tqdm(range(standard_data.shape[0])):
            feature = standard_data[index]
            label = raw_data.e[index]  # label
            if not label:
                continue
            data = np.append(feature,label)
            data = data.reshape((-1,1))   #将特征变为4行一列，用于LSTM的输入，相当于序列长度为4，每个特征维度为1
            contents.append(data)
        return contents
    train = load_dataset(config.train_path)
    train,test =train_test_split(train,test_size=0.1,random_state=2020)
    train = np.array(train)
    test = np.array(test)
    print(train.shape,test.shape)
    return train,test

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))