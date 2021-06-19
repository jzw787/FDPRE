# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import Kfold_train_reg
from train_eval import test
from importlib import import_module
from torch.utils.data import DataLoader
#import argparse
#import pandas as pb
#import utils
from utils import build_dataset, get_time_dif

#import gc

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
from sklearn.metrics import r2_score
#from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error

if __name__ == '__main__':
    dataset = 'Dataset'  # 数据集

    model_name = ['RNN']  #model_name = ['RNN','CNN']
    result_collection = []
    start_time = time.time()
    for model_item in model_name:
        result = []
        x = import_module('models.' + model_item)
        config = x.Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        print("Loading data...")
        train_data, test_data = build_dataset(config)

        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        # train
        start=time.time()
        Rmse,Pearson,R2,labels1, predict1,train_labels_all, train_predict_all = Kfold_train_reg(config,train_data, test_data)
        start=time.time()
        train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
        model=x.Model(config).to(config.device)
        a_Rmse,a_Pearson,a_R2,a_dif,labels_aa,predict_aa=test(config, model, train_iter)
        end=time.time()
        print("Time usage train: %s Seconds"%(end-start))
        model=x.Model(config).to(config.device)
        test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)
        start1=time.time()
        test_Rmse,test_Pearson,test_R2,time_dif,labels_all, predict_all=test(config, model, test_iter)
        end1=time.time()
        print("Time usage test: %s Seconds"%(end1-start1))
        
        print(predict1, labels1)
        lw=2
        plt.plot(labels1,labels1, color='navy', lw=lw, label='y=x')
        plt.scatter(labels1, predict1, color='c', lw=lw, label='LSTM')
        plt.xlabel('Observed water level after rain (m)')
        plt.ylabel('Predicted water level after rain (m)')
        plt.title('LSTM')
        plt.legend()
        plt.savefig(str(model_name)+'.jpg', dpi=300)
        plt.show()
        print('R-squared value of LSTM is',r2_score(labels1, predict1))
        print('R-squared value of CNN train is',r2_score(train_labels_all, train_predict_all))
        print('RMSE of CNN train is',mean_squared_error(train_labels_all, train_predict_all))
        
        result.append(Rmse)
        result.append(Pearson)
        result.append(R2)
        result_collection.append(result)
      
        del train_data, test_data
    #打印两个模型最好的结果

    for index in range(len(model_name)):
        msg = 'Test_Rmse: {0:>6.2%}, Test_Pearson: {1:>6.2%},Test_R_2: {2:>6.2%}'
        print(model_name[index],'result',msg.format(result_collection[index][0],result_collection[index][1],
                                                    result_collection[index][2]))
        

