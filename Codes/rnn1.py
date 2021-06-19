# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:41:16 2021

@author: 37984
"""
#import time
import torch
import numpy as np
from importlib import import_module
from torch.utils.data import DataLoader
#import argparse
#import utils
from utils import build_dataset #get_time_dif
from RNN import Config,Model
#import gc

import matplotlib.pyplot as plt
plt.rc('font',family='Times New Roman')
#from sklearn.metrics import r2_score
#from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
#from matplotlib import rcParams
#from pylab import *

#mpl.rcParams['font.sans-serif'] = ['simhei']

data = pd.read_csv(r".\trainingdata.csv",encoding='utf-8')
data1 = pd.read_csv(r".\testdata.csv",encoding='utf-8')
x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x1 = data1.iloc[:,:-1]
y1 = data1.iloc[:,-1]


x=x.values.tolist()
x=np.asarray(x)
y=y.values.tolist()
y=np.asarray(y)
x1=x1.values.tolist()
x1=np.asarray(x1)
y1=y1.values.tolist()
y1=np.asarray(y1)

class From1:
    def __init__(self): 
        self.root=tk.Tk()                    
        self.canvas=tk.Canvas()              
        self.figure=self.create_matplotlib1() 
        self.create_form1(self.figure)        
        self.root.mainloop()
 
    def create_matplotlib1(self):
        f1=plt.figure(num=2,figsize=(10,4),dpi=100,facecolor="pink",edgecolor='green',frameon=True)
        dataset = 'Dataset'  # 数据集
        from train_eval import test
        model_name = ['RNN']  #model_name = ['RNN','CNN']
        result_collection = []
        for model_item in model_name:
            result = []
            # labels1_all_collection = []
            # predict1_all_collection = []
            config = Config(dataset)
            np.random.seed(1)
            torch.manual_seed(1)
            torch.cuda.manual_seed_all(1)
            torch.backends.cudnn.deterministic = True  # 保证每次结果一样
            print("Loading data...")
            train_data, test_data = build_dataset(config)
    
            train_iter = DataLoader(train_data, batch_size=config.batch_size, shuffle=True)
            model=Model(config).to(config.device)
            a_Rmse,a_Pearson,a_R2,a_dif,labels_aa,predict_aa=test(config, model, train_iter)
#            end=time.time()
#            print("Time usage train: %s Seconds"%(end-start))
            model=Model(config).to(config.device)
            test_iter = DataLoader(test_data, batch_size=config.batch_size, shuffle=True)
            test_Rmse,test_Pearson,test_R2,time_dif,labels_all, predict_all=test(config, model, test_iter)
            
            print(labels_aa,predict_aa)
            lw=2
            plt.subplot(1,2,1)
            plt.plot(labels_aa,labels_aa, color='b', lw=lw, label='y=x')
            plt.scatter(labels_aa, predict_aa, color='r', lw=lw, label='LSTM')
            plt.xlabel('Observed water level after rain (m)')
            plt.ylabel('Predicted water level after rain (m)')
            plt.title('LSTM-XJK-train')
            plt.legend()           
            lw = 2
            plt.subplot(1,2,2)
            plt.plot(labels_all, labels_all, color='b', label='Input data')
#        plt.scatter(y1, y1_SVM_rbf, color='y', lw=lw, label='SVM_RBF C=10 ')
#        plt.scatter(y1,y1_SVM_lin, color='c', lw=lw, label='SVM_Linear C=10')
            plt.scatter(labels_all, predict_all, color='r', lw=lw, label='LSTM')
            plt.xlabel('Observed water level after rainfall (m)')
            plt.ylabel('Predicted water level after rainfall (m)')
            plt.title('LSTM-XJK-test')
            plt.legend()
            plt.show()
            plt.savefig(str(model_name)+'.jpg', dpi=300)
            result_collection.append(result)
          
            del train_data, test_data#        fig1.grid(which='major',axis='x',color='r', linestyle='-', linewidth=2)              
        return f1
 
    def create_form1(self,figure):
        self.canvas=FigureCanvasTkAgg(figure,self.root)
        self.canvas.draw() 
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
 

        toolbar =NavigationToolbar2Tk(self.canvas, self.root)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        
def main():
        form=From1()

if __name__=="__main__":
    main()