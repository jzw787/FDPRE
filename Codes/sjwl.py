# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 21:24:17 2020

@author: 37984
"""


import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

from sklearn.neural_network import  MLPRegressor
#from sklearn.metrics import r2_score

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk
from matplotlib import rcParams
from pylab import *
mpl.rcParams['font.sans-serif'] = ['simhei']

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

class From2:
    def __init__(self): 
        self.root=tk.Tk()                    
        self.canvas=tk.Canvas()              
        self.figure=self.create_matplotlib2() 
        self.create_form2(self.figure)        
        self.root.mainloop()
 
    def create_matplotlib2(self):
        f2=plt.figure(num=2,figsize=(10,4),dpi=100,facecolor="pink",edgecolor='green',frameon=True)
        y_noise=y
        y1_noise=y1
        svr_rbf = SVR(kernel='rbf', C=10)
        svr_lin = SVR(kernel='linear', C=10)
        mlp=MLPRegressor(solver='lbfgs',hidden_layer_sizes=(100))
        y_SVM_rbf = svr_rbf.fit(x, y).predict(x)
        y_SVM_lin = svr_lin.fit(x, y).predict(x)
        y_MLPR_mlp = mlp.fit(x, y).predict(x)
        
        
        y1_SVM_rbf = svr_rbf.fit(x1, y1).predict(x1)
        y1_SVM_lin = svr_lin.fit(x1, y1).predict(x1)
        y1_MLPR_mlp = mlp.fit(x1, y1).predict(x1)
    # ############################
        lw = 2
        plt.subplot(1,2,1)
        plt.plot(y, y_noise, color='b', label='Input data')
#        plt.scatter(y, y_SVM_rbf, color='y', lw=lw, label='SVM_RBF C=10 ')
#        plt.scatter(y,y_SVM_lin, color='c', lw=lw, label='SVM_Linear C=10')
        plt.scatter(y,y_MLPR_mlp, color='r', lw=lw, label='y_MLPR_mlp')
        plt.xlabel('Observed water level after rainfall (m)')
        plt.ylabel('Predicted water level after rainfall (m)')
        plt.title('MLPR-XJK-train')
        plt.legend()
#        plt.savefig('神经网络-习家口站训练.jpg', dpi=300)
        lw = 2
        plt.subplot(1,2,2)
        plt.plot(y1, y1_noise, color='b', label='Input data')
#        plt.scatter(y1, y1_SVM_rbf, color='y', lw=lw, label='SVM_RBF C=10 ')
#        plt.scatter(y1,y1_SVM_lin, color='c', lw=lw, label='SVM_Linear C=10')
        plt.scatter(y1,y1_MLPR_mlp, color='r', lw=lw, label='y_MLPR_mlp')
        plt.xlabel('Observed water level after rainfall (m)')
        plt.ylabel('Predicted water level after rainfall (m)')
        plt.title('MLPR-XJK-test')
#        plt.savefig('神经网络-习家口站预测.jpg', dpi=300)
        plt.legend()
        plt.show()
#        fig1.grid(which='major',axis='x',color='r', linestyle='-', linewidth=2)          
        return f2
 
    def create_form2(self,figure):
        self.canvas=FigureCanvasTkAgg(figure,self.root)
        self.canvas.draw() 
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
 
        toolbar =NavigationToolbar2Tk(self.canvas, self.root) 
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)

def main():
        form=From2()

if __name__=="__main__":
    main()