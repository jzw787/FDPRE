# -*- coding: utf-8 -*-
"""
Created on Sat Nov  7 21:22:44 2020

@author: 37984
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
#from sklearn.neural_network import  MLPRegressor

import tkinter as tk

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg,NavigationToolbar2Tk

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
        X=x
        X1=x1
        y_noise=y
        y1_noise=y1
        rf=RandomForestRegressor()
        y_RF_rf = rf.fit(X, y_noise).predict(X)
        y1_RF_rf = rf.fit(X1, y1_noise).predict(X1)
    # ############################
        lw = 2
        plt.subplot(1,2,1)
        plt.plot(y, y_noise, color='b', label='Input data')
#        plt.scatter(y, y_SVM_rbf, color='y', lw=lw, label='SVM_RBF C=10 ')
#        plt.scatter(y,y_SVM_lin, color='c', lw=lw, label='SVM_Linear C=10')
        plt.scatter(y, y_RF_rf, color='r', lw=lw, label='RandomFrest_RF')
        plt.xlabel('Observed water level after rainfall (m)')
        plt.ylabel('Predicted water level after rainfall (m)')
        plt.title('RF-XJK-train')
        plt.legend()
        lw = 2
        plt.subplot(1,2,2)
        plt.plot(y1, y1_noise, color='b', label='Input data')
#        plt.scatter(y1, y1_SVM_rbf, color='y', lw=lw, label='SVM_RBF C=10 ')
#        plt.scatter(y1,y1_SVM_lin, color='c', lw=lw, label='SVM_Linear C=10')
        plt.scatter(y1, y1_RF_rf, color='r', lw=lw, label='RandomFrest_RF')
        plt.xlabel('Observed water level after rainfall (m)')
        plt.ylabel('Predicted water level after rainfall (m)')
        plt.title('RF-XJK-test')
        plt.legend()
        plt.show()
#        fig1.grid(which='major',axis='x',color='r', linestyle='-', linewidth=2)              
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