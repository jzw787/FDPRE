# -*- coding: utf-8 -*-
"""
Created on Sat Oct 17 10:44:54 2020

@author: 37984
"""

import tkinter as tk
from tkinter import ttk
from tkinter import Menu
import tkinter.messagebox

from tkinter import filedialog, dialog
import os
import test1

import cnn1
import rnn1
import rf
import sjwl

from pandas import read_csv
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import  MLPRegressor
#a=os.getcwd()
#os.chdir(a)
data = read_csv(r".\Dataset\data\data.csv",encoding='utf-8')

x = data.iloc[:,:-1]
y = data.iloc[:,-1]
x=x.values.tolist()
x=np.asarray(x)
y=y.values.tolist()
y=np.asarray(y)

# Create instance
win = tk.Tk()

# Add a title
win.title("FDPRE model")  

canvas = tk.Canvas(win, width=1200,height=699,bd=0, highlightthickness=0)
root = tk.Tk()
root.withdraw()
photo = tk.PhotoImage(file=".\\test.png")
theLabel = tk.Label(win,
                    text="Welcome to FDPRE",#内容
                    justify=tk.LEFT,#对齐方式
                    image=photo,#加入图片
                    compound = tk.CENTER,#关键:设置为背景图片
                    font=("Times New Roman",18),#字体和字号
                    fg = "white")#前景色
theLabel.pack()

#app=App(win)
#win.geometry("800x600") #英文x
#tk.mainloop()

tabControl = ttk.Notebook(win)          # Create Tab Control

tab1 = ttk.Frame(tabControl)            # Create a tab 
tabControl.add(tab1, text='Water level predict')      # Add the tab
tab2 = ttk.Frame(tabControl)            # Add a second tab
tabControl.add(tab2, text='Weather forecast')      # Make second tab visible
tab3 = ttk.Frame(tabControl)            # Add a second tab
tabControl.add(tab3, text='Interactive chart drawing')      # Make second tab visible

tabControl.pack(expand=1, fill="both")  # Pack to make visible

# LabelFrame using tab1 as the parent
#mighty = ttk.LabelFrame(tab1, text=' Mighty Python ')
mighty = ttk.LabelFrame(tab1)
mighty.grid(column=0, row=0, padx=8, pady=4)

# Modify adding a Label using mighty as the parent instead of win
#a_label = ttk.Label(mighty, text="请输入闸站名")
#a_label.grid(column=0, row=0, sticky='W')
b_label = ttk.Label(mighty, text="Water level before rainfall(m)")
b_label.grid(column=0, row=2, sticky='W')

b1_label = ttk.Label(mighty, text="Pumping station flow(m3/s)")
b1_label.grid(column=1, row=2, sticky='W')

b2_label = ttk.Label(mighty, text="Rainfall(mm)")
b2_label.grid(column=2, row=2, sticky='W')

b3_label = ttk.Label(mighty, text="Rainfall days(d)")
b3_label.grid(column=3, row=2, sticky='W')

# Modified Button Click Function
def click_me(): 
    action.configure(text='Hello ' + name.get() + ' ' + 
                     number_chosen.get())

# Adding a Textbox Entry widget
#name = tk.StringVar()
#name_entered = ttk.Entry(mighty, width=12, textvariable=name)
#name_entered.grid(column=1, row=0, sticky='W')               # align left/West

name = tk.StringVar()
name_entered = ttk.Entry(mighty, width=20, textvariable=name)
name1 = tk.StringVar()
name1_entered = ttk.Entry(mighty, width=20, textvariable=name1)
name2 = tk.StringVar()
name2_entered = ttk.Entry(mighty, width=20, textvariable=name2)
name3 = tk.StringVar()
name3_entered = ttk.Entry(mighty, width=12, textvariable=name3)
name_entered.grid(column=0, row=3, sticky='W')               # align left/West
name1_entered.grid(column=1, row=3, sticky='W')
name2_entered.grid(column=2, row=3, sticky='W')
name3_entered.grid(column=3, row=3, sticky='W')

def run_me(): 
    shuiwei=name.get()
    liuliang=name1.get()
    yuliang=name2.get()
    tianshu=name3.get()
    a1=float(shuiwei)
    a2=float(yuliang)
    a3=float(tianshu)
    a4=float(liuliang)
    b=[a1,a2,a3,a4]
#    from sklearn.svm import SVR
#    svr_rbf = SVR(kernel='rbf', C=10)
#    svr_lin = SVR(kernel='linear', C=10)
#    svr_poly = SVR(kernel='poly', C=10, degree=2)
#    y_rbf = svr_rbf.fit(x, y).predict([b])
#    y_lin = svr_lin.fit(x, y).predict([b])
#    y_poly = svr_poly.fit(x, y).predict([b])
    rf=RandomForestRegressor()
    y_RF_rf = rf.fit(x, y).predict([b])
    mlp=MLPRegressor(solver='lbfgs',hidden_layer_sizes=(100))
    y_MLPR_mlp = mlp.fit(x, y).predict([b])
    y_cnn=test1.cnn(a1,a2,a3,a4)
    y_rnn=test1.lstm(a1,a2,a3,a4)

    radSel=radVar.get()
    if   radSel == 0: tk.messagebox.showinfo('RF result',y_RF_rf)  # zero-based
    elif radSel == 1: tk.messagebox.showinfo('MLPR result',y_MLPR_mlp)  # using list
    elif radSel == 2: tk.messagebox.showinfo('CNN result',y_cnn)
    elif radSel == 3: tk.messagebox.showinfo('LSTM result',y_rnn)


ttk.Label(mighty, text="Machine learning methods").grid(column=0, row=4)
b_label = ttk.Label(mighty, text="Select pumping stations")
b_label.grid(column=0, row=1, sticky='W')
number = tk.StringVar()
number_chosen = ttk.Combobox(mighty, width=20, textvariable=number, state='readonly')
number_chosen['values'] = ('XJK','TG')
ttt=number_chosen.get()
number_chosen.grid(column=1, row=1)
number_chosen.current(0)

def run_new():
    tk.messagebox.showinfo(title = 'New stations',message='Please import dataset')
    '''
    Open files
    :return:
    '''
    global file_path
    global file_text
    file_path = filedialog.askopenfilename(title=u'Select files', initialdir=(os.path.expanduser('H:/')))
    print('Open files：', file_path)
    if file_path is not None:
        with open(file=file_path, mode='r+', encoding='utf-8') as file:
            file_text = file.read()
            tk.messagebox.showinfo('Read data',file_text)
            if os.path.basename(file_path) != 'trainingdata1.csv' and os.path.basename(file_path) != 'testdata1.csv':
                tk.messagebox.askokcancel('Notice','Please notice the file name of the training set.')
            else:
                tk.messagebox.askokcancel('Notice','The training set of the new pumping station is read successfully.')
    data2 = read_csv(r".\testdata1.csv",encoding='utf-8')
    x = data2.iloc[:,:-1]
    y = data2.iloc[:,-1]
    x=x.values.tolist()
    x=np.asarray(x)
    y=y.values.tolist()
    y=np.asarray(y)

action1 = ttk.Button(mighty, text="New pumping stations", command=run_new)
action1.grid(column=2, row=1)
name5 = tk.StringVar()
name5_entered = ttk.Entry(mighty, width=12, textvariable=name5)
name5_entered.grid(column=3, row=1, sticky='W')

number_chosen = ttk.Combobox(mighty, width=20, textvariable=number, state='readonly')
number_chosen['values'] = ('XJK','TG')
ttt=number_chosen.get()
number_chosen.grid(column=1, row=1)
number_chosen.current(0)

########################################

mighty2 = ttk.LabelFrame(tab2)
mighty2.grid(column=0, row=0, padx=12, pady=3)

colors = ['RF', 'MLPR', 'CNN', 'LSTM']   

radVar = tk.IntVar()
for col in range(4):                             
    curRad = tk.Radiobutton(mighty2, text=colors[col], variable=radVar, 
                            value=col)          
    curRad.grid(column=col, row=5, sticky=tk.W)             # row=5 ... SURPRISE!


#########
mighty1 = ttk.LabelFrame(tab3)
mighty1.grid(column=0, row=0, padx=12, pady=3)

# Adding a Button
b_label5 = ttk.Label(mighty1, text="Training result")
b_label5.grid(column=0, row=2, sticky='W')
action = ttk.Button(mighty1, text="RF", command=rf.main)   
action.grid(column=1, row=2)

action = ttk.Button(mighty1, text="MLPR", command=sjwl.main)   
action.grid(column=2, row=2)                                

action = ttk.Button(mighty1, text="CNN", command=cnn1.main)   
action.grid(column=3, row=2)

action = ttk.Button(mighty1, text="LSTM", command=rnn1.main)   
action.grid(column=4, row=2)
########################

import json
from datetime import date
import random
import socket

socket.setdefaulttimeout(600)

def get_ua():
    user_agents = [
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36 OPR/26.0.1656.60',
		'Opera/8.0 (Windows NT 5.1; U; en)',
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/74.0.3729.108 Safari/537.36',
		'Mozilla/5.0 (Windows NT 5.1; U; en; rv:1.8.1) Gecko/20061208 Firefox/2.0.0 Opera 9.50',
		'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; en) Opera 9.50',
		'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:34.0) Gecko/20100101 Firefox/34.0',
		'Mozilla/5.0 (X11; U; Linux x86_64; zh-CN; rv:1.9.2.10) Gecko/20100922 Ubuntu/10.10 (maverick) Firefox/3.6.10',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/534.57.2 (KHTML, like Gecko) Version/5.1.7 Safari/534.57.2 ',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/39.0.2171.71 Safari/537.36',
		'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
		'Mozilla/5.0 (Windows; U; Windows NT 6.1; en-US) AppleWebKit/534.16 (KHTML, like Gecko) Chrome/10.0.648.133 Safari/534.16',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/30.0.1599.101 Safari/537.36',
		'Mozilla/5.0 (Windows NT 6.1; WOW64; Trident/7.0; rv:11.0) like Gecko',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/536.11 (KHTML, like Gecko) Chrome/20.0.1132.11 TaoBrowser/2.0 Safari/536.11',
		'Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.1 (KHTML, like Gecko) Chrome/21.0.1180.71 Safari/537.1 LBBROWSER',
		'Mozilla/4.0 (compatible; MSIE 6.0; Windows NT 5.1; SV1; QQDownload 732; .NET4.0C; .NET4.0E)',
		'Mozilla/5.0 (Windows NT 5.1) AppleWebKit/535.11 (KHTML, like Gecko) Chrome/17.0.963.84 Safari/535.11 SE 2.X MetaSr 1.0',
		'Mozilla/4.0 (compatible; MSIE 7.0; Windows NT 5.1; Trident/4.0; SV1; QQDownload 732; .NET4.0C; .NET4.0E; SE 2.X MetaSr 1.0) ',
	]
    user_agent = random.choice(user_agents)
    return user_agent
ua=get_ua()

headers = {"User-Agent": ua}

url6 = "http://api.openweathermap.org/data/2.5/forecast?q=Cenhe&APPID=a4f8ae1355db7975f6eb9cd4bf99d9c4"  #沙市区岑河镇

from urllib import request
str6 = request.urlopen(request.Request(url6, headers=headers),timeout=10).read().decode('utf-8')

a = os.getcwd()
today_time = str(date.today())


def rain():
    global ss1
    os.chdir(a)
    with open(today_time+'_weather6.json','a') as f:   
        json.dump(str6,f) 
        load_dict6 = json.loads(str6)
    ss=load_dict6['list']
    if 'rain' in ss[0]:
        ss1=ss[0]['rain']['3h']
    else:
        ss1=0
    print(ss1)
    tk.messagebox.showinfo('rainfall in next 3h',str(ss1)+"mm")

def predict():
    data1 = read_csv(r".\testdata.csv",encoding='utf-8')
    x1 = data1.iloc[:,:-1]
    y1 = data1.iloc[:,-1]
    x1=x1.values.tolist()
    x1=np.asarray(x1)
    y1=y1.values.tolist()
    y1=np.asarray(y1)
    os.chdir(a)
    with open(today_time+'_weather6.json','a') as f:   
        json.dump(str6,f) 
        load_dict6 = json.loads(str6)
    mm=load_dict6['list']
    if 'rain' in mm:
        ss1=mm[0]['rain']['3h']
    else:
        ss1=0
    shuiwei=name.get()
    liuliang=name1.get()
    tianshu=name3.get()
    a1=float(shuiwei)
    a3=float(tianshu)
    a4=float(liuliang)
    b=[a1,ss1,a3,a4]
    rf=RandomForestRegressor()
    y_RF_rf = rf.fit(x, y).predict([b])
    mlp=MLPRegressor(solver='lbfgs',hidden_layer_sizes=(100))
    y_MLPR_mlp = mlp.fit(x, y).predict([b])
    y_CNN=test1.cnn(a1,ss1,a3,a4)
    y_RNN=test1.lstm(a1,ss1,a3,a4)
    radSel=radVar.get()
    if   radSel == 0: tk.messagebox.showinfo('RF result',y_RF_rf)  # zero-based
    elif radSel == 1: tk.messagebox.showinfo('MLPR result',y_MLPR_mlp)  # using list
    elif radSel == 2: tk.messagebox.showinfo('CNN result',y_CNN)
    elif radSel == 3: tk.messagebox.showinfo('LSTM result',y_RNN)

#########################


# Adding a Button
b_label6 = ttk.Label(mighty2, text="Crawler acquisition of weather forecast")
b_label6.grid(column=0, row=2, sticky='W')
action = ttk.Button(mighty2, text="Future rainfall", command=rain)   
action.grid(column=1, row=2)

action = ttk.Button(mighty2, text="Predicted water level after rainfall", command=predict)   
action.grid(column=2, row=2)


###########################################
# Using a scrolled Text control    
radVar.set(99)                                 
#
for col in range(4):                             
    curRad = tk.Radiobutton(mighty, text=colors[col], variable=radVar, 
                            value=col)          
    curRad.grid(column=col, row=5, sticky=tk.W)             # row=5 ... SURPRISE!
action = ttk.Button(mighty, text="run", command=run_me)   
action.grid(column=3, row=4)

####################################################

# Exit GUI cleanly
def _quit():
    result = tk.messagebox.askokcancel(title = 'Warning',message='Are you sure you want to quit？')
    if result is True:
        win.quit()
        win.destroy()
        exit()

file_path = ''
file_text = ''

def open_file():
    '''
    Open files
    :return:
    '''
    global file_path
    global file_text
    file_path = filedialog.askopenfilename(title=u'Select files', initialdir=(os.path.expanduser('H:/')))
    print('Open files：', file_path)
    if file_path is not None:
        with open(file=file_path, mode='r+', encoding='utf-8') as file:
            file_text = file.read()
            tk.messagebox.showinfo('Read data',file_text)
            if os.path.basename(file_path) != 'trainingdata.csv' and os.path.basename(file_path) != 'testdata.csv':
                tk.messagebox.showerror('Error','Wrong!')

#        text1.insert('insert', file_text)

def save_file():
    global file_path
    global file_text
    file_path = filedialog.asksaveasfilename(title=u'Save fiels')
    print('Save fiels：', file_path)
#    file_text = text1.get('1.0', tk.END)
    if file_path is not None:
        with open(file=file_path, mode='a+', encoding='utf-8') as file:
            file.write(file_text)
#        text1.delete('1.0', tk.END)
        dialog.Dialog(None, {'title': 'File Modified', 'text': 'Save finished', 'bitmap': 'warning', 'default': 0,
                             'strings': ('OK', 'Cancle')})
        print('Save finished')

# Creating a Menu Bar
menu_bar = Menu(win)
win.config(menu=menu_bar)

# Add menu items
file_menu = Menu(menu_bar, tearoff=0)
file_menu.add_command(label="Open",command=open_file)
file_menu.add_command(label="Save",command=save_file)
file_menu.add_separator()
file_menu.add_command(label="Quit", command=_quit)
menu_bar.add_cascade(label="Files", menu=file_menu)

# Add another Menu to the Menu Bar and an item
def about():
    tk.messagebox.showinfo('Instruction','This is a software based on machine learning and weather forecast to predict the water level in front of the gate after rainfall\nAt present, it is only support RF, MLPR, CNN, and LSTM\nThe software is still under development, please look forward to it！')

help_menu = Menu(menu_bar, tearoff=0)
help_menu.add_command(label="About",command=about)
menu_bar.add_cascade(label="Help", menu=help_menu)

name_entered.focus()      # Place cursor into name Entry
#======================
# Start GUI
#======================
win.mainloop()