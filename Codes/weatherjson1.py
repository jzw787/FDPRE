# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 09:14:16 2021

@author: 37984
"""

import os
import json
#import time
#import signal
#import sys
#import jsonpath
import re
#import codecs
#import csv
#import pandas as pd
#import pymysql  
from datetime import date
import random
import socket
#import re
import numpy as np

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

def contextarray(context):
    return np.matrix([map(float,re.split('\s+',ln.strip()))
                      for ln in context.splitlines() if ln.strip()])

def rain():
    global ss1
    os.chdir(a)
    with open(today_time+'_weather6.json','a') as f:   
        json.dump(str6,f) 
        load_dict6 = json.loads(str6)
#        result6 = re.findall(r'"rain.*?dt_txt.*?\}',str6)       
#    weather_list6 = load_dict6.get('list')
    ss=load_dict6['list']
#        ss1=json.loads(ss)
#        with open(today_time+'_rain6.csv','a') as f:
#            ss=pd.read_csv(f)
#        result6 = re.findall(r'"rain.*?dt_txt.*?\}',str6)
#        test6 = pd.DataFrame(data=result6)
    ss1=ss[0]['rain']['3h']
    ss2=int(ss1)
    print('观音垱未来3h降雨量')
##        print(load_dict6.get('rain'))
    print(ss1)
    print(ss2)
rain()