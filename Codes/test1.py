from importlib import import_module
from torch.utils.data import  DataLoader
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from pandas import read_csv,DataFrame
import numpy as np



def test1(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path,map_location='cpu'))
    model.eval()
#    start_time = time.time()
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts in test_iter:
            outputs = model(texts)
            predict_all = np.append(predict_all, outputs)
    return predict_all

def test2(config, model, test_iter):
    # test
    model.load_state_dict(torch.load('Dataset/saved_dict/CNN1.ckpt',map_location='cpu'))
    model.eval()
#    start_time = time.time()
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts in test_iter:
            outputs = model(texts)
            predict_all = np.append(predict_all, outputs)
    return predict_all
 
def input2output(path):
    try:
        dataset = 'Dataset'  # 数据集
        model_name = 'RNN'   #要用哪个模型在这里改
        x = import_module('models.' + model_name)
        config = x.Config(dataset)
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        model = x.Model(config).to(config.device)
#        test_contents = []
#        contents_value = []
        contents = []
        raw_data = read_csv("./trainingdata.csv")
#        train,raw_data=train_test_split(raw_data,test_size=0.1,random_state=2020)
#        raw_data = np.array(raw_data)
        path_data=config.train_path
        contents = []
        raw_data = read_csv(path_data)
        index0=max(raw_data.index)+1
        path = '1.csv'
        raw_data1 = read_csv(path)
        m=raw_data1['a']
        mm=DataFrame(m.tolist(), index = m.index)
        mm=mm.iloc[[0]].values[0][0]            
        raw_data1.index = [index0]
        raw_data =raw_data.append(raw_data1)
        
        raw_data = raw_data.reset_index()
        raw_data =raw_data.fillna(mm)          
        standardScaler = StandardScaler()   #对特征做归一化
        standard_data = standardScaler.fit_transform(raw_data.loc[:,['a','b','c','d']])
        standard_data =standard_data[-1]
#        standard_data = raw_data.loc[:, ['a', 'b', 'c', 'd']].values
        for index in tqdm(range(standard_data.shape[0])):
            feature = standard_data[index]
            label = raw_data.e[index]  # label
            if not label:
                continue
            data = np.append(feature,label)
            data = data.reshape((-1,1))   #将特征变为4行一列，用于LSTM的输入，相当于序列长度为4，每个特征维度为1
            contents.append(data)  
        test_data1 = torch.tensor(contents)
        test_iter = DataLoader(test_data1, batch_size=1, shuffle=True)
        predict0=test1(config, model, test_iter)
        predict=predict0[-1]
#        print(predict)
        return predict
    except Exception as e:
        print(e)
        return 404
    
 
def input3output(path):
    #"text1 split('\n')
    #text2[text1,text2]
    #' '.join()
    try:
        dataset = 'Dataset'  # 数据集
        model_name = 'CNN'   #要用哪个模型在这里改
        x = import_module('models.' + model_name)
        config = x.Config(dataset)
        model = x.Model(config).to(config.device)
#        test_contents = []
#        contents_value = []
        contents = []
        raw_data = read_csv(path)
        raw_data = raw_data.dropna().reset_index()
        standardScaler = StandardScaler()  # 对特征做归一化
        standard_data = standardScaler.fit_transform(raw_data.loc[:, ['a', 'b', 'c', 'd']].values)
        standard_data = raw_data.loc[:, ['a', 'b', 'c', 'd']].values
        for index in tqdm(range(standard_data.shape[0])):
            feature = standard_data[index]  # label
            data = feature.reshape((-1, 1))  # 将特征变为4行一列，用于LSTM的输入，相当于序列长度为4，每个特征维度为1
            contents.append(data)
        test_data = torch.tensor(contents)
        test_iter = DataLoader(test_data, batch_size=1, shuffle=True)
        predict=test2(config, model, test_iter)
#        print(predict)
        return predict
    except Exception as e:
        print(e)
        return 404

def cnn(a,b,c,d):
    import csv
    f = open('1.csv','w',encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["a","b","c","d"])
    csv_writer.writerow([a,b,c,d])
    f.close()    
    path = '1.csv'
    cnn=input3output(path)
    return cnn

def lstm(a,b,c,d):
    import csv
    f = open('1.csv','w',encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    csv_writer.writerow(["a","b","c","d"])
    csv_writer.writerow([a,b,c,d])
    f.close()    
    path = '1.csv'
#    rnn=test(config, model, test_iter)[-1][-1]
    rnn=input2output(path)
    return rnn