from importlib import import_module
from utils import build_dataset, build_iterator, get_time_dif
from train_eval import test
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import time
import pandas as pd
import numpy as np



def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    model.eval()
    predict_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in test_iter:
            outputs = model(texts, config)
            predic = torch.max(outputs.data, 1)[1].cpu().numpy()
            predict_all = np.append(predict_all, predic)
    return predict_all

def input2output(path):
    #"text1 split('\n')
    #text2[text1,text2]
    #' '.join()
    try:
        dataset = 'Dataset'  # 数据集
        model_name = 'RNN'   #要用哪个模型在这里改
        x = import_module('models.' + model_name)
        config = x.Config(dataset)
        model = x.Model(config).to(config.device)
        test_contents = []
        contents_value = []
        contents = []
        raw_data = pd.read_csv(path)
        raw_data = raw_data.dropna().reset_index()
        standardScaler = StandardScaler()  # 对特征做归一化
        standard_data = standardScaler.fit_transform(raw_data.loc[:, ['a', 'b', 'c', 'd']])
        for index in tqdm(range(standard_data.shape[0])):
            feature = standard_data[index]
            label = raw_data.e[index]  # label
            if not label:
                continue
            data = np.append(feature, label)
            data = data.reshape((-1, 1))  # 将特征变为4行一列，用于LSTM的输入，相当于序列长度为4，每个特征维度为1
            contents.append(data)
        test_contents_iter = build_iterator(test_contents, config,1)
        predict=test(config, model, test_contents_iter)
        print(predict)
    except Exception as e:
        print(e)
        return 404
'''
path = 输入你要的文件地址，文件格式为csv，数据每列为（a,b,c,d,e）,其中e为预测值，如果不是，就改成这种格式
input2output(path)
'''