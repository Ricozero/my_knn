from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import time

import knn

def standardize(x):
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    sigma[sigma == 0] = 1
    return (x - mu) / sigma

# 1、读取数据集
t0 = time.time()
dataset_name = 'iris'
if dataset_name == 'iris':
    dataset = datasets.load_iris()
    x = dataset.data
    y = dataset.target
elif dataset_name == 'digits':
    dataset = datasets.load_digits()
    x = dataset.data
    y = dataset.target
    #x = standardize(x)
elif dataset_name == 'skin':
    xlist = []
    ylist = []
    with open('Skin_NonSkin.txt', 'r') as f:
        for line in f.readlines():
            numlist = line[:-1].split('\t')
            xlist.append([int(num) for num in numlist[:-1]])
            ylist.append(int(numlist[-1]))
    x = np.array(xlist)
    y = np.array(ylist) - 1
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print('读取数据集\n耗时：%.4fs\n' % (time.time() - t0))

# 2、测试简单kNN
t0 = time.time()
y_pred = knn.knn(x_train, y_train, x_test, 4)
print('简单kNN\n耗时：%.4fs' % (time.time() - t0))
print(classification_report(y_test, y_pred))

# 3、测试基于k-d树的kNN
t0 = time.time()
y_pred = knn.knn_kdt(x_train, y_train, x_test, 4)
print('基于k-d树的kNN\n耗时：%.4fs' % (time.time() - t0))
print(classification_report(y_test, y_pred))