#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: cluster_da.py
@time: 2020/5/16 23:20
@desc: cluster DA
'''

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=0.3)
print(y_train)

# 训练数据
knn = KNeighborsClassifier()  # 引入训练方法
knn.fit(X_train, y_train)  # 进行填充测试数据进行训练

# 预测数据###
print(knn.predict(X_test))  # 预测特征值
'''
[1 1 1 0 2 2 1 1 1 0 0 0 2 2 0 1 2 2 0 1 0 0 0 0 0 0 2 1 0 0 0 1 0 2 0 2 0
 1 2 1 0 0 1 0 2]
'''
print(y_test)  # 真实特征值
'''
[1 1 1 0 1 2 1 1 1 0 0 0 2 2 0 1 2 2 0 1 0 0 0 0 0 0 2 1 0 0 0 1 0 2 0 2 0
 1 2 1 0 0 1 0 2]
'''
