#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: DBScan_learn.py
@time: 2020/5/17 0:41
@desc:
'''

import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster as skc
from sklearn import metrics

mac2id = dict()
onlinetimes = []
f = open('data/TestData.txt', encoding='utf-8')
for line in f:
    mac = line.split(',')[2]
    onlinetime = int(line.split(',')[6])
    starttime = int(line.split(',')[4].split(' ')[1].split(':')[0])
    if mac not in mac2id:
        mac2id[mac] = len(onlinetimes)
        onlinetimes.append((starttime, onlinetime))
    else:
        onlinetimes[mac2id[mac]] = [(starttime, onlinetime)]
# 列数 = 2  -1 行数未知
real_X = np.array(onlinetimes).reshape((-1, 2))
# print(real_X)

X = real_X[:, 0:1]
print(X)

# 对时间段 标签进行分类， 所以希望我要做的就是这种 关联聚类分析 其实他不太懂
db = skc.DBSCAN(eps=0.01, min_samples=20).fit(X)
labels = db.labels_

print('Labels:')
print(labels)
raito = len(labels[labels[:] == -1]) / len(labels)
print('Noise raito:', format(raito, '.2%'))

print('len labels', len(set(labels)))
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

print('Estimated number of clusters: %d' % n_clusters_)
print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

for i in range(n_clusters_):
    print('Cluster ', i, ':')
    # flatten是numpy.ndarray.flatten的一个函数，即返回一个一维数组。
    # flatten只能适用于numpy对象，即array或者mat，普通的list列表不适用！。
    # a.flatten()：a是个数组，a.flatten()就是把a降到一维，默认是按行的方向降 。
    print(list(X[labels == i].flatten()))

plt.hist(X, 24)
plt.show()
