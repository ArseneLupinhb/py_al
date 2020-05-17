#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: kmeans_learn.py
@time: 2020/5/17 0:00
@desc:
'''

import numpy as np
from sklearn.cluster import KMeans


def loadData(filePath):
    fr = open(filePath, 'r+')
    lines = fr.readlines()
    retData = []
    retCityName = []
    for line in lines:
        items = line.strip().split(",")
        retCityName.append(items[0])
        retData.append([float(items[i]) for i in range(1, len(items))])
    return retData, retCityName


if __name__ == '__main__':
    data, cityName = loadData('data/city.txt')
    print(len(data), len(cityName))
    print(data)
    km = KMeans(n_clusters=4)
    label = km.fit_predict(data)
    print(label)
    expenses = np.sum(km.cluster_centers_, axis=1)
    print(expenses)
    # print(expenses)
    CityCluster = [[], [], [], []]
    for i in range(len(cityName)):
        CityCluster[label[i]].append(cityName[i])
    for i in range(len(CityCluster)):
        print("Expenses:%.2f" % expenses[i])
        print(CityCluster[i])
