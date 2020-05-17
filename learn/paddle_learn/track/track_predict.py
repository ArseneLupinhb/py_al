#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: track_predict.py
@time: 2020/5/17 8:54
@desc:
'''
import os

# 决策树
import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd

os.chdir(r"D:\soft_own\source\py_al\learn\paddle_learn\track")
os.getcwd()
train = pd.read_csv('data/train.csv')
test1 = pd.read_csv('data/test1.csv')
labels = train['label']
# print(labels)
features = train.drop(['label', 'Unnamed: 0'], axis=1)
# print(labels)
print(features)

train.info()
features.columns
features.info()

col = ['android_id', 'apptype', 'carrier', 'dev_height', 'dev_ppi',
       'dev_width', 'media_id', 'ntt', 'package',
       'timestamp', 'version', 'fea_hash', 'location', 'fea1_hash',
       'cus_type']
features = features[col]
# 训练集训练数据
# 生成train_model
clf = lgb.LGBMClassifier()
features['fea_hash'] = features['fea_hash'].map(lambda x: 0 if len(str(x)) > 16 else int(x))
features['fea1_hash'] = features['fea1_hash'].map(lambda x: 0 if len(str(x)) > 16 else int(x))
features['version'] = features['version'].map(lambda x: int(x) if str(x).isdigit() else 0)
# feature and labels
clf.fit(features, labels)

# 测试集
# eval_model
test_fea = test1[features.columns]
test_fea['fea_hash'] = test_fea['fea_hash'].map(lambda x: 0 if len(str(x)) > 16 else int(x))
test_fea['fea1_hash'] = test_fea['fea1_hash'].map(lambda x: 0 if len(str(x)) > 16 else int(x))
test_fea['version'] = test_fea['version'].map(lambda x: int(x) if str(x).isdigit() else 0)
clf.predict(test_fea)

# 生成测试集
eval_result = pd.DataFrame(test1['sid'])
eval_result['label'] = clf.predict(test_fea)
eval_result
eval_result.to_csv('work/eval_result.csv', index=False)
eval_result.info()
eval_result['label'].value_counts(normalize=True).plot.bar()
plt.show()
