# 导入库
import os

import pandas as pd

os.getcwd()

source_path = os.getcwd() + r'/crawler/yhs_crawler/data/'
# 读入数据
df_1 = pd.read_csv(source_path + 'yhs弹幕.csv', encoding='utf-8', engine='python')
df_1.head(20)

df_1.info()
df_1.shape
