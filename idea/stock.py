import os

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import tushare as ts

# 显示matplotlib路径
print(matplotlib.matplotlib_fname())
# 全局路径设置
os.getcwd()
source_path = os.getcwd() + r'/idea/work/'

# 一次性获取最近一个日交易日所有股票的交易数据
ts_df = ts.get_today_all()
# 新闻数据
new_df = ts.get_notices()

# 保存数据到csv
ts_df.to_csv(source_path + r'ts_df.csv', encoding='utf_8_sig')
# new_df.to_csv(source_path + r'new_df.csv', encoding='utf_8_sig')

# 从csv读取数据
ts_df = pd.read_csv(source_path + r'ts_df.csv')

ts_df.info()
ts_df.head()
ts_df.shape()
# 使用 utf_8_sig 可以解决中文乱码问题

(ts_df['trade'] - ts_df['open']) / ts_df['open'] * 100
ts_df['settlement'] * (1 + ts_df['changepercent'] / 100)

((ts_df['trade'] - ts_df['settlement']) / ts_df['settlement']) * 100

ts_df[ts_df['changepercent'] > 0].shape
ts_df[ts_df['changepercent'] <= 0].shape

ts_df.sort_values(by=['changepercent'], ascending=[True], inplace=True)
ts_df.reset_index(drop=True)
ts_df['changepercent_per'] = ts_df['changepercent'] / 100
ts_df['changepercent_per'].plot.bar(stacked=False)
ts_df['changepercent_per'].plot(stacked=False)
ts_df['changepercent_per'].plot.hist(stacked=False)
plt.show(drop=True)

temp_df = ts_df[(ts_df['changepercent'] > 0)].sort_values(by=['changepercent'], ascending=[False]).reset_index(
    drop=True)
# temp_df = temp_df.set_index("name")
# 是否保留原列
temp_df = temp_df.set_index("name", drop=False)
temp_df['changepercent_per'] = temp_df['changepercent'] / 100
temp_df[['changepercent', 'trade']][:40].plot.bar()
temp_df[['changepercent']][:40].plot()
plt.show()
