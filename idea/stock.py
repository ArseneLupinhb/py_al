import os

import tushare as ts

# 全局路径设置
os.getcwd()
source_path = os.getcwd() + r'/idea/work/'

# 一次性获取最近一个日交易日所有股票的交易数据
ts_df = ts.get_today_all()
# 新闻数据
new_df = ts.get_notices

ts_df.info()
ts_df.head()
# 使用 utf_8_sig 可以解决中文乱码问题
ts_df.to_csv(source_path + r'ts_df.csv', encoding='utf_8_sig')
