import io
import os

import numpy as np
import pandas as pd
import requests

os.getcwd()
work_path = os.getcwd() + r'/learn/data_analysis/work'

url = "https://raw.github.com/vincentarelbundock/Rdatasets/master/csv/datasets/AirPassengers.csv"
s = requests.get(url).content
df = pd.read_csv(io.StringIO(s.decode('utf-8')), nrows=10, index_col=0)
df
df.shape

# dict to DataFrame
data = {"grammer": ["python", "c", "java", "go", "R", "sql", "php", "python"],
        "score": [1, 2, np.nan, 4, 5, 6, 7, 10]}
df = pd.DataFrame(data)
df
df.shape

result = df.loc[df['grammer'] == "python"]
result

# 提取含有python 的行
result = df[df['grammer'].str.contains("python")]
result.shape
result.head()

# 获取列名
df.columns

# 更改列名
df.rename(columns={'score': 'popularity'}, inplace=True)

# 统计没门编程语言出现的次数
df['grammer'].value_counts()

# 将空值用上下值得均值填充
df['popularity'] = df['popularity'].fillna(df['popularity'].interpolate())
df

# 提取popularity列大于3的值
df[df['popularity'] > 3]

# 对grammer列去重
df.drop_duplicates(['grammer'])

# 计算popularity的均值
df['popularity'].mean()

# 将grammer列转换成列表
df['grammer'].to_list()

# 将dataframe保存为excel和csv
df.to_excel(work_path + r'/test.xlsx')
df.to_csv(work_path + r'/test.csv')

# 提取popularity列值大于3 小于7
df[(df['popularity'] > 3) & (df['popularity'] < 7)]

# 交换两列的位置
temp = df['popularity']
df.drop(labels=['popularity'], inplace=True, axis=1)
df.insert(0, 'popularity', temp)
df

# 提取popularity列最大值所在行
df[df['popularity'] == df['popularity'].max()]

# 查看后5列数据
df.tail()

# 删除第7行数据
df.drop(labels=7)

# 添加一行
row = {'grammer': 'Perl', 'popularity': 7}
df.append(row, ignore_index=True)

#
df.sort_values('popularity', inplace=True)
df

# 统计个字符串的长度
df['length'] = df['grammer'].map(lambda x: len(x))
df
