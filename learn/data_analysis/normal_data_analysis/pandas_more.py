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

pd.__version__

# 丛列表创建Series
arr = [0, 1, 2, 3, 4]
df = pd.Series(arr)  # 如果不指定索引，则默认从 0 开始
df

# 丛字典创建Series, 列式创建，key 为索引
d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
df = pd.Series(d)
df

# 从numpy创建
dates = pd.date_range('today', periods=6)  # 定义时间序列作为 index
num_arr = np.random.randn(6, 4)  # 传入 numpy 随机数组
columns = ['A', 'B', 'C', 'D']  # 将列表作为列名
df = pd.DataFrame(num_arr, index=dates, columns=columns)
df

data = {'animal': ['cat', 'cat', 'snake', 'dog', 'dog', 'cat', 'snake', 'cat', 'dog', 'dog'],
        'age': [2.5, 3, 0.5, np.nan, 5, 2, 4.5, np.nan, 7, 3],
        'visits': [1, 3, 2, 3, 2, 3, 1, 1, 2, 1],
        'priority': ['yes', 'yes', 'no', 'yes', 'no', 'no', 'no', 'yes', 'no', 'no']}

labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
df = pd.DataFrame(data, index=labels)
df
df.shape
df.info()

df.loc[:, ['animal', 'age']]
# 选择列名使用[]
temp_df = df[['animal', 'age']]
temp_df

df.loc[df.index[[3, 4, 8]], ['animal', 'age']]

df[df['age'] > 3]

df[df['age'].isnull()]

df[(df['age'] > 2) & (df['age'] < 4)]

# 根据行索引修改列值
df.loc['f', 'age'] = 5
df

df['visits'].sum()
# 行的累计值
df['visits'].cumsum()

df['age'].mean()
df['age'].median()
df['age'].tail()
df.groupby('animal')['age'].mean()

# 根据行索引添加数据
df.loc['k'] = ['dog', 5.5, '0', 2]
# 删除
df = df.drop('k')
df
df['age'] = df['age'].fillna(df['age'].interpolate())
df['animal'].value_counts()
df.groupby('animal').sum()

df.sort_values(by=['age', 'visits'], ascending=[True, True])
df['priority'] = df['priority'].map({'yes': True, 'no': False})
df.shape

df['animal'] = df['animal'].replace('snake', 'python')
df.info()
df['friends'] = pd.Series([i for i in range(0, 11)])
df
temp_list = [i for i in range(0, 11)]
temp_list
df['friends'] = temp_list
# 在做透视表之前，要进行数据清洗
df
# 对每种animal的每种不同数量visits，计算平均age，即，返回一个表格，行是aniaml种类，列是visits数量，表格值是行动物种类列访客数量的平均年龄
df.pivot_table(index='animal', columns='visits', values='age', aggfunc='mean')
df.pivot_table(index='animal')
# 要做聚合的列一定是数值型
pd.pivot_table(df, index=['animal', 'priority'])
