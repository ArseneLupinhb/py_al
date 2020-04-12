import matplotlib.pyplot as plt
import numpy as  np
import pandas as pa

# local_conn = mu.get_conn()
# local_conn = create_engine('mysql+pymysql://root:root@localhost:3306/test?charset=utf8')

# 显示所有列
pa.set_option('display.max_columns', None)
# 显示所有行
pa.set_option('display.max_rows', None)

path = r'C:\Users\AL\Desktop\test\text\text_data.csv'
text_df = pa.read_csv(path)
text_df.info()
text_df.head()
text_df.shape
text_df.count()
temp_data = text_df.groupby('user_id').apply(lambda t: t[t.buy_time == t.buy_time.max()])
temp_data.shape

text_df.info()
use_clo = ['send_text', 's_time', 's_u', 're_u']

temp_data = text_df[text_df['s_u'] == 47]
temp_data.size

text_df.groupby('text_t').size()

temp_data = text_df[use_clo][text_df['s_u'] == 47].head(5)
temp_data.head()
text_df[text_df['s_u'] == 47].head(5)
temp_data = text_df[((text_df['s_u'] == 47) & (text_df['re_u'] == 4003)) | (
		(text_df['s_u'] == 4003) & (text_df['re_u'] == 47))]
temp_data = text_df[(text_df['s_u'] == 47) | (text_df['re_u'] == 4003)]

null_data = text_df[text_df['send_text'].isna()]
not_null_data = text_df[text_df['send_text'].notna()]

temp_data.groupby('text_t').size()
temp_data.groupby('re_u').size()
temp_data.groupby('text_t').count()
temp_data.groupby('text_t')['text_t'].count()
temp_data.groupby('text_t').agg({'s_time': np.mean, 'text_t': np.size})

# text_df.to_sql('text_data', con=local_conn, if_exists='replace')

df1 = pa.DataFrame({'key': ['A', 'B', 'C', 'D'],
                    'value': np.random.randn(4)})

df2 = pa.DataFrame({'key': ['B', 'D', 'D', 'E'],
                    'value': np.random.randn(4)})
pa.merge(df1, df2, on='key')

pa.concat([df1, df2])
pa.concat([df1, df2]).drop_duplicates()

# temp_data.nlargest(10 + 1, columns='re_u').tail(10)

path_random = r'C:\Users\AL\Desktop\test\test.csv'
test_data_df = pa.read_csv(path_random)
test_data_df.head()

# 获取重复值
test_data_df[test_data_df.duplicated()]
np.sum(test_data_df.duplicated())
# 删除重复值
test_data_df.drop_duplicates(inplace=True)

test_data_df.isnull
test_data_df.isna
test_data_df.prod
np.sum(test_data_df.isnull(), axis=1)
test_data_df.apply(lambda x: sum(x.isnull()) / len(x), axis=1)
# 删除缺失值
test_data_df.dropna(inplace=True)
# 填补缺失值
test_data_df.fillna(test_data_df.mean())
# 不同的列按照不同的标准选择缺失值
test_data_df.fillna(value={"name1": 123, "name2": test_data_df.name2.mean()})
# 用前一个填补缺失值
test_data_df.fillna(method="ffill")

# 异常值处理
s_mean = test_data_df['age'].mean()
s_std = test_data_df['age'].std()

s_mean + s_std * 2
s_mean - s_std * 2

test_data_df['age'] > s_mean + s_std * 2
test_data_df['age'] < s_mean - s_std * 2

test_data_df['age'].plot(kind="hist")
plt.show()

text_df.dtypes
text_df.head()
text_df.describe()

# delete a columns
text_df = text_df.drop(['diff_date'], axis=1)
# text_df = text_df.drop(columns=['time_stamp'], axis=1)


# 把时间戳 转换 日期
text_df['s_time'] = pa.to_datetime(text_df['s_time'], unit='s')
# text_df['s_time'] = pa.to_timedelta(text_df['s_time'],unit='s')

# 日期格式转换
# 方法 1
# text_df['s_time'] = text_df['s_time'].apply(lambda x : x.strftime('%Y-%m-%d'))
# 方法 2 参数 M 表示月份，Q 表示季度，A 表示年度，D 表示按天，这几个参数比较常用。
text_df['test_time'] = text_df['s_time'].dt.to_period('D')
text_df['test_price'] = text_df['s_u'].astype(float)

text_df['diff_date'] = pa.datetime.today() - text_df['s_time']
text_df['diff_year'] = pa.datetime.today().year - text_df['s_time'].dt.year

# apply
text_df['total_price'] = text_df[['test_price', 're_u']].apply(np.prod, axis=1)

# groupby
text_df_group = text_df.groupby(by='test_time').count()
text_df_group = text_df.groupby(by='test_time').sum()

# take some columns
col_n = ['test_time', 'test_price', 'total_price']
temp_df = pa.DataFrame(text_df, columns=col_n)
temp_df.head()

temp_df = temp_df.groupby(by='test_time').sum()
temp_df.index = pa.to_datetime(temp_df.index)

# 下面一个减去上面一个数 的 %
temp_df['总价变化率%'] = temp_df['total_price'].pct_change()
temp_df.rename(columns={'总价变化率': '总价变化率%'}, inplace=True)
temp_df.drop(index=[7], axis=1)
# 每5行去一个平均值
temp_df['sma_5'] = temp_df['total_price'].rolling(5).mean()
temp_df['sma_10'] = temp_df['total_price'].rolling(10).mean()

temp_df[['sma_5', 'sma_10']].plot()
plt.show()

# 上下一定某列
temp_df['total_price_before'] = temp_df['总价变化率'].shift(-1)
temp_df['total_price_diff%'] = (temp_df['总价变化率'].shift(-1) - temp_df['total_price']) / temp_df['总价变化率']
