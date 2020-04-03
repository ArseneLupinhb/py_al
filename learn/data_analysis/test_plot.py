# -*- coding: utf-8 -*-
# coding: utf-8
import decimal
import sys

import matplotlib.pyplot as plt
# Importing necessary Libraries
import numpy as np
import pandas as pd

from sql import sql_serve_utils as ssu

sys.path
# myfont = matplotlib.font_manager.FontProperties(fname=r'C:\Windows\Fonts\simsun.ttc')

sql = "select * from  GGJBCC"
conn = ssu.get_conn_u8_UFDATA_202_2016()

data_list = pd.read_sql_query(sql, conn)

csv_filename = r'E:\Files\test_u81.csv'
# data_list.to_csv(csv_filename, float_format='%.3f', index=False, header=True)

data_list = pd.read_csv(csv_filename, encoding="utf-8")
print(data_list.head())

data_list.dtypes

data_list['supplier_code']
data_supplier = data_list.loc[(data_list['supplier_code'] == 11012) & (data_list['stock_id'] == "3116003070A")]
data_supplier['accumulation_price'] = data_supplier['amount'] * data_supplier['price']
print(data_supplier.head())

data_supplier[1]
data_supplier = data_supplier.reset_index()
data_supplier['id'] = range(len(data_supplier))

data_supplier['index']
id = data_supplier.pop('id')
data_supplier.insert(0, 'id', id)
data_supplier = data_supplier.drop(['index'], axis=1)
plt.plot(data_supplier['amount'], data_supplier['price'])
plt.bar(data_supplier['amount'])

data_list['supplier_name'].value_counts().head(10).plot()
(data_list['supplier_name'].value_counts().head(10) / len(data_list)).plot.bar()
data_list['supplier_name'].value_counts().sort_index().plot.bar()

data_list.loc['amount'].plot(kind='line', label='amount')  # 取出 Algeria 这一行的数据
plt.legend(loc='upper left')

plt.show()

plt.title("stock_amount")
plt.xlabel("x stock")
plt.ylabel("y amount")
plt.plot(data_supplier['stock_name'], data_supplier['amount'])
plt.show()

data_supplier_id = data_list.drop_duplicates('supplier_code')
data_supplier_stock = data_supplier.loc[data_list['存货编码'] == "3116003070A"]
print(data_supplier_stock)

for index, row in data_supplier_id.iterrows():
	print(row["供应商编码"])
# sql = "select * from  GGJBCC  where 供应商编码='"+row["供应商编码"]+"' and 存货编码='82101013613D'   order by '入库日期' asc "

data_stock_pname = data_list.drop_duplicates('存货名称')
print(data_stock_pname)
data_supplier_stock_id = data_list.drop_duplicates('存货编码')
print(data_supplier_stock_id)

data_supplier_id.describe()
print(data_supplier_id.dtypes)

print(data_supplier_id['供应商名称'])

data_supplier_id['原币无税单价'] = pd.DataFrame(data_supplier_id['原币无税单价'], dtype=np.dtype(decimal.Decimal))
print(data_supplier_id.dtypes)

data_supplier_id.drop(['one'], axis=1)

# data_list = data_list.drop_duplicates(
# 　　subset=['YJML','EJML','SJML','WZLB','GGXHPZ','CGMS'], # 去重列，按这些列进行去重
# 　　keep='first' # 保存第一条重复数据
#
# )


# data_list = ssu.query_value(sql,conn)
#
# print(data_list)
# print(len(data_list))
#
# # print(data_list.head())
#
#
# def read_data():
# 	sql = "select * from  GGJBCC"
# 	ssu.get_conn_u8_UFDATA_202_2016()
