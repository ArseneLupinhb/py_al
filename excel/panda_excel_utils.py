# 导入需要使用到的数据模块
import pandas as pa


def write_excel(data, path):
	writer = pa.ExcelWriter(path)
	data.to_excel(writer, index=False, encoding='utf-8', sheet_name='Sheet1')
	# 生成csv文件
	# df.to_csv(r'./1.csv',columns=['save1','save2'],index=False,sep=',')
	writer.save()


def write_csv(data, path):
	# 生成csv文件
	data.to_csv(path, encoding='utf-8', index=False, sep=',')
