# 打乱列表
import random

list_data = [20, 16, 10, 5];
random.shuffle(list_data)
print("随机排序列表 : ", list_data)

# np.reshape
import numpy as np

np_data = np.arange(12)
print(np_data)
np.reshape(np_data, [3, 4]).astype('float32')

# list 转换成 向量
array_data = [1, 2, 3]
np.array(array_data)

# np 数组常用函数
print(np.arange(0, 7, 1, dtype=np.int16))  # 0为起点，间隔为1时可缺省(引起歧义下不可缺省)
print(np.ones((2, 3, 4), dtype=np.int16))  # 2页，3行，4列，全1，指定数据类型
print(np.zeros((2, 3, 4)))  # 2页，3行，4列，全0
print(np.empty((2, 3)))  # 值取决于内存
print(np.arange(0, 10, 2))  # 起点为0，不超过10，步长为2
print(np.linspace(-1, 2, 5))  # 起点为-1，终点为2，取5个点
print(np.random.randint(0, 3, (2, 3)))  # 大于等于0，小于3，2行3列的随机整数

# enumerate
seasons = ['Spring', 'Summer', 'Fall', 'Winter']
list(enumerate(seasons))
for season_id, season in enumerate(seasons):
	print(id, season)

import numpy as np

# ps：这里的num的绝对值小于等于x中元素的个数
# 当num>=0时，np.argsort()[num]就可以理解为y[num];
# 当num<0时，np.argsort()[num]就是把数组y的元素反向输出，例如np.argsort()[-1]即输出x中最大值对应的index，np.argsort()[-2]
x = np.array([1, 4, 3, -1, 6, 9])
print(x)
# array([3, 0, 2, 1, 4, 5] 元素从小到大的索引
# 取出最大值
np.argsort(x)[-1]
