# 打乱列表
import random

list_data = [20, 16, 10, 5];
random.shuffle(list_data)
print("随机排序列表 : ", list_data)

# np.reshape
import numpy as np

np_data = np.arange(12)
print(np_data)
np.reshape(np_data, [3, 4])
