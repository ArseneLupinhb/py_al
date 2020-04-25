import numpy as np

temp_na = np.array([1, 2, 3, 4, 5, 6])
temp_na

temp_na.shape

# 基本统计
temp_na.max()
temp_na.min()
temp_na.mean()
temp_na.sum()

# np reshape 改变矩阵形状
temp_na = temp_na.reshape(2, 3)
temp_na

# tensor 切片
# 所有行的前面两列
temp_na[:, :2]

# 矩阵计算
np.random([3, 5])
na1 = np.random.rand(2, 3)
na2 = np.random.rand(3, 5)
na1
na2
np.dot(na1, na2)

# 创建各种矩阵
zeroarray = np.zeros((2, 3))
print(zeroarray)

onearray = np.ones((3, 4), dtype='int64')
print(onearray)

emptyarray = np.empty((3, 4))
print(emptyarray)

array = np.arange(10, 31, 5)
print(array)

# 矩阵的na的属性
array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
print(array)
# 数组维度
print(array.ndim)
# 数组形状
print(array.shape)
# 数组元素个数
print(array.size)
# 数组元素类型
print(array.dtype)

# 改变na的形状
array1 = np.arange(6).reshape([2, 3])
print(array1)

array2 = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int64).reshape([3, 2])
print(array2)

# 矩阵的计算
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.ones([2, 3], dtype=np.int64)
print(arr1)
print(arr2)

# 矩阵的基本运算
print(arr1 + arr2)
print(arr1 - arr2)
print(arr1 * arr2)
print(arr1 / arr2)
# 平方
print(arr1 ** 2)

# 矩阵的乘法
arr3 = np.array([[1, 2, 3], [4, 5, 6]])
arr4 = np.ones([3, 2], dtype=np.int64)
print(arr3)
print(arr4)
print(np.dot(arr3, arr4))

# np的其他统计分析函数
print(arr3)
print(np.sum(arr3, axis=1))  # axis=1,每一行求和 axie=0,每一列求和
print(np.max(arr3))
print(np.min(arr3))
print(np.mean(arr3))
print(np.argmax(arr3))
print(np.argmin(arr3))

# 矩阵的转置
arr3_tran = arr3.transpose()
print(arr3_tran)
print(arr3.flatten())

# 矩阵的索引和切片
arr5 = np.arange(0, 6).reshape([2, 3])
print(arr5)
print(arr5[1])
print(arr5[1][2])
print(arr5[1, 2])
print(arr5[1, :])
print(arr5[:, 1])
print(arr5[1, 0:2])
