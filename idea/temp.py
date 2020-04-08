import copy

a = [1, 2, 3, 4, 5, ['a', 'b']]
# 原始对象
b = a  # 赋值，传对象的引用
c = copy.copy(a)  # 对象拷贝，浅拷贝
d = copy.deepcopy(a)  # 对象拷贝，深拷贝

print(a)
print(b)
print(c)
print(d)

a.append(6)  # 修改对象a
a[5].append('c')  # 修改对象a中的['a','b']数组对象

print(a)
print(b)
print(c)
print(d)
