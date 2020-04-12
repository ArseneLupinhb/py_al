import copy

from utils import zip_utils as zu

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

from time import sleep


def progress(percent=0, width=30):
    left = width * percent // 100
    right = width - left
    print('\r[', '#' * left, ' ' * right, ']',
          f' {percent:.0f}%',
          sep='', end='', flush=True)


for i in range(101):
    progress(i)
    sleep(0.1)

a_list = 'avbs'
a_list[-2:]

# Python字符编码转换
s = '匆匆'
print(s)
s1 = s.encode().decode("utf-8")  # utf-8 转成 Unicode，decode(解码)需要注明当前编码格式
print(s1, type(s1))

s2 = s1.encode("gbk")  # unicode 转成 gbk，encode(编码)需要注明生成的编码格式
print(s2, type(s2))

s3 = s1.encode("utf-8")  # unicode 转成 utf-8，encode(编码)注明生成的编码格式
print(s3, type(s3))

zu.create_zip('test.txt', 'hello world')
