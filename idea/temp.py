import copy
import os

import pandas as pd

from utils import zip_utils as zu

os.getcwd()
source_path = os.getcwd() + r'/idea/work/'

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

# sigmoid 函数作图
import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    # 直接返回sigmoid函数
    return 1. / (1. + np.exp(-x))


# param:起点，终点，间距
x = np.arange(-8, 8, 0.2)
y = sigmoid(x)
plt.plot(x, y)
plt.show()

res = pd.DataFrame(columns=('lib', 'qty1', 'qty2'))
res = res.append([{'qty1': 10.0}], ignore_index=True)
print(res.head())
res.to_csv('result.csv')


# windows 发出声音
def raise_alarm(voice):
    import winsound
    # winsound.Beep(500, 1000)
    winsound.MessageBeep(500)
    winsound.MessageBeep(500)


raise_alarm(voice="hello")

import plotly.express as px
import matplotlib.pyplot as plt

df = px.data.tips()
fig = px.sunburst(df, path=['day', 'time', 'sex'], values='total_bill')
fig.show()
plt.show()

import plotly.graph_objects as go
import urllib, json

url = 'https://raw.githubusercontent.com/plotly/plotly.js/master/test/image/mocks/sankey_energy.json'
response = urllib.request.urlopen(url)
data = json.loads(response.read())

# override gray link colors with 'source' colors
opacity = 0.4
# change 'magenta' to its 'rgba' value to add opacity
data['data'][0]['node']['color'] = ['rgba(255,0,255, 0.8)' if color == "magenta" else color for color in
                                    data['data'][0]['node']['color']]
data['data'][0]['link']['color'] = [data['data'][0]['node']['color'][src].replace("0.8", str(opacity))
                                    for src in data['data'][0]['link']['source']]

fig = go.Figure(data=[go.Sankey(
    valueformat=".0f",
    valuesuffix="TWh",
    # Define nodes
    node=dict(
        pad=15,
        thickness=15,
        line=dict(color="black", width=0.5),
        label=data['data'][0]['node']['label'],
        color=data['data'][0]['node']['color']
    ),
    # Add links
    link=dict(
        source=data['data'][0]['link']['source'],
        target=data['data'][0]['link']['target'],
        value=data['data'][0]['link']['value'],
        label=data['data'][0]['link']['label'],
        color=data['data'][0]['link']['color']
    ))])

fig.update_layout(
    title_text="Energy forecast for 2050<br>Source: Department of Energy & Climate Change, Tom Counsell via <a href='https://bost.ocks.org/mike/sankey/'>Mike Bostock</a>",
    font_size=10)
fig.show()

import plotly.graph_objects as go

categories = ['processing cost', 'mechanical properties', 'chemical stability',
              'thermal stability', 'device integration']

fig = go.Figure()

fig.add_trace(go.Scatterpolar(
    r=[1, 5, 2, 2, 3],
    theta=categories,
    fill='toself',
    name='Product A'
))
fig.add_trace(go.Scatterpolar(
    r=[4, 3, 2.5, 1, 2],
    theta=categories,
    fill='toself',
    name='Product B'
))

fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 5]
        )),
    showlegend=False
)

fig.show()

# python 列表表达式
l1 = [i for i in range(1, 101)]
print(l1)

print([i for i in range(0, 11)])
print([i * 2 for i in range(0, 11)])

# step = 2 输出0 100 的所有的偶数
print([i for i in range(0, 101, 2)])
# 通过添加if判断条件 输出偶数
print([i for i in range(0, 101) if i % 2 == 0])

print([i for i in range(0, 101) if i > 4])

print([f'python---{i}' for i in range(1, 101)])
l1 = ['太白金星', 'fdsaf', 'alex', 'sb', 'ab']
l2 = [i.upper() for i in l1 if len(i) > 3]
print(l2)

names = [['Tom', 'Billy', 'Jefferson', 'Andrew', 'Wesley', 'Steven', 'Joe'],
         ['Alice', 'Jill', 'Ana', 'Wendy', 'Jennifer', 'Sherry', 'Eva']]
# 第一个是最外层
print([j for i in names for j in i if j.count('e') >= 2])

for i in range(1, 10):
    for j in range(1, i + 1):
        print('{}x{}={}\t'.format(j, i, i * j), end='')
    print()

a = 255
b = 255
a is b

c = 257
d = 257
c is d

pd.show_versions()

import wget

url = 'http://www.futurecrew.com/skaven/song_files/mp3/razorback.mp3'
filename = wget.download(url)

animals = ['cat', 'dog', 'tiger', 'snake', 'mouse', 'bird']
print(animals[2:5])
print(animals[-1:])
print(animals[-3:-1])
print(animals[-5:-1:2])
print(animals[::2])

# 列表排序
sorted(animals)


class Animal:

    def __init__(self, name):
        self.name = name
        print('动物名称实例化')

    def eat(self):
        print(self.name + '要吃东西啦！')

    def drink(self):
        print(self.name + '要喝水啦！')


cat = Animal('miaomiao')
print(cat.name)
cat.eat()
cat.drink()


class Person:
    def __init__(self, name):
        self.name = name
        print('调用父类构造函数')

    def eat(self):
        print('调用父类方法')


class Student(Person):  # 定义子类
    def __init__(self):
        print('调用子类构造方法')

    def study(self):
        print('调用子类方法')


s = Student()  # 实例化子类
s.study()  # 调用子类的方法
s.eat()  # 调用父类方法
