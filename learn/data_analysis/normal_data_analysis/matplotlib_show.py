import matplotlib.pyplot as plt
import numpy as  np
import pandas as pa

# 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# np产生等差数列
x = np.linspace(0, 100, 1000)
y = np.sin(x)

plt.plot(x, y, lw=1.5, ls='-', c='b', label="xy的关系")
# 展示标签 位置
plt.legend(loc='upper center')
plt.show()

x = np.arange(0, 1.1, 0.01)
y = x * 2

# 折线图
plt.figure(figsize=(6.4, 4.84), dpi=100, facecolor='black')
plt.title('这是一幅图')
plt.xlabel('x1')
plt.ylabel('y1')
plt.xticks([])
plt.plot(x, y, label='123')
plt.savefig(r'C:\Users\AL\Desktop\test\text\text_data.pdf')
plt.show()

path_random = r'C:\Users\AL\Desktop\test\test.csv'
test_data_df = pa.read_csv(path_random)
test_data_df.head()
data = test_data_df.groupby('name1').mean()['age']

# 饼状图
plt.pie(x=data, labels=['1', '2', '3', '张si'], radius=1.5)
plt.title("订单金额", pad=30)
plt.show()

labels = ['娱乐', '育儿', '饮食', '房贷', '交通', '其它']
sizes = [2, 5, 12, 70, 2, 9]
# explode : 每一块饼图 离开中心距离，默认值为（0,0），就是不离开中心；
explode = (0, 0, 0, 0.1, 0, 0)
plt.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', shadow=False, startangle=150)
plt.title("饼图示例-8月份家庭支出")
plt.show()

# 条形图
plt.bar(x=np.arange(test_data_df['age'].__len__()), height=test_data_df['age'])
plt.xlabel('姓名')
plt.ylabel('age')
# 添加x轴刻度标签
plt.xticks(np.arange(test_data_df['age'].__len__()), test_data_df['name1'])
plt.show()

N = 5
y = [20, 30, 10, 25, 15]
index = np.arange(N)
plt.bar(x=index, height=y)
plt.show()

# 直方图
plt.hist(x=test_data_df['age'], label=test_data_df['name1'])
plt.show()

# 散点图

plt.scatter(x=test_data_df['name1'].loc[:3], y=test_data_df['age'].loc[:3])
plt.show()

xValue = list(range(0, 1000))
yValue = [x * np.random.rand() for x in xValue]
plt.title(u'散点图示例')
plt.xlabel('x-value')
plt.ylabel('y-label')
plt.scatter(xValue, yValue, s=20, c="#ff1212", marker='o')
plt.show()
