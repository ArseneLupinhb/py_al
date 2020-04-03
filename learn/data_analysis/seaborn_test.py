import matplotlib.pyplot as plt
import pandas as pa
import seaborn as sb

# 中文乱码
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

path_random = r'C:\Users\AL\Desktop\test\test.csv'
test_data_df = pa.read_csv(path_random)
test_data_df.head()
# tick_label x轴对应标签
plt.style.use('seaborn')
plt.bar(test_data_df.index, test_data_df.age, tick_label=test_data_df.name1)
plt.ylabel('age')
plt.show()

# seaborm 用法
# hue 分类变量
sb.barplot('name', 'age', data=test_data_df, hue='type', palette='husl')
# for x, y in enumerate(test_data_df.age):
# 	plt.text(x, y, "%s岁" % y, ha='center', fontsize=12)
plt.legend(bbox_to_anchor=(1.01, 0.85), ncol=1)
plt.show()

# 散点图
sb.scatterplot(x='age', y='score', data=test_data_df, hue="type", style='type')
plt.show()
