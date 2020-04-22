# 查看数据集种类
import seaborn as sns

sns.get_dataset_names()

import seaborn as sns

# 导出鸢尾花数据集
data = sns.load_dataset('iris')
data.head()

import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

# %matplotlib inline
# 小费数据集
tips = sns.load_dataset('tips')
ax = sns.scatterplot(x='total_bill', y='tip', data=tips)
plt.show()

import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

# %matplotlib inline
# 小费数据集
tips = sns.load_dataset("tips")
ax = sns.barplot(x="day", y="total_bill", data=tips)
plt.show()

import seaborn as sns

sns.set()
import matplotlib.pyplot as plt

# %matplotlib inline
tips = sns.load_dataset("tips")
ax = sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()
