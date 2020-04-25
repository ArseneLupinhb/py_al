import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.preprocessing as preprocessing
from sklearn import linear_model
from sklearn import model_selection
from sklearn.ensemble import RandomForestRegressor

print(os.getcwd())

# data_path = r'C:\Users\ArseneLupin\Desktop\OrderType.csv'
data_path = os.getcwd() + r'\dataset\train.csv'
data_train = pd.read_csv(data_path)
data_train.shape
# 使用 len 和 df
for i in range(len(data_train)):
    for j in range(12):
        cur_data = data_train.loc[i][j]
        print(cur_data)

# 使用 .iteriterms()
for i, series in data_train.iteritems():
    # print(i, ":", type(series))
    # print(i + '  : ' + series)
    print(series)

print(data_train.head())
# data.select_dtypes()

data_train.info()

# show the data
fig = plt.figure()
plt.subplot2grid((2, 3), (0, 0))
# the Survived is the y
data_train.Survived.value_counts().plot(kind='bar')  # 柱状图
plt.title(u'live num')
plt.ylabel(u'num')

plt.subplot2grid((2, 3), (0, 1))
data_train.Pclass.value_counts().plot(kind="bar")
plt.ylabel(u"num")
plt.title(u"passenger class")

plt.subplot2grid((2, 3), (0, 2))
plt.scatter(data_train.Age, data_train.Survived)
plt.ylabel(u"live")  # 设定纵坐标名称
plt.grid(b=True, which='major', axis='y')
plt.title(u"live by age")

plt.subplot2grid((2, 3), (1, 0), colspan=2)
data_train.Age[data_train.Pclass == 1].plot(kind='kde')
data_train.Age[data_train.Pclass == 2].plot(kind='kde')
data_train.Age[data_train.Pclass == 3].plot(kind='kde')
plt.xlabel(u"age")  # plots an axis lable
plt.ylabel(u"density")
plt.title(u"passerger class by age ")
plt.legend((u'1 class ', u'2 class', u'3 class'), loc='best')  # sets our legend for our graph.

plt.subplot2grid((2, 3), (1, 2))
data_train.Embarked.value_counts().plot(kind='bar')
plt.title(u"num at embarked ")
plt.ylabel(u"num")

# 看看各乘客等级的获救情况
fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数

Survived_0 = data_train.Pclass[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Pclass[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'live': Survived_1, u'unlive': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"live by class")
plt.xlabel(u"passenger class")
plt.ylabel(u"num")
plt.show()

# the baby all lived the old lives little than the new
Survived_age = data_train.Age[data_train.Survived == 1].value_counts()
unSurvived_age = data_train.Age[data_train.Survived == 0].value_counts()
temp_data = {u'live': Survived_age, u'unlive': unSurvived_age}
df = pd.DataFrame(temp_data)
df.plot(kind='bar', stacked=True)
plt.title(u'live by age')
plt.ylabel(u'num')
plt.xlabel(u'age')

print(df)
plt.show()

print(df.head())
print(df.size)

print(df.shape)
df.describe()
df.get_dtype_counts()
df.idxmax()
df.idxmin()
df.info()
data_list = df.iteritems

# 看看各性别的获救情况
fig = plt.figure()
# fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# most of the people died and in the live ,women is more
Survived_m = data_train.Survived[data_train.Sex == 'male'].value_counts()
Survived_f = data_train.Survived[data_train.Sex == 'female'].value_counts()
df = pd.DataFrame({u'man': Survived_m, u'female': Survived_f})
df.plot(kind='bar', stacked=True)
plt.title(u"survied by sex")
plt.xlabel(u"sex")
plt.ylabel(u"num")
plt.show()

# 然后我们再来看看各种舱级别情况下各性别的获救情况
fig = plt.figure()
# fig.set(alpha=0.65) # 设置图像透明度，无所谓
plt.title(u"surviced by class and sex")

ax1 = fig.add_subplot(141)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                            label="female highclass",
                                                                                            color='#FA2479')
ax1.set_xticklabels([u"unlive", u"live"], rotation=0)
ax1.legend([u"femall/high class"], loc='best')

ax2 = fig.add_subplot(142, sharey=ax1)
data_train.Survived[data_train.Sex == 'female'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                            label='female, low class',
                                                                                            color='pink')
ax2.set_xticklabels([u"live", u"unlive"], rotation=0)
plt.legend([u"female/low class"], loc='best')

ax3 = fig.add_subplot(143, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass != 3].value_counts().plot(kind='bar',
                                                                                          label='male, high class',
                                                                                          color='lightblue')
ax3.set_xticklabels([u"unlive", u"live"], rotation=0)
plt.legend([u"man/high class"], loc='best')

ax4 = fig.add_subplot(144, sharey=ax1)
data_train.Survived[data_train.Sex == 'male'][data_train.Pclass == 3].value_counts().plot(kind='bar',
                                                                                          label='male low class',
                                                                                          color='steelblue')
ax4.set_xticklabels([u"unlive", u"live"], rotation=0)
plt.legend([u"man/low class"], loc='best')

plt.show()

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# x is the Embarked and y is the num and in the dataframe the row is the Embarked and the clo is the num
Survived_0 = data_train.Embarked[data_train.Survived == 0].value_counts()
Survived_1 = data_train.Embarked[data_train.Survived == 1].value_counts()
df = pd.DataFrame({u'live': Survived_1, u'unlive': Survived_0})
df.plot(kind='bar', stacked=True)
plt.title(u"live by Embarked")
plt.xlabel(u"Embarked")
plt.ylabel(u"num")

plt.show()
df

# 堂兄妹个数  这就是特征工程，就是属性队结果的影响就是侦探的直觉，所以这些东西吸引我的原因，因为这些东西 在一开始就是和我在一起。很久很久以前。
g = data_train.groupby(['SibSp', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df

g = data_train.groupby(['Parch', 'Survived'])
df = pd.DataFrame(g.count()['PassengerId'])
df

temp_data = data_train.Parch
temp_data
data_train.head()

# ticket是船票编号，应该是unique的，和最后的结果没有太大的关系，先不纳入考虑的特征范畴把
# cabin只有204个乘客有值，我们先看看它的一个分布
temp_data = data_train.Cabin.value_counts()
temp_data

fig = plt.figure()
fig.set(alpha=0.2)  # 设定图表颜色alpha参数

# cabin 客舱
Survived_cabin = data_train.Survived[pd.notnull(data_train.Cabin)].value_counts()
Survived_nocabin = data_train.Survived[pd.isnull(data_train.Cabin)].value_counts()
df = pd.DataFrame({u'yes cabin': Survived_cabin, u'no cabin': Survived_nocabin}).transpose()
df.plot(kind='bar', stacked=True)
plt.title(u"live by cabin")
plt.xlabel(u"Cabin exit")
plt.ylabel(u"num")
plt.show()


### 使用 RandomForestClassifier 填补缺失的年龄属性
def set_missing_ages(df):
    # 把已有的数值型特征取出来丢进Random Forest Regressor中
    age_df = df[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]

    # 乘客分成已知年龄和未知年龄两部分
    known_age = age_df[age_df.Age.notnull()].as_matrix()
    unknown_age = age_df[age_df.Age.isnull()].as_matrix()

    # y即目标年龄
    y = known_age[:, 0]

    # X即特征属性值
    X = known_age[:, 1:]

    # fit到RandomForestRegressor之中
    rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
    rfr.fit(X, y)

    # 用得到的模型进行未知年龄结果预测
    predictedAges = rfr.predict(unknown_age[:, 1::])

    # 用得到的预测结果填补原缺失数据
    df.loc[(df.Age.isnull()), 'Age'] = predictedAges

    return df, rfr


def set_Cabin_type(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


data_train, rfr = set_missing_ages(data_train)
data_train = set_Cabin_type(data_train)

age_df = data_train[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
known_age = age_df[age_df.Age.notnull()].values
unknown_age = age_df[age_df.Age.isnull()].values

# y即目标年龄
y = known_age[:, 0]
X = known_age[:, 1:]

# fit到RandomForestRegressor之中
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(X, y)

predictedAges = rfr.predict(unknown_age[:, 1::])
# 用得到的预测结果填补原缺失数据
df.loc[(df.Age.isnull()), 'Age'] = predictedAges

dummies_Cabin = pd.get_dummies(data_train['Cabin'], prefix='Cabin')

dummies_Embarked = pd.get_dummies(data_train['Embarked'], prefix='Embarked')

dummies_Sex = pd.get_dummies(data_train['Sex'], prefix='Sex')

dummies_Pclass = pd.get_dummies(data_train['Pclass'], prefix='Pclass')

df = pd.concat([data_train, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df

scaler = preprocessing.StandardScaler()
age_scale_param = scaler.fit(df['Age'])
df['Age_scaled'] = scaler.fit_transform(df['Age'], age_scale_param)
fare_scale_param = scaler.fit(df['Fare'])
df['Fare_scaled'] = scaler.fit_transform(df['Fare'], fare_scale_param)
df

# 用正则取出我们要的属性值
train_df = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
train_np = train_df.as_matrix()

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到RandomForestRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(X, y)

clf

data_test = pd.read_csv(os.getcwd() + r'\dataset\test.csv')
data_test.loc[(data_test.Fare.isnull()), 'Fare'] = 0
# 接着我们对test_data做和train_data中一致的特征变换
# 首先用同样的RandomForestRegressor模型填上丢失的年龄
tmp_df = data_test[['Age', 'Fare', 'Parch', 'SibSp', 'Pclass']]
null_age = tmp_df[data_test.Age.isnull()].values
# 根据特征属性X预测年龄并补上
X = null_age[:, 1:]
predictedAges = rfr.predict(X)
data_test.loc[(data_test.Age.isnull()), 'Age'] = predictedAges

data_test = set_Cabin_type(data_test)
dummies_Cabin = pd.get_dummies(data_test['Cabin'], prefix='Cabin')
dummies_Embarked = pd.get_dummies(data_test['Embarked'], prefix='Embarked')
dummies_Sex = pd.get_dummies(data_test['Sex'], prefix='Sex')
dummies_Pclass = pd.get_dummies(data_test['Pclass'], prefix='Pclass')

df_test = pd.concat([data_test, dummies_Cabin, dummies_Embarked, dummies_Sex, dummies_Pclass], axis=1)
df_test.drop(['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1, inplace=True)
df_test['Age_scaled'] = scaler.fit_transform(df_test['Age'], age_scale_param)
df_test['Fare_scaled'] = scaler.fit_transform(df_test['Fare'], fare_scale_param)
df_test

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(os.getcwd() + r'\dataset\logistic_regression_predictions.csv', index=False)

pd.DataFrame({"columns": list(train_df.columns)[1:], "coef": list(clf.coef_.T)})

# 简单看看打分情况
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
all_data = df.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
X = all_data.values[:, 1:]
y = all_data.values[:, 0]
print(model_selection.cross_val_score(clf, X, y, cv=5))

# 分割数据，按照 训练数据:cv数据 = 7:3的比例
split_train, split_cv = model_selection.train_test_split(df, test_size=0.3, random_state=0)
train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
# 生成模型
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])

# 对cross validation数据进行预测

cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
predictions = clf.predict(cv_df.values[:, 1:])

origin_data_train = pd.read_csv(data_path)
bad_cases = origin_data_train.loc[
    origin_data_train['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
bad_cases


# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """
    train_sizes, train_scores, test_scores = model_selection(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
        plt.gca().invert_yaxis()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


plot_learning_curve(clf, u"学习曲线", X, y)

from sklearn.ensemble import BaggingRegressor

train_df = df.filter(
    regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
train_np = train_df.values

# y即Survival结果
y = train_np[:, 0]

# X即特征属性值
X = train_np[:, 1:]

# fit到BaggingRegressor之中
clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
bagging_clf = BaggingRegressor(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                               bootstrap_features=False, n_jobs=-1)
bagging_clf.fit(X, y)

test = df_test.filter(regex='Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass.*|Mother|Child|Family|Title')
predictions = bagging_clf.predict(test)
result = pd.DataFrame({'PassengerId': data_test['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
result.to_csv(os.getcwd() + r'\dataset\logistic_regression_bagging_predictions.csv', index=False)
