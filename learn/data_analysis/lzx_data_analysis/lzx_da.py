import os

import jieba.analyse
import pandas as pd
from pyecharts import options as opts
from pyecharts.charts import Bar

from visualization import text_cloud as tc

# 全局设置
pd.set_option('display.max_columns', None)
os.getcwd()
source_path = os.getcwd() + r'/learn/data_analysis/lzx_data_analysis/work/'

# 数据读取
df = pd.read_csv(source_path + 'data.csv')
df.head()
# 评论去重
df = df.drop_duplicates('commentId').reset_index(drop=True)
# 格式转换 只获取日期
df['new_time'] = df.apply(lambda x: x['createTime'].split(':', 1)[0], axis=1)


def get_comment_word(df):
    # 集合形式存储-去重
    stop_words = set()
    print(stop_words)

    # 加载停用词
    cwd = os.getcwd()
    stop_words_path = source_path + r'stopwords.txt'
    print(stop_words_path)

    with open(stop_words_path, 'r', encoding="ISO-8859-1") as sw:
        for line in sw.readlines():
            stop_words.add(line.strip())
    print(stop_words)

    # 合并评论信息
    df_comment_all = df['content'].str.cat()

    # 使用TF-IDF算法提取关键词
    word_num = jieba.analyse.extract_tags(df_comment_all, topK=300, withWeight=True, allowPOS=())
    print(word_num)
    # 做一步筛选
    word_num_selected = []

    # 筛选掉停用词
    for i in word_num:
        if i[0] not in stop_words:
            word_num_selected.append(i)
        else:
            pass

    return word_num_selected


word_num_selected = get_comment_word(df)
tc.get_text_cloud_by_data(word_num_selected, source_path + r"basic_wordcloud.html")
# (
#     WordCloud()
#     .add(series_name="热点分析", data_pair=word_num_selected, word_size_range=[6, 66])
#     .set_global_opts(
#         title_opts=opts.TitleOpts(
#             title="热点分析", title_textstyle_opts=opts.TextStyleOpts(font_size=23)
#         ),
#         tooltip_opts=opts.TooltipOpts(is_show=True),
#     )
#     .render(source_path + r"basic_wordcloud.html")
# )


df.info()
df = df.groupby(['user.location']).agg({'commentId': 'count'}).reset_index()
df.rename(columns={'place': 'user.location'}, inplace=True)
# 剔除了['上海', '中国', '来自火星', '火星'] 的数据
df = df[~df['user.location'].isin(['上海', '中国', '来自火星', '火星'])]
df = df.sort_values(['commentId'], axis=0, ascending=False)
df_gb_top = df[:15]


def bar_chart() -> Bar:
    c = (
        Bar()
            .add_xaxis(list(df_gb_top['user.location']))
            .add_yaxis("写评论Top15的地区", list(df_gb_top['commentId']))
            .reversal_axis()
            .set_series_opts(label_opts=opts.LabelOpts(position="right"))
            .set_global_opts(title_opts=opts.TitleOpts(title="排行榜"))
            .render(source_path + r'test.html')
    )
    return c


bar_chart()
