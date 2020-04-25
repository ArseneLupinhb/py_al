from pyecharts.charts import WordCloud


def get_text_cloud(name, value, path):
    wordcloud = WordCloud(width=1000, height=620)
    wordcloud.add("", name, value, word_size_range=[20, 80])
    generatepath = path
    wordcloud.render(generatepath)
    wordcloud
