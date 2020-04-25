from pyecharts.charts import WordCloud


def get_text_cloud_by_label_value(name, value, generate_path):
    wordcloud = WordCloud(width=1000, height=620)
    wordcloud.add("", name, value, word_size_range=[20, 80])
    wordcloud.render(generate_path)
    wordcloud


def get_text_cloud_by_data(data, generate_path):
    wordcloud = WordCloud()
    wordcloud.add("", data, word_size_range=[20, 80])
    wordcloud.render(generate_path)
    wordcloud
