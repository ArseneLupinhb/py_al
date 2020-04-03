from pyecharts import WordCloud


def get_text_cloud(name, value, generate_path):
	wordcloud = WordCloud(width=1000, height=620)
	wordcloud.add("", name, value, word_size_range=[20, 80])
	wordcloud.render(generate_path)
	# wordcloud.render(r'D:\source\repos_branch_2\Algorithms\Python-master\test\word.html')
	wordcloud
