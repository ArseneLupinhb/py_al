import pandas as pa
import os

import jieba
import pandas as pa
from pyecharts import WordCloud

from utils import json_utils as ju
from visualization import text_cloud as tc

path = os.getcwd() + r"/test/text_data.csv"
text_data_df = pa.read_csv(path)
text_data_df.describe()
text_data_df.size
temp_data = text_data_df.head(100)

content = text_data_df["CONTENT"]
word_str = ""
for index, row in text_data_df.iterrows():
	# print(row['CONTENT'])
	word_str += str(row['CONTENT']).strip()
word_str

filename = os.getcwd() + r"/test/text_data.txt"
with open(filename, 'r') as file_object:
	word_str = file_object.read()
with open(filename, 'w') as file_object:
	file_object.write(word_str)

# segment = jieba.cut(content.to_string())
segment = jieba.cut(word_str)


def stopwordslist(stop_word_path):
	stopwords = [line.strip() for line in open(stop_word_path, encoding='UTF-8').readlines()]
	return stopwords


stop_word_path = r'C:\Users\AL\Desktop\test\text\stopwords.txt'
stopwords = stopwordslist(stop_word_path)

# 去停用词,统计词频
word_ = {}
name = []
value = []
for word in segment:
	if word.strip() not in stopwords:
		if len(word) > 1:
			if word != '\t':
				if word != '\r\n':
					# 计算词频
					if word in word_:
						word_[word] += 1

					else:
						word_[word] = 1

print(word_)
len(word_)
del word_['nbsp']
ju.write_json(word_, os.getcwd() + r"/test/text_data.json")
word_ = sorted(word_.items(), key=lambda x: x[1], reverse=True)
dic_temp = {}
for word in word_:
	dic_temp[word[0]] = word[1]
ju.write_json(dic_temp, os.getcwd() + r"/test/text_data.json")
dic_temp["姜明"]
for word in word_:
	name.append(word[0])
	value.append(word[1])

name.index("雪艳姐")
value[184]
generatepath = os.getcwd() + r"/test/test_cloud.html"
name[:200]
value[:200]
tc.get_text_cloud(name[:200], value[:200], generatepath)

wordcloud = WordCloud(width=1000, height=620)
wordcloud.add("", name[:100], value[:100], word_size_range=[20, 80])
wordcloud.render(generatepath)
wordcloud
