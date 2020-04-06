import itertools
import json
import re

import requests

from utils import json_utils as ju


# 嵌套列表转换
def change_mlist_to_list():
	Input = ["a", "b", ["c"], [[], "d"]]
	out = list(itertools.chain.from_iterable(Input))
	print(out)
	for i in out:
		if i == []:
			out.remove(i)
	print(out)


# get movie_info from doupan
def get_movie_info():
	headers = {'User-Agent': 'Mozilla/5.0 xxxxxx'}
	basel = 'https://movie.douban.com/subject/1292213/'
	html = requests.get(basel, headers=headers).content.decode('utf-8', 'ignore')
	url_content = re.search(r'"@context": "http://schema.org",(.*?)"ratingValue": "9.2"', html, re.S)
	texts = url_content.group()  # 获取匹配正则表达式的整体结果
	texts = str("{" + texts + "}}")
	# important
	data = json.loads(texts, strict=False)
	movie_info = {'name': data['name'], 'author': data['author'], 'actor': data['actor'], 'director': data['director']}
	print(movie_info)
	ju.write_json(data, r'data/data.json')


if __name__ == '__main__':
	change_mlist_to_list()
	get_movie_info()
