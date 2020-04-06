import json
import re
import time
from datetime import datetime
from random import random

import requests


def page_view_csdn():
	url = 'https://blog.csdn.net/qq_33627496/article/details/105273711'
	headers = {'User-Agent': 'Mozilla/5.0 xxxxxx'}
	html = requests.get(url, headers=headers).content.decode('utf-8', 'ignore')
	url_content = re.search(r'阅读数 (.*?)</span>', html, re.S)
	texts = url_content.group()
	texts = texts.replace('</span>', '').replace('阅读数 ', '')
	return int(texts)


# 定时任务
def page_view():
	url = 'https://aistudio.baidu.com/studio/project/detail'
	payload = {"projectId": '368518'}  # 值以字典的形式传入
	response = requests.post(url=url, data=payload)
	resp_data = json.loads(response.text)
	viewCount = resp_data['result']['viewCount']
	return viewCount


if __name__ == '__main__':
	count = 0
	count_csdn = 0
	viewCount = 0
	viewCount_csdn = 0

	page_view_csdn()
	while viewCount_csdn < 200:
		count_csdn = count_csdn + 1
		time_str = int(random())
		viewCount_csdn = page_view_csdn()
		time.sleep(time_str)
		print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count_csdn, time_str)
		print(viewCount_csdn, count_csdn)

	while viewCount < 218:
		count = count + 1
		time_str = int(random() * 10)
		viewCount = page_view()
		time.sleep(time_str)
		print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count, time_str)
		print(viewCount, count)
