import json
import time
from datetime import datetime
from random import random

import requests


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
	viewCount = 0
	while viewCount < 120:
		count = count + 1
		time_str = int(random() * 10)
		viewCount = page_view()
		time.sleep(time_str)
		print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"), count, time_str)
		print(viewCount, count)
