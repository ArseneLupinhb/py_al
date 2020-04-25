# 用数据说话，是数据工作者的意义所在，整个数据分析的过程分为三步：
# 数据获取
# 数据预处理
# 数据可视化及数据分析
import json
import os

import pandas as pd
import requests
from pandas.io.json import json_normalize

os.getcwd()
source_path = os.getcwd() + r'/learn/data_analysis/lzx_data_analysis/work/'

headers = {
	'user-agent': 'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.106 Safari/537.36'}
# 评论地址
url = "http://comment.api.163.com/api/v1/products/a2869674571f77b5a0867c3d71db5856/threads/FASTLQ7I00038FO9/comments/newList?ibc=newspc&limit=30&showLevelThreshold=72&headLimit=1&tailLimit=2&offset={}"
# 循环爬取
df = pd.DataFrame(None)
i = 0
while True:
	ret = requests.get(url.format(str(i * 30)), headers=headers)
	text = ret.text
	result = json.loads(text)
	t = result['comments'].values()
	s = json_normalize(t)
	i += 1
	if len(s) == 0:
		print("爬取结束")
		break
	else:
		df = df.append(s)
		print("第{}页爬取完毕".format(i))

df.to_csv(source_path + 'data.csv', encoding='utf-8')
df.head(5)
df['content']
