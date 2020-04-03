import os
from urllib.request import urlopen

from utils.fileUtils import file_utils as fu

os.getcwd()

url = "https://weibo.com/u/5571549493/home?wvr=5&sudaref=login.sina.com.cn"
# 发送请求
response = urlopen(url)
# 读取内容
info = response.read()
# 打印内容
info
# print(info.decode())
path = os.getcwd() + r'\utils\test.html'
fu.write_file(path, str(info))
username = '15671558130'
password = 'AL-15671558130'

# 打印状态码
print(response.getcode())
# # 打印真实url
print(response.geturl())
# # 打印响应头
print(response.info())
