import re
import time

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent


def getHTMLText(url, header, proxy):
	try:
		r = requests.get(url, headers=header, proxies=proxy, timeout=30)
		r.encoding = r.apparent_encoding
		r.raise_for_status()
		return r.text
	except:
		print('getHTML error')


def HTML2CONTENT(html):
	soup = BeautifulSoup(html, 'html.parser')
	title = soup.find('title').string


def main():
	url = 'https://blog.csdn.net/qq_33627496/article/details/105273711'
	ua = UserAgent()
	proxy = {'http': '109.197.188.12:8080'}
	for i in range(10):
		headers = {'User-Agent': ua.random}
		print(headers)
		html = getHTMLText(url, headers, proxy)
		url_content = re.search(r'阅读数 (.*?)</span>', html, re.S)
		texts = url_content.group()
		texts = texts.replace('</span>', '').replace('阅读数 ', '')
		print(int(texts))
		HTML2CONTENT(html)
		time.sleep(31)


if __name__ == '__main__':
	main()
