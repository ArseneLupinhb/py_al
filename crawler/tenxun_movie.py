#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: tenxun_movie.py
@time: 2020/5/20 0:31
@desc:
'''
import bs4
import requests

url = 'https://v.qq.com/channel/movie?listpage=1&channel=movie&sort=18&_all=1'
r = requests.get(url)
# 解析网页
soup = bs4.BeautifulSoup(r.content.decode('utf-8'), 'lxml')
dic_film = {}
for i in range(6):
    filter_line_type = 'filter_line_{}'.format(i)
    film_type = [i.findAll('a') for i in soup.find_all('div', class_=[filter_line_type, 'filter_item'])]
    dic_film_type = {}
    for i in film_type[0]:
        dic_film_type[i.text] = i['data-value']
    dic_film[filter_line_type] = dic_film_type

try:
    film_figure = i.findAll('a', class_="figure")
    img_href = film_figure[0]['href']
    film_title = film_figure[0]['title']
    film_caption = film_figure[0].find('div', class_='figure_caption').text
    film_score = film_figure[0].find('div', class_='figure_score').text
    if bool(film_figure[0].find('img', class_="mark_v")):
        is_vip = film_figure[0].find('img', class_="mark_v")['alt']
    else:
        is_vip = ''
    film_detail = i.findAll('div', class_="figure_detail_two_row")
    film_href = film_detail[0].find('a')['href']
    film_actor = getattr(film_detail[0].find('div'), 'text', None)  ###主演
except:
    pass
