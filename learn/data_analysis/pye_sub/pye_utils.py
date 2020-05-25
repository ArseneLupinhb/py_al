#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: pye_utils.py
@time: 2020/5/21 23:48
@desc:
'''

import os

from pyecharts import options as opts
from pyecharts.charts import Bar
from pyecharts.charts import Line
from pyecharts.charts import Map
from pyecharts.globals import CurrentConfig
from pyecharts.globals import ThemeType
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot

CurrentConfig.ONLINE_HOST = "http://127.0.0.1:8000/assets/"


def pye_bar(x_data, y_data, ylabel):
    bar = Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
    bar.add_xaxis(x_data)
    bar.add_yaxis(ylabel, y_data)
    # render 会生成本地 HTML 文件，默认会在当前目录生成 render.html 文件
    # 也可以传入路径参数，如 bar.render("mycharts.html")
    bar.render()


def pye_bar_phoot(x_data, y_data, ylabel):
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
            .add_xaxis(x_data)
            .add_yaxis(ylabel, y_data)
    )
    make_snapshot(snapshot, bar.render(), "bar.png")


def get_line():
    lin = (
        Line(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
            .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
            .add_yaxis("商家A", [5, 20, 36, 10, 75, 90], is_smooth=True)
            .add_yaxis("商家B", [15, 6, 45, 20, 35, 66], is_smooth=True)
            .set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"),
                             xaxis_opts=opts.AxisOpts(name='日期'),
                             yaxis_opts=opts.AxisOpts(name='数量', max_=16000, min_=1, type_="log",  # 坐标轴配置项
                                                      splitline_opts=opts.SplitLineOpts(is_show=True),  # 分割线配置项
                                                      axisline_opts=opts.AxisLineOpts(is_show=True))
                             )

    )
    lin.render()


if __name__ == '__main__':
    os.chdir(r'D:\soft_own\source\py_al\learn\data_analysis\pye_sub')
    os.getcwd()
    x_data = ["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"]
    y_data = [5, 20, 36, 10, 75, 90]
    pye_bar(x_data, y_data, '商家')

    # 使用 snapshot-selenium 渲染图片
    pye_bar_phoot(x_data, y_data, '商家')

    # bar demo2
    bar = (
        Bar(init_opts=opts.InitOpts(theme=ThemeType.CHALK))
            .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
            .add_yaxis("商家A", [5, 20, 36, 10, 75, 90], is_smooth=True)
            .add_yaxis("商家B", [15, 6, 45, 20, 35, 66], is_smooth=True)
            .set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
    )
    bar.render()

    get_line()

    value = [95.1, 23.2, 43.3, 66.4, 88.5]
    attr = ["China", "Canada", "Brazil", "Russia", "United States"]
    map = Map("世界地图示例")
    map.add("", attr, value, maptype="world", is_visualmap=True, visual_text_color='#000')
    map.render('Map-World.html')
