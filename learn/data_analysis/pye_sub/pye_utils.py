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
        Bar(init_opts=opts.InitOpts(theme=ThemeType.DARK))
            .add_xaxis(["衬衫", "羊毛衫", "雪纺衫", "裤子", "高跟鞋", "袜子"])
            .add_yaxis("商家A", [5, 20, 36, 10, 75, 90])
            .add_yaxis("商家B", [15, 6, 45, 20, 35, 66])
            .set_global_opts(title_opts=opts.TitleOpts(title="主标题", subtitle="副标题"))
    )
    bar.render()
