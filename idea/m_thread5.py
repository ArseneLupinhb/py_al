#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: m_thread5.py
@time: 2020/5/13 0:32
@desc:
'''

import threading


def sing(num):
    for i in range(num):
        print("sing%d" % i)
        # time.sleep(0.5)


def dance(num):
    for i in range(num):
        print("dancing%d" % i)
        # time.sleep(0.5)


def main():
    """创建启动线程"""
    t_sing = threading.Thread(target=sing, args=(5,))
    t_dance = threading.Thread(target=dance, args=(6,))
    t_sing.start()
    t_dance.start()


if __name__ == '__main__':
    main()
