import time

from bluetooth import *
from wcwidth import wcswidth as ww

alreadyFound = []


def lpad(s, n, c=' '):
	# lpad('你好', 6) => '  \u4f60\u597d'
	return (n - ww(s)) * c + s


def rpad(s, n, c=' '):
	# rpad('你好', 6) => '\u4f60\u597d  '
	return s + (n - ww(s)) * c


def findDevs():
	foundDevs = discover_devices(lookup_names=True)
	for (addr, name) in foundDevs:
		if addr not in alreadyFound:
			# print("[*] Found Bluetooth Device :  " + str(name))
			# print("[+] MAC address :  " + str(addr))
			# print(str(name)+" --- "+str(addr))
			# tplt = "{0:{3}^10}\t{1:{3}^20}"
			# print(tplt.format("名称", "MAC", chr(12288)))
			# print(tplt.format(str(name), str(addr), chr(12288)))
			# print('{0:15} ,{1:5}, {1:20}'.format(str(name), " --- ", str(addr)))
			# print('%-30s%-20s' % (str(name), str(addr)))
			# print('{:<30}{:>25}'.format(str(name), str(addr)))  # {:30d}含义是 右对齐，且占用30个字符位
			print('{} {}'.format(rpad(str(name).strip(), 15), lpad(str(addr).strip(), 20)))
			alreadyFound.append(addr)


while True:
	findDevs()
	time.sleep(5)
