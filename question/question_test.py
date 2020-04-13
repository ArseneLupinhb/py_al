import threading
import time


def run(n):
	print("task", n)
	time.sleep(1)
	print('2s')
	time.sleep(1)
	print('1s')
	time.sleep(1)
	print('0s')
	time.sleep(1)


if __name__ == '__main__':
	# Python 多线程
	t1 = threading.Thread(target=run, args=("t1",))
	t2 = threading.Thread(target=run, args=("t2",))
	t1.start()
	t2.start()
