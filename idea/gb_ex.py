def one():
	global a
	print("a", a)


def two():
	global a
	print("b", a)


if __name__ == '__main__':
	a = 0
	while a < 10:
		a = a + 1
		one()
		two()
