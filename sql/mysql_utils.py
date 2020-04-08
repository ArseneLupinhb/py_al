import pymysql


def get_conn():
	# own connect
	connect = pymysql.connect('127.0.0.1', 'root', 'root', 'test')
	if connect:
		print("连接成功")
	return connect


def query_value(sql):
	"""

	Args:
		sql (str): sql words


	Returns:
		list_tuple(list[tuple]): list of the dataset
	"""
	conn = get_conn()
	cursor = conn.cursor()
	cursor.execute(sql)
	list_tuple = cursor.fetchall()
	cursor.close()
	conn.close()
	return list_tuple


def no_query(sql):
	'''

	Args:
		sql(str): sql word

	Returns:
		rows(int): the effect rows

	'''
	conn = get_conn()
	cursor = conn.cursor()
	rows = cursor.execute(sql)
	conn.commit()  # 提交
	cursor.close()
	conn.close()
	return rows


def no_query(sql, values):
	'''

	Args:
		sql (str): sql word
		values(tuple): data of tuple

	Returns:
		rows(int): the effect rows
	'''
	conn = get_conn()
	cursor = conn.cursor()
	rows = cursor.execute(sql, values)

	conn.commit()
	cursor.close()
	conn.close()
	return rows
