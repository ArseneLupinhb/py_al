import pymssql  # 引入pymssql模块


def conn():
	connect = pymssql.connect('(local)', 'sa', 'sa-123456', 'test')  # 服务器名,账户,密码,数据库名
	if connect:
		print("连接成功!")
	return connect


def no_query(sql):
	connect = conn()
	with connect.cursor() as cursor:
		cursor.execute(sql)
		count = connect.commit()
		print(count)
		connect.close()
		return count


def select(sql):
	connect = conn()
	cursor = connect.cursor()
	cursor.execute(sql)

	row = cursor.fetchone()  # 读取查询结果,
	while row:  # 循环读取所有结果
		print("Id=%s, name1=%s, name2=%s" % (row[0], row[1], row[2]))  # 输出结果
		row = cursor.fetchone()

	cursor.close()
	connect.close()


sql_insert = "insert into names (Id, name1,name2)values('1006', '张si','3')"
no_query(sql_insert)

sql_select = "select * from names "
select(sql_select)

connect = conn()
# 查询操作
conn.execute_query('SELECT * FROM names WHERE Id=%s', '1006')
for row in conn:
	print("ID=%d, Name=%s" % (row['id'], row['name1']))
