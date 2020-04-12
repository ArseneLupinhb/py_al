from pymongo import MongoClient

host = '127.0.0.1'
client = MongoClient(host, 27017)
# 连接mydb数据库,账号密码认证
db = client.admin  # 连接系统默认数据库admin
db.authenticate("adminUser", "adminPass")
collection = db.myset  # myset集合，同上解释
collection.insert({"name": "zhangsan", "age": 18})  # 插入一条数据，如果没出错那么说明连接成功

# #连接mydb数据库,账号密码认证
# db = client.mydb    # mydb数据库，同上解释
# db.authenticate("adminUser", "adminPass")
# collection = db.myset   # myset集合，同上解释
# collection.insert({"name":"zhangsan","age":18})   # 插入一条数据，如果没出错那么说明连接成功
