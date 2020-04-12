import gzip
import os
import zipfile


def unzip_data(src_path, target_path):
	'''
	解压原始数据集，将src_path路径下的zip包解压至data目录下
	'''
	if (not os.path.isdir(target_path + "unzip_data")):
		z = zipfile.ZipFile(src_path, 'r')
		z.extractall(path=target_path)
		z.close()


def create_zip(filename, content):
	'''
	创建压缩文件
	:param filename: 文件名
	:param content: 要写入的内容
	:return:
	'''
	g = gzip.GzipFile(filename, 'wb')
	g.write(content.encode())
	g.close()
	print("压缩完成")


def ungz(filename):
	'''
	解压gz文件
	:param filename:
	:return:文件的内容
	'''
	g = gzip.GzipFile(filename, 'rb')
	content = g.read().decode()
	return content


if __name__ == '__main__':
	os.getcwd()
	# create_zip(r'data/test.gz', 'content')
	str1 = ungz(r'data/test.gz')
	print(str1)
