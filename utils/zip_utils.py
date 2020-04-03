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
