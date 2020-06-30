#!/usr/bin/env python
# encoding: utf-8
'''
@author: al
@file: text_classification.py
@time: 2020/7/1 0:38
@desc:
'''
# 导入必要的包
import os
from multiprocessing import cpu_count

import numpy as np
import paddle
import paddle.fluid as fluid


# 创建数据集和数据字典


def create_data_list(data_root_path):
	with open(data_root_path + 'test_list.txt', 'w') as f:
		pass
	with open(data_root_path + 'train_list.txt', 'w') as f:
		pass

	with open(os.path.join(data_root_path, 'dict_txt.txt'), 'r', encoding='utf-8') as f_data:
		dict_txt = eval(f_data.readlines()[0])

	with open(os.path.join(data_root_path, 'news_classify_data.txt'), 'r', encoding='utf-8') as f_data:
		lines = f_data.readlines()
	i = 0
	for line in lines:
		title = line.split('_!_')[-1].replace('\n', '')
		l = line.split('_!_')[1]
		labs = ""
		if i % 10 == 0:
			with open(os.path.join(data_root_path, 'test_list.txt'), 'a', encoding='utf-8') as f_test:
				for s in title:
					lab = str(dict_txt[s])
					labs = labs + lab + ','
				labs = labs[:-1]
				labs = labs + '\t' + l + '\n'
				f_test.write(labs)
		else:
			with open(os.path.join(data_root_path, 'train_list.txt'), 'a', encoding='utf-8') as f_train:
				for s in title:
					lab = str(dict_txt[s])
					labs = labs + lab + ','
				labs = labs[:-1]
				labs = labs + '\t' + l + '\n'
				f_train.write(labs)
		i += 1
	print("数据列表生成完成！")


# 把下载得数据生成一个字典
def create_dict(data_path, dict_path):
	dict_set = set()
	# 读取已经下载得数据
	with open(data_path, 'r', encoding='utf-8') as f:
		lines = f.readlines()
	# 把数据生成一个元组
	for line in lines:
		title = line.split('_!_')[-1].replace('\n', '')
		for s in title:
			dict_set.add(s)
	# 把元组转换成字典，一个字对应一个数字
	dict_list = []
	i = 0
	for s in dict_set:
		dict_list.append([s, i])
		i += 1
	# 添加未知字符
	dict_txt = dict(dict_list)
	end_dict = {"<unk>": i}
	dict_txt.update(end_dict)
	# 把这些字典保存到本地中
	with open(dict_path, 'w', encoding='utf-8') as f:
		f.write(str(dict_txt))

	print("数据字典生成完成！")


# 获取字典的长度
def get_dict_len(dict_path):
	with open(dict_path, 'r', encoding='utf-8') as f:
		line = eval(f.readlines()[0])

	return len(line.keys())


# 创建数据读取器train_reader 和test_reader
# 训练/测试数据的预处理
def data_mapper(sample):
	data, label = sample
	data = [int(data) for data in data.split(',')]
	return data, int(label)


# 创建数据读取器train_reader
def train_reader(train_list_path):
	def reader():
		with open(train_list_path, 'r') as f:
			lines = f.readlines()
			# 打乱数据
			np.random.shuffle(lines)
			# 开始获取每张图像和标签
			for line in lines:
				data, label = line.split('\t')
				yield data, label

	return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


#  创建数据读取器test_reader
def test_reader(test_list_path):
	def reader():
		with open(test_list_path, 'r') as f:
			lines = f.readlines()
			for line in lines:
				data, label = line.split('\t')
				yield data, label

	return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 1024)


# 创建CNN网络

def CNN_net(data, dict_dim, class_dim=10, emb_dim=128, hid_dim=128, hid_dim2=98):
	emb = fluid.layers.embedding(input=data,
	                             size=[dict_dim, emb_dim])
	conv_3 = fluid.nets.sequence_conv_pool(
		input=emb,
		num_filters=hid_dim,
		filter_size=3,
		act="tanh",
		pool_type="sqrt")
	conv_4 = fluid.nets.sequence_conv_pool(
		input=emb,
		num_filters=hid_dim2,
		filter_size=4,
		act="tanh",
		pool_type="sqrt")

	output = fluid.layers.fc(
		input=[conv_3, conv_4], size=class_dim, act='softmax')
	return output


if __name__ == '__main__':
	data_root_path = 'data/'
	# 把生产的数据列表都放在自己的总类别文件夹中
	data_root_path = "data/"
	data_path = os.path.join(data_root_path, 'news_classify_data.txt')
	dict_path = os.path.join(data_root_path, "dict_txt.txt")
	# 创建数据字典
	create_dict(data_path, dict_path)
	# 创建数据列表
	create_data_list(data_root_path)

	# 定义输入数据， lod_level不为0指定输入数据为序列数据
	words = fluid.layers.data(name='words', shape=[1], dtype='int64', lod_level=1)
	label = fluid.layers.data(name='label', shape=[1], dtype='int64')
	# 获取数据字典长度
	dict_dim = get_dict_len('data/dict_txt.txt')
	# 获取卷积神经网络
	# model = CNN_net(words, dict_dim, 15)
	# 获取分类器
	model = CNN_net(words, dict_dim)
	# 获取损失函数和准确率
	cost = fluid.layers.cross_entropy(input=model, label=label)
	avg_cost = fluid.layers.mean(cost)
	acc = fluid.layers.accuracy(input=model, label=label)

	# 获取预测程序
	test_program = fluid.default_main_program().clone(for_test=True)

	# 定义优化方法
	optimizer = fluid.optimizer.AdagradOptimizer(learning_rate=0.002)
	opt = optimizer.minimize(avg_cost)

	# 创建一个执行器，CPU训练速度比较慢
	# place = fluid.CPUPlace()
	place = fluid.CUDAPlace(0)
	exe = fluid.Executor(place)
	# 进行参数初始化
	exe.run(fluid.default_startup_program())

	# 获取训练数据读取器和测试数据读取器
	train_reader = paddle.batch(reader=train_reader('data/train_list.txt'), batch_size=128)
	test_reader = paddle.batch(reader=test_reader('data/test_list.txt'), batch_size=128)

	# 定义数据映射器
	feeder = fluid.DataFeeder(place=place, feed_list=[words, label])

	EPOCH_NUM = 1
	model_save_dir = 'work/infer_model/'
	# 开始训练

	for pass_id in range(EPOCH_NUM):
		# 进行训练
		for batch_id, data in enumerate(train_reader()):
			train_cost, train_acc = exe.run(program=fluid.default_main_program(),
			                                feed=feeder.feed(data),
			                                fetch_list=[avg_cost, acc])

			if batch_id % 100 == 0:
				print('Pass:%d, Batch:%d, Cost:%0.5f, Acc:%0.5f' % (pass_id, batch_id, train_cost[0], train_acc[0]))
		# 进行测试
		test_costs = []
		test_accs = []
		for batch_id, data in enumerate(test_reader()):
			test_cost, test_acc = exe.run(program=test_program,
			                              feed=feeder.feed(data),
			                              fetch_list=[avg_cost, acc])
			test_costs.append(test_cost[0])
			test_accs.append(test_acc[0])
		# 计算平均预测损失在和准确率
		test_cost = (sum(test_costs) / len(test_costs))
		test_acc = (sum(test_accs) / len(test_accs))
		print('Test:%d, Cost:%0.5f, ACC:%0.5f' % (pass_id, test_cost, test_acc))

	# 保存预测模型
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	fluid.io.save_inference_model(model_save_dir,
	                              feeded_var_names=[words.name],
	                              target_vars=[model],
	                              executor=exe)
	print('训练模型保存完成！')

	# 用训练好的模型进行预测并输出预测结果
	# 创建执行器
	place = fluid.CPUPlace()
	exe = fluid.Executor(place)
	exe.run(fluid.default_startup_program())

	save_path = 'work/infer_model/'

	# 从模型中获取预测程序、输入数据名称列表、分类器
	[infer_program, feeded_var_names, target_var] = fluid.io.load_inference_model(dirname=save_path, executor=exe)


	# 获取数据
	def get_data(sentence):
		# 读取数据字典
		with open('data/dict_txt.txt', 'r', encoding='utf-8') as f_data:
			dict_txt = eval(f_data.readlines()[0])
		dict_txt = dict(dict_txt)
		# 把字符串数据转换成列表数据
		keys = dict_txt.keys()
		data = []
		for s in sentence:
			# 判断是否存在未知字符
			if not s in keys:
				s = '<unk>'
			data.append(int(dict_txt[s]))
		return np.array(data, dtype=np.int64)


	data = []
	# 获取图片数据
	data1 = get_data('在获得诺贝尔文学奖7年之后，莫言15日晚间在山西汾阳贾家庄如是说')
	data2 = get_data('综合“今日美国”、《世界日报》等当地媒体报道，芝加哥河滨警察局表示，')
	data.append(data1)
	data.append(data2)

	# 获取每句话的单词数量
	base_shape = [[len(c) for c in data]]

	# 生成预测数据
	tensor_words = fluid.create_lod_tensor(data, base_shape, place)

	# 执行预测
	result = exe.run(program=infer_program,
	                 feed={feeded_var_names[0]: tensor_words},
	                 fetch_list=target_var)

	# 分类名称
	names = ['文化', '娱乐', '体育', '财经', '房产', '汽车', '教育', '科技', '国际', '证券']

	# 获取结果概率最大的label
	for i in range(len(data)):
		lab = np.argsort(result)[0][i][-1]
		print('预测结果标签为：%d， 名称为：%s， 概率为：%f' % (lab, names[lab], result[0][i][lab]))
