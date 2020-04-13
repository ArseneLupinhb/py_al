# 数据处理部分之前的代码，加入部分数据处理的库
import gzip
import json
import random

import numpy as np
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Linear


def load_data(mode='train'):
	datafile = 'work/mnist.json.gz'
	print('loading mnist dataset from {} ......'.format(datafile))
	# 加载json数据文件
	data = json.load(gzip.open(datafile))
	print('mnist dataset load done')

	# 读取到的数据区分训练集，验证集，测试集
	train_set, val_set, eval_set = data
	if mode == 'train':
		# 获得训练数据集
		imgs, labels = train_set[0], train_set[1]
	elif mode == 'valid':
		# 获得验证数据集
		imgs, labels = val_set[0], val_set[1]
	elif mode == 'eval':
		# 获得测试数据集
		imgs, labels = eval_set[0], eval_set[1]
	else:
		raise Exception("mode can only be one of ['train', 'valid', 'eval']")
	print("训练数据集数量: ", len(imgs))

	# 校验数据
	imgs_length = len(imgs)

	# 校验数据
	assert len(imgs) == len(labels), \
		"length of train_imgs({}) should be the same as train_labels({})".format(len(imgs), len(labels))

	# 获得数据集长度
	imgs_length = len(imgs)

	# 定义数据集每个数据的序号，根据序号读取数据
	index_list = list(range(imgs_length))
	# 读入数据时用到的批次大小
	BATCHSIZE = 100
	IMG_ROWS = 28
	IMG_COLS = 28

	# 定义数据生成器
	def data_generator():
		if mode == 'train':
			# 训练模式下打乱数据
			random.shuffle(index_list)

		imgs_list = []
		labels_list = []
		for i in index_list:
			# 将数据处理成希望的格式，比如类型为float32，shape为[1, 28, 28]
			img = np.reshape(imgs[i], [1, IMG_ROWS, IMG_COLS]).astype('float32')
			label = np.reshape(labels[i], [1]).astype('int64')
			imgs_list.append(img)
			labels_list.append(label)
			if len(imgs_list) == BATCHSIZE:
				# 获得一个batchsize的数据，并返回
				yield np.array(imgs_list), np.array(labels_list)

				# 清空数据读取列表
				imgs_list = []
				labels_list = []

		# 如果剩余数据的数目小于BATCHSIZE，
		# 则剩余数据一起构成一个大小为len(imgs_list)的mini-batch
		if len(imgs_list) > 0:
			yield np.array(imgs_list), np.array(labels_list)

	return data_generator


# 数据处理部分之后的代码，数据读取的部分调用Load_data函数
# 定义网络结构，同上一节所使用的网络结构
# 多层卷积神经网络实现
class MNIST(fluid.dygraph.Layer):
	def __init__(self, name_scope):
		super(MNIST, self).__init__(name_scope)

		# 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
		# 激活函数使用relu
		self.conv1 = Conv2D(num_channels=1, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
		# 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
		self.pool1 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
		# 定义卷积层，输出特征通道num_filters设置为20，卷积核的大小filter_size为5，卷积步长stride=1，padding=2
		self.conv2 = Conv2D(num_channels=20, num_filters=20, filter_size=5, stride=1, padding=2, act='relu')
		# 定义池化层，池化核pool_size=2，池化步长为2，选择最大池化方式
		self.pool2 = Pool2D(pool_size=2, pool_stride=2, pool_type='max')
		# 定义一层全连接层，输出维度是1，不使用激活函数
		self.fc = Linear(input_dim=980, output_dim=10, act="softmax")

	# 定义网络前向计算过程，卷积后紧接着使用池化层，最后使用全连接层计算最终输出
	def forward(self, inputs):
		x = self.conv1(inputs)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = fluid.layers.reshape(x, [x.shape[0], -1])
		x = self.fc(x)
		return x


def trian_model():
	# 训练配置，并启动训练过程
	with fluid.dygraph.guard():
		model = MNIST("mnist")
		model.train()
		# 调用加载数据的函数
		train_loader = load_data('train')
		# 创建异步数据读取器
		place = fluid.CPUPlace()
		data_loader = fluid.io.DataLoader.from_generator(capacity=5, return_list=True)
		data_loader.set_batch_generator(train_loader, places=place)

		optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001, parameter_list=model.parameters())
		EPOCH_NUM = 10
		for epoch_id in range(EPOCH_NUM):
			for batch_id, data in enumerate(data_loader):
				# 准备数据，变得更加简洁
				image_data, label_data = data
				image = fluid.dygraph.to_variable(image_data)
				label = fluid.dygraph.to_variable(label_data)

				# 前向计算的过程
				predict = model(image)

				# 计算损失，取一个批次样本损失的平均值
				# loss = fluid.layers.square_error_cost(predict, label)
				# 损失函数改为交叉熵
				loss = fluid.layers.cross_entropy(predict, label)

				avg_loss = fluid.layers.mean(loss)

				# 每训练了200批次的数据，打印下当前Loss的情况
				if batch_id % 200 == 0:
					print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

				# 后向传播，更新参数的过程
				avg_loss.backward()
				optimizer.minimize(avg_loss)
				model.clear_gradients()

		# 保存模型参数
		fluid.save_dygraph(model.state_dict(), 'mnist')


if __name__ == '__main__':
	# load_data()
	trian_model()
