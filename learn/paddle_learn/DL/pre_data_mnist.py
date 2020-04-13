# 数据处理部分之前的代码，加入部分数据处理的库
import gzip
import json
import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import paddle.fluid as fluid
import pandas as pd
from PIL import Image
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
	print("{}-数据集数量: {}".format(mode, len(imgs)))

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
	def forward(self, inputs, label=None):
		x = self.conv1(inputs)
		x = self.pool1(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = fluid.layers.reshape(x, [x.shape[0], -1])
		x = self.fc(x)
		# 如果label不是None，则计算分类精度并返回
		if label is not None:
			# print(label)
			acc = fluid.layers.accuracy(input=x, label=label)
			return x, acc
		else:
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
		iter = 0
		iters = []
		losses = []
		accs = []
		for epoch_id in range(EPOCH_NUM):
			for batch_id, data in enumerate(data_loader):
				# 准备数据，变得更加简洁
				image_data, label_data = data
				image = fluid.dygraph.to_variable(image_data)
				label = fluid.dygraph.to_variable(label_data)

				# 前向计算的过程
				predict, acc = model(image, label)

				# 计算损失，取一个批次样本损失的平均值
				# loss = fluid.layers.square_error_cost(predict, label)
				# 损失函数改为交叉熵
				loss = fluid.layers.cross_entropy(predict, label)

				avg_loss = fluid.layers.mean(loss)

				# 每训练了200批次的数据，打印下当前Loss的情况
				if batch_id % 200 == 0:
					print("epoch: {}, batch: {}, loss is: {}, acc is {}".format(epoch_id, batch_id, avg_loss.numpy(),
					                                                            acc.numpy()))
					iters.append(iter)
					losses.append(avg_loss.numpy())
					accs.append(acc.numpy())
					iter = iter + 100
				# show_trianning(iters, accs)

				# 后向传播，更新参数的过程
				avg_loss.backward()
				optimizer.minimize(avg_loss)
				model.clear_gradients()

		show_trianning(iters, losses)
		show_trianning(iters, accs)

		# 保存模型参数
		fluid.save_dygraph(model.state_dict(), 'mnist')


def show_trianning(iters, losses):
	# 画出训练过程中Loss的变化曲线
	plt.figure()
	plt.title("trainning", fontsize=24)
	plt.xlabel("iter", fontsize=14)
	plt.ylabel("trainning", fontsize=14)
	plt.plot(iters, losses, color='red', label='train loss')
	plt.grid()
	plt.show()


# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
	# 从img_path中读取图像，并转为灰度图
	im = Image.open(img_path).convert('L')
	# im.show()
	im = im.resize((28, 28), Image.ANTIALIAS)
	im = np.array(im).reshape(1, 1, 28, 28).astype(np.float32)
	# 图像归一化
	im = 1.0 - im / 255.
	return im


def eval_model():
	# 定义预测过程
	with fluid.dygraph.guard():
		model = MNIST("mnist")
		params_file_path = 'mnist'
		img_path = r'work/example_0.png'
		# 加载模型参数
		model_dict, _ = fluid.load_dygraph("mnist")
		model.load_dict(model_dict)

		model.eval()
		tensor_img = load_image(img_path)
		# 模型反馈10个分类标签的对应概率
		results = model(fluid.dygraph.to_variable(tensor_img))
		# 取概率最大的标签作为预测输出
		lab = np.argsort(results.numpy())
		print(lab)
		print("本次预测的数字是: ", lab[0][-1])


def show_test_img():
	example = mpimg.imread('work/example_0.png')
	# 显示图像
	plt.imshow(example)
	plt.show()


def test_model():
	with fluid.dygraph.guard():
		print('start evaluation .......')
		# 加载模型参数
		model = MNIST("mnist")
		model_state_dict, _ = fluid.load_dygraph('mnist')
		model.load_dict(model_state_dict)

		model.eval()
		eval_loader = load_data('eval')

		acc_set = []
		avg_loss_set = []
		for batch_id, data in enumerate(eval_loader()):
			x_data, y_data = data
			img = fluid.dygraph.to_variable(x_data)
			label = fluid.dygraph.to_variable(y_data)
			prediction, acc = model(img, label)
			loss = fluid.layers.cross_entropy(input=prediction, label=label)
			avg_loss = fluid.layers.mean(loss)
			acc_set.append(float(acc.numpy()))
			avg_loss_set.append(float(avg_loss.numpy()))

		# 计算多个batch的平均损失和准确率
		acc_val_mean = np.array(acc_set).mean()
		avg_loss_val_mean = np.array(avg_loss_set).mean()

		print('loss={}, acc={}'.format(avg_loss_val_mean, acc_val_mean))
		record_result(avg_loss_val_mean, acc_val_mean)


def record_result(avg_loss_val_mean, acc_val_mean):
	result_path = r'results.csv'
	if not os.path.exists(result_path):
		result = pd.DataFrame(columns=('loss', 'acc'))
		row = {'loss': avg_loss_val_mean, 'acc': acc_val_mean}
		result = result.append(row, ignore_index=True)
		result.to_csv('results.csv')
	else:
		result = pd.read_csv(result_path)
		row = {'loss': avg_loss_val_mean, 'acc': acc_val_mean}
		result = result.append(row, ignore_index=True)
		result.to_csv('results.csv')
	print("record done")


if __name__ == '__main__':
	# load_data()
	# trian_model()
	# show_test_img()
	# eval_model()
	# test_model()
	for i in range(10):
		trian_model()
		# eval_model()
		test_model()
