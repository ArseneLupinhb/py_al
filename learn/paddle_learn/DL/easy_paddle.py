# 加载飞桨和相关类库
import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.fluid as fluid
from PIL import Image
# from paddle.fluid.dygraph.nn import FC
from paddle.fluid.dygraph import Linear


# 定义mnist数据识别网络结构，同房价预测网络
class MNIST(fluid.dygraph.Layer):
	def __init__(self, name_scope):
		super(MNIST, self).__init__(name_scope)
		name_scope = self.full_name()
		# 定义一层全连接层，输出维度是1，激活函数为None，即不使用激活函数
		# self.fc = FC(name_scope, size=1, act=None)
		self.fc = Linear(28 * 28, output_dim=1, act=None)

	# 定义网络结构的前向计算过程
	def forward(self, inputs):
		outputs = self.fc(inputs)
		return outputs


def train_model():
	with fluid.dygraph.guard():
		model = MNIST("mnist")
		model.train()
		train_loader = paddle.batch(paddle.dataset.mnist.train(), batch_size=16)
		optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001,
		                                         parameter_list=model.parameters())  # 优化器选用SGD随机梯度下降，学习率为0.001.
		EPOCH_NUM = 10
		start = time.perf_counter()
		for epoch_id in range(EPOCH_NUM):
			for batch_id, data in enumerate(train_loader()):
				# 准备数据，格式需要转换成符合框架要求的
				image_data = np.array([x[0] for x in data]).astype('float32')
				label_data = np.array([x[1] for x in data]).astype('float32').reshape(-1, 1)
				# 将数据转为飞桨动态图格式
				image = fluid.dygraph.to_variable(image_data)
				label = fluid.dygraph.to_variable(label_data)

				# 前向计算的过程
				predict = model(image)

				# 计算损失，取一个批次样本损失的平均值
				loss = fluid.layers.square_error_cost(predict, label)
				avg_loss = fluid.layers.mean(loss)

				# 每训练了1000批次的数据，打印下当前Loss的情况
				if batch_id != 0 and batch_id % 1000 == 0:
					end = time.perf_counter()
					print("time is {} s".format(end - start))
					start = end
					print("epoch: {}, batch: {}, loss is: {}".format(epoch_id, batch_id, avg_loss.numpy()))

				# 后向传播，更新参数的过程
				avg_loss.backward()
				optimizer.minimize(avg_loss)
				model.clear_gradients()

		# 保存模型
		fluid.save_dygraph(model.state_dict(), 'mnist')


# 读取一张本地的样例图片，转变成模型输入的格式
def load_image(img_path):
	# 从img_path中读取图像，并转为灰度图
	im = Image.open(img_path).convert('L')
	print(np.array(im))
	im = im.resize((28, 28), Image.ANTIALIAS)
	im = np.array(im).reshape(1, -1).astype(np.float32)
	# 图像归一化，保持和数据集的数据范围一致
	im = 1 - im / 127.5
	return im


def eval_model():
	# 定义预测过程
	with fluid.dygraph.guard():
		model = MNIST("mnist")
		params_file_path = 'mnist'
		img_path = 'work/example_0.png'
		# 加载模型参数 只需要模型的参数在， 就可以调用运行了。
		model_dict, _ = fluid.load_dygraph("mnist")
		model.load_dict(model_dict)
		# 灌入数据
		model.eval()
		tensor_img = load_image(img_path)
		result = model(fluid.dygraph.to_variable(tensor_img))
		#  预测输出取整，即为预测的数字，打印结果
		print("本次预测的数字是", result.numpy().astype('int32'))


def show_test_img():
	example = mpimg.imread('work/example_0.png')
	# 显示图像
	plt.imshow(example)
	plt.show()


def test_model():
	pass


# 通过with语句创建一个dygraph运行的context，
# 动态图下的一些操作需要在guard下进行
# 还是面向对象好
if __name__ == '__main__':
	# 训练模型
	# train_model()
	# 导入图像读取第三方库
	# 读取图像
	show_test_img()

	# 只需要有pdparams 这个文件在就可以调用运行了
	# 评估模型，并用于参数调优
	eval_model()

	# 测试模型
	test_model()
