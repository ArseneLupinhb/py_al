import os
import random

import numpy as np
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from PIL import Image
from paddle.fluid.dygraph import Linear, Embedding, Conv2D

source_path = os.getcwd() + r'/learn/paddle_learn/recommended_system/'


class MovieLen(object):
	def __init__(self, use_poster):
		self.use_poster = use_poster
		# 声明每个数据文件的路径
		usr_info_path = "work/ml-1m/users.dat"
		if use_poster:
			rating_path = "work/ml-1m/new_rating.txt"
		else:
			rating_path = "work/ml-1m/ratings.dat"

		movie_info_path = "work/ml-1m/movies.dat"
		self.poster_path = "work/ml-1m/posters/"
		# 得到电影数据
		self.movie_info, self.movie_cat, self.movie_title = self.get_movie_info(movie_info_path)
		# 记录电影的最大ID
		self.max_mov_cat = np.max([self.movie_cat[k] for k in self.movie_cat])
		self.max_mov_tit = np.max([self.movie_title[k] for k in self.movie_title])
		self.max_mov_id = np.max(list(map(int, self.movie_info.keys())))
		# 记录用户数据的最大ID
		self.max_usr_id = 0
		self.max_usr_age = 0
		self.max_usr_job = 0
		# 得到用户数据
		self.usr_info = self.get_usr_info(usr_info_path)
		# 得到评分数据
		self.rating_info = self.get_rating_info(rating_path)
		# 构建数据集
		self.dataset = self.get_dataset(usr_info=self.usr_info,
		                                rating_info=self.rating_info,
		                                movie_info=self.movie_info)
		# 划分数据集，获得数据加载器
		self.train_dataset = self.dataset[:int(len(self.dataset) * 0.9)]
		self.valid_dataset = self.dataset[int(len(self.dataset) * 0.9):]
		print("##Total dataset instances: ", len(self.dataset))
		print("##MovieLens dataset information: \nusr num: {}\n"
		      "movies num: {}".format(len(self.usr_info), len(self.movie_info)))

	# 得到电影数据
	def get_movie_info(self, path):
		# 打开文件，编码方式选择ISO-8859-1，读取所有数据到data中
		with open(path, 'r', encoding="ISO-8859-1") as f:
			data = f.readlines()
		# 建立三个字典，分别用户存放电影所有信息，电影的名字信息、类别信息
		movie_info, movie_titles, movie_cat = {}, {}, {}
		# 对电影名字、类别中不同的单词计数
		t_count, c_count = 1, 1

		count_tit = {}
		# 按行读取数据并处理
		for item in data:
			item = item.strip().split("::")
			v_id = item[0]
			v_title = item[1][:-7]
			cats = item[2].split('|')
			v_year = item[1][-5:-1]

			titles = v_title.split()
			# 统计电影名字的单词，并给每个单词一个序号，放在movie_titles中
			for t in titles:
				if t not in movie_titles:
					movie_titles[t] = t_count
					t_count += 1
			# 统计电影类别单词，并给每个单词一个序号，放在movie_cat中
			for cat in cats:
				if cat not in movie_cat:
					movie_cat[cat] = c_count
					c_count += 1
			# 补0使电影名称对应的列表长度为15
			v_tit = [movie_titles[k] for k in titles]
			while len(v_tit) < 15:
				v_tit.append(0)
			# 补0使电影种类对应的列表长度为6
			v_cat = [movie_cat[k] for k in cats]
			while len(v_cat) < 6:
				v_cat.append(0)
			# 保存电影数据到movie_info中
			movie_info[v_id] = {'mov_id': int(v_id),
			                    'title': v_tit,
			                    'category': v_cat,
			                    'years': int(v_year)}
		return movie_info, movie_cat, movie_titles

	def get_usr_info(self, path):
		# 性别转换函数，M-0， F-1
		def gender2num(gender):
			return 1 if gender == 'F' else 0

		# 打开文件，读取所有行到data中
		with open(path, 'r') as f:
			data = f.readlines()
		# 建立用户信息的字典
		use_info = {}

		max_usr_id = 0
		# 按行索引数据
		for item in data:
			# 去除每一行中和数据无关的部分
			item = item.strip().split("::")
			usr_id = item[0]
			# 将字符数据转成数字并保存在字典中
			use_info[usr_id] = {'usr_id': int(usr_id),
			                    'gender': gender2num(item[1]),
			                    'age': int(item[2]),
			                    'job': int(item[3])}
			self.max_usr_id = max(self.max_usr_id, int(usr_id))
			self.max_usr_age = max(self.max_usr_age, int(item[2]))
			self.max_usr_job = max(self.max_usr_job, int(item[3]))
		return use_info

	# 得到评分数据
	def get_rating_info(self, path):
		# 读取文件里的数据
		with open(path, 'r') as f:
			data = f.readlines()
		# 将数据保存在字典中并返回
		rating_info = {}
		for item in data:
			item = item.strip().split("::")
			usr_id, movie_id, score = item[0], item[1], item[2]
			if usr_id not in rating_info.keys():
				rating_info[usr_id] = {movie_id: float(score)}
			else:
				rating_info[usr_id][movie_id] = float(score)
		return rating_info

	# 构建数据集
	def get_dataset(self, usr_info, rating_info, movie_info):
		trainset = []
		for usr_id in rating_info.keys():
			usr_ratings = rating_info[usr_id]
			for movie_id in usr_ratings:
				trainset.append({'usr_info': usr_info[usr_id],
				                 'mov_info': movie_info[movie_id],
				                 'scores': usr_ratings[movie_id]})
		return trainset

	def load_data(self, dataset=None, mode='train'):
		use_poster = False

		# 定义数据迭代Batch大小
		BATCHSIZE = 256

		data_length = len(dataset)
		index_list = list(range(data_length))

		# 定义数据迭代加载器
		def data_generator():
			# 训练模式下，打乱训练数据
			if mode == 'train':
				random.shuffle(index_list)
			# 声明每个特征的列表
			usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
			mov_id_list, mov_tit_list, mov_cat_list, mov_poster_list = [], [], [], []
			score_list = []
			# 索引遍历输入数据集
			for idx, i in enumerate(index_list):
				# 获得特征数据保存到对应特征列表中
				usr_id_list.append(dataset[i]['usr_info']['usr_id'])
				usr_gender_list.append(dataset[i]['usr_info']['gender'])
				usr_age_list.append(dataset[i]['usr_info']['age'])
				usr_job_list.append(dataset[i]['usr_info']['job'])

				mov_id_list.append(dataset[i]['mov_info']['mov_id'])
				mov_tit_list.append(dataset[i]['mov_info']['title'])
				mov_cat_list.append(dataset[i]['mov_info']['category'])
				mov_id = dataset[i]['mov_info']['mov_id']

				if use_poster:
					# 不使用图像特征时，不读取图像数据，加快数据读取速度
					poster = Image.open(self.poster_path + 'mov_id{}.jpg'.format(str(mov_id[0])))
					poster = poster.resize([64, 64])
					if len(poster.size) <= 2:
						poster = poster.convert("RGB")

					mov_poster_list.append(np.array(poster))

				score_list.append(int(dataset[i]['scores']))
				# 如果读取的数据量达到当前的batch大小，就返回当前批次
				if len(usr_id_list) == BATCHSIZE:
					# 转换列表数据为数组形式，reshape到固定形状
					usr_id_arr = np.array(usr_id_list).astype(np.int64)
					usr_gender_arr = np.array(usr_gender_list).astype(np.int64)
					usr_age_arr = np.array(usr_age_list).astype(np.int64)
					usr_job_arr = np.array(usr_job_list).astype(np.int64)

					mov_id_arr = np.array(mov_id_list).astype(np.int64).astype(np.int64)
					mov_cat_arr = np.reshape(np.array(mov_cat_list), [BATCHSIZE, 6]).astype(np.int64)
					mov_tit_arr = np.reshape(np.array(mov_tit_list), [BATCHSIZE, 1, 15]).astype(np.int64)

					if use_poster:
						mov_poster_arr = np.reshape(np.array(mov_poster_list) / 127.5 - 1,
						                            [BATCHSIZE, 3, 64, 64]).astype(np.float32)
					else:
						mov_poster_arr = np.array([0.])

					scores_arr = np.reshape(np.array(score_list), [-1, 1]).astype(np.float32)

					# 放回当前批次数据
					yield [usr_id_arr, usr_gender_arr, usr_age_arr, usr_job_arr], \
					      [mov_id_arr, mov_cat_arr, mov_tit_arr, mov_poster_arr], scores_arr

					# 清空数据
					usr_id_list, usr_gender_list, usr_age_list, usr_job_list = [], [], [], []
					mov_id_list, mov_tit_list, mov_cat_list, score_list = [], [], [], []
					mov_poster_list = []

		return data_generator


class Model(dygraph.layers.Layer):
	def __init__(self, name_scope, use_poster, use_mov_title, use_mov_cat, use_age_job):
		super(Model, self).__init__(name_scope)
		name = self.full_name()

		# 将传入的name信息和bool型参数添加到模型类中
		self.use_mov_poster = use_poster
		self.use_mov_title = use_mov_title
		self.use_usr_age_job = use_age_job
		self.use_mov_cat = use_mov_cat

		# 获取数据集的信息，并构建训练和验证集的数据迭代器
		Dataset = MovieLen(self.use_mov_poster)
		self.Dataset = Dataset
		self.trainset = self.Dataset.train_dataset
		self.valset = self.Dataset.valid_dataset
		self.train_loader = self.Dataset.load_data(dataset=self.trainset, mode='train')
		self.valid_loader = self.Dataset.load_data(dataset=self.valset, mode='valid')

		""" define network layer for embedding usr info """
		USR_ID_NUM = Dataset.max_usr_id + 1
		# 对用户ID做映射，并紧接着一个FC层
		self.usr_emb = Embedding([USR_ID_NUM, 32], is_sparse=False)
		self.usr_fc = Linear(32, 32)

		# 对用户性别信息做映射，并紧接着一个FC层
		USR_GENDER_DICT_SIZE = 2
		self.usr_gender_emb = Embedding([USR_GENDER_DICT_SIZE, 16])
		self.usr_gender_fc = Linear(16, 16)

		# 对用户年龄信息做映射，并紧接着一个FC层
		USR_AGE_DICT_SIZE = Dataset.max_usr_age + 1
		self.usr_age_emb = Embedding([USR_AGE_DICT_SIZE, 16])
		self.usr_age_fc = Linear(16, 16)

		# 对用户职业信息做映射，并紧接着一个FC层
		USR_JOB_DICT_SIZE = Dataset.max_usr_job + 1
		self.usr_job_emb = Embedding([USR_JOB_DICT_SIZE, 16])
		self.usr_job_fc = Linear(16, 16)

		# 新建一个FC层，用于整合用户数据信息
		self.usr_combined = Linear(80, 200, act='tanh')

		""" define network layer for embedding usr info """
		# 对电影ID信息做映射，并紧接着一个FC层
		MOV_DICT_SIZE = Dataset.max_mov_id + 1
		self.mov_emb = Embedding([MOV_DICT_SIZE, 32])
		self.mov_fc = Linear(32, 32)

		# 对电影类别做映射
		CATEGORY_DICT_SIZE = len(Dataset.movie_cat) + 1
		self.mov_cat_emb = Embedding([CATEGORY_DICT_SIZE, 32], is_sparse=False)
		self.mov_cat_fc = Linear(32, 32)

		# 对电影名称做映射
		MOV_TITLE_DICT_SIZE = len(Dataset.movie_title) + 1
		self.mov_title_emb = Embedding([MOV_TITLE_DICT_SIZE, 32], is_sparse=False)
		self.mov_title_conv = Conv2D(1, 1, filter_size=(3, 1), stride=(2, 1), padding=0, act='relu')
		self.mov_title_conv2 = Conv2D(1, 1, filter_size=(3, 1), stride=1, padding=0, act='relu')

		# 新建一个FC层，用于整合电影特征
		self.mov_concat_embed = Linear(96, 200, act='tanh')

	# 定义计算用户特征的前向运算过程
	def get_usr_feat(self, usr_var):
		""" get usr features"""
		# 获取到用户数据
		usr_id, usr_gender, usr_age, usr_job = usr_var
		# 将用户的ID数据经过embedding和FC计算，得到的特征保存在feats_collect中
		feats_collect = []
		usr_id = self.usr_emb(usr_id)
		usr_id = self.usr_fc(usr_id)
		usr_id = fluid.layers.relu(usr_id)
		feats_collect.append(usr_id)

		# 计算用户的性别特征，并保存在feats_collect中
		usr_gender = self.usr_gender_emb(usr_gender)
		usr_gender = self.usr_gender_fc(usr_gender)
		usr_gender = fluid.layers.relu(usr_gender)
		feats_collect.append(usr_gender)
		# 选择是否使用用户的年龄-职业特征
		if self.use_usr_age_job:
			# 计算用户的年龄特征，并保存在feats_collect中
			usr_age = self.usr_age_emb(usr_age)
			usr_age = self.usr_age_fc(usr_age)
			usr_age = fluid.layers.relu(usr_age)
			feats_collect.append(usr_age)
			# 计算用户的职业特征，并保存在feats_collect中
			usr_job = self.usr_job_emb(usr_job)
			usr_job = self.usr_job_fc(usr_job)
			usr_job = fluid.layers.relu(usr_job)
			feats_collect.append(usr_job)

		# 将用户的特征级联，并通过FC层得到最终的用户特征
		usr_feat = fluid.layers.concat(feats_collect, axis=1)
		usr_feat = self.usr_combined(usr_feat)
		return usr_feat

	# 定义电影特征的前向计算过程
	def get_mov_feat(self, mov_var):
		""" get movie features"""
		# 获得电影数据
		mov_id, mov_cat, mov_title, mov_poster = mov_var
		feats_collect = []
		# 获得batchsize的大小
		batch_size = mov_id.shape[0]
		# 计算电影ID的特征，并存在feats_collect中
		mov_id = self.mov_emb(mov_id)
		mov_id = self.mov_fc(mov_id)
		mov_id = fluid.layers.relu(mov_id)
		feats_collect.append(mov_id)

		# 如果使用电影的种类数据，计算电影种类特征的映射
		if self.use_mov_cat:
			# 计算电影种类的特征映射，对多个种类的特征求和得到最终特征
			mov_cat = self.mov_cat_emb(mov_cat)
			mov_cat = fluid.layers.reduce_sum(mov_cat, dim=1, keep_dim=False)

			mov_cat = self.mov_cat_fc(mov_cat)
			feats_collect.append(mov_cat)

		if self.use_mov_title:
			# 计算电影名字的特征映射，对特征映射使用卷积计算最终的特征
			mov_title = self.mov_title_emb(mov_title)
			mov_title = self.mov_title_conv2(self.mov_title_conv(mov_title))
			mov_title = fluid.layers.reduce_sum(mov_title, dim=2, keep_dim=False)
			mov_title = fluid.layers.relu(mov_title)
			mov_title = fluid.layers.reshape(mov_title, [batch_size, -1])

			feats_collect.append(mov_title)

		# 使用一个全连接层，整合所有电影特征，映射为一个200维的特征向量
		mov_feat = fluid.layers.concat(feats_collect, axis=1)
		mov_feat = self.mov_concat_embed(mov_feat)
		return mov_feat

	# 定义个性化推荐算法的前向计算
	def forward(self, usr_var, mov_var):
		# 计算用户特征和电影特征
		usr_feat = self.get_usr_feat(usr_var)
		mov_feat = self.get_mov_feat(mov_var)
		# 根据计算的特征计算相似度
		res = fluid.layers.cos_sim(usr_feat, mov_feat)
		# 将相似度扩大范围到和电影评分相同数据范围
		res = fluid.layers.scale(res, scale=5)
		return usr_feat, mov_feat, res


def train(model):
	# 配置训练参数
	use_gpu = False
	lr = 0.01
	Epoches = 10

	place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
	with fluid.dygraph.guard(place):
		# 启动训练
		model.train()
		# 获得数据读取器
		data_loader = model.train_loader
		# 使用adam优化器，学习率使用0.01
		opt = fluid.optimizer.Adam(learning_rate=lr, parameter_list=model.parameters())

		for epoch in range(0, Epoches):
			for idx, data in enumerate(data_loader()):
				# 获得数据，并转为动态图格式
				usr, mov, score = data
				usr[0] = usr[0].astype(np.int64)
				mov[0] = mov[0].astype(np.int64)

				usr_v = [dygraph.to_variable(var) for var in usr]
				mov_v = [dygraph.to_variable(var) for var in mov]
				scores_label = dygraph.to_variable(score)

				# 计算出算法的前向计算结果
				_, _, scores_predict = model.forward(usr_v, mov_v)
				# 计算loss
				loss = fluid.layers.square_error_cost(scores_predict, scores_label)
				avg_loss = fluid.layers.mean(loss)
				if idx % 500 == 0:
					print("epoch: {}, batch_id: {}, loss is: {}".format(epoch, idx, avg_loss.numpy()))

				# 损失函数下降，并清除梯度
				avg_loss.backward()
				opt.minimize(avg_loss)
				model.clear_gradients()
			# 每个epoch 保存一次模型
			fluid.save_dygraph(model.state_dict(), source_path + '/checkpoint/epoch' + str(epoch))


# 启动训练
if __name__ == '__main__':
	with dygraph.guard():
		use_poster, use_mov_title, use_mov_cat, use_age_job = False, True, True, True
		model = Model('Recommend', use_poster, use_mov_title, use_mov_cat, use_age_job)
		train(model)
