## 腾讯视频弹幕的爬虫思路
# 1. 先获取一页，封装成函数
# 2. 找到timestamp规律构建URL循环翻页，获取一集所有的页数，封装成函数
# 3. 找到target_id和vid的规律，获取12集的弹幕

import time

import pandas as pd
# 导入所需库
import requests


def get_danmu_one_page(url_dm):
	"""
	:param url_dm: 视频弹幕URL地址
	:return: 一页的弹幕数据
	"""
	# 添加headers
	headers = {
		'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.163 Safari/537.36',
		'cookie': 'pgv_pvi=3763519488; RK=FNgMjvw30+; ptcz=596865a55db9032f40c0771caad844e54a3d57361c2eb99c1358fca69b4cbb59; tvfe_boss_uuid=8148a90d1c7db998; video_guid=d44bf48103b94fd9; video_platform=2; pgv_pvid=7168286806; o_cookie=2315561922; pac_uid=1_2315561922; _video_qq_login_time_init=1586937438; m_pvid=1581718132; pgv_info=ssid=s6250434168; pgv_si=s3805727744; _qpsvr_localtk=0.6045838508524248; ptui_loginuin=751068118; main_login=qq; vqq_access_token=80E4CDAED4CB878C9B5510E9BFEB656C; vqq_appid=101483052; vqq_openid=C9E60E4EE691057C69DD81ED500B1B0C; vqq_vuserid=1427072167; vqq_vusession=WGEG8zpPKfJkjjR8nSyz4A..; vqq_refresh_token=46C0E6F2AD094F344AA1B74CF029919F; login_time_init=2020-4-16 15:17:6; uid=825372102; vqq_next_refresh_time=6598; vqq_login_time_init=1587021428; login_time_last=2020-4-16 15:17:8',
		'referer': 'https://v.qq.com/x/cover/mzc00200bll9mha.html',
	}

	# 发起请求
	try:
		r = requests.get(url_dm, headers=headers, timeout=3)
	except Exception as e:
		time.sleep(3)
		r = requests.get(url_dm, headers=headers, timeout=3)

	# 解析网页
	data = r.json()['comments']

	# 获取评论ID
	comment_id = [i['commentid'] for i in data]
	# 获取用户名
	oper_name = [i['opername'] for i in data]
	# 获取会员等级
	vip_degree = [i['uservip_degree'] for i in data]
	# 获取评论内容
	content = [i['content'] for i in data]
	# 获取评论时间点
	time_point = [i['timepoint'] for i in data]
	# 获取评论点赞
	up_count = [i['upcount'] for i in data]

	# 存储数据
	df_one = pd.DataFrame({
		'comment_id': comment_id,
		'oper_name': oper_name,
		'vip_degree': vip_degree,
		'content': content,
		'time_point': time_point,
		'up_count': up_count
	})
	return df_one


def get_danmu_all_page(target_id, vid):
	"""
	:param target_id: target_id
	:param vid: vid
	:return: 所有页弹幕
	"""
	df_all = pd.DataFrame()
	# 记录步数
	step = 1
	for time_stamp in range(15, 100000, 30):  # 右侧设置一个足够大的数
		try:  # 异常处理
			# 构建URL
			url_dm = 'https://mfm.video.qq.com/danmu?target_id={}&vid={}&timestamp={}'.format(target_id, vid,
			                                                                                  time_stamp)
			# 调用函数
			df = get_danmu_one_page(url_dm)
			# 终止条件
			if df.shape[0] == 0:
				break
			else:
				df_all = df_all.append(df, ignore_index=True)
				# 打印进度
				print('我正在获取第{}页的信息'.format(step))
				step += 1
				# 休眠一秒
				time.sleep(1)
		except Exception as e:
			continue

	return df_all


# 获取target_id和vid，此处代码省略
if __name__ == '__main__':
	# 第一集
	df_1 = get_danmu_all_page(target_id='5035751720', vid='j00336vecpt')
	df_1.insert(0, 'episodes', 1)

	# 第二集
	df_2 = get_danmu_all_page(target_id='5035751775', vid='y0033grdnk8')
	df_2.insert(0, 'episodes', 2)

	# 第三集
	df_3 = get_danmu_all_page(target_id='5035751777', vid='x00336xs3k8')
	df_3.insert(0, 'episodes', 3)

	# 第四集
	df_4 = get_danmu_all_page(target_id='5035753463', vid='h00339qal0k')
	df_4.insert(0, 'episodes', 4)

	# 第五集
	df_5 = get_danmu_all_page(target_id='5041112031', vid='v0033blmf6h')
	df_5.insert(0, 'episodes', 5)

	# 第六集
	df_6 = get_danmu_all_page(target_id='5041112030', vid='l0033rcr6id')
	df_6.insert(0, 'episodes', 6)

	# 第七集
	df_7 = get_danmu_all_page(target_id='5073281708', vid='a0033kdhuw6')
	df_7.insert(0, 'episodes', 7)

	# 第八集
	df_8 = get_danmu_all_page(target_id='5073281752', vid='x0033cutafk')
	df_8.insert(0, 'episodes', 8)

	# 第九集
	df_9 = get_danmu_all_page(target_id='5073281763', vid='d0033lhvt0u')
	df_9.insert(0, 'episodes', 9)

	# 第十集
	df_10 = get_danmu_all_page(target_id='5073281764', vid='y0033g6mubt')
	df_10.insert(0, 'episodes', 10)

	# 第十一集
	df_11 = get_danmu_all_page(target_id='5073281766', vid='a0033oeld03')
	df_11.insert(0, 'episodes', 11)

	# 第十二集
	df_12 = get_danmu_all_page(target_id='5073281765', vid='v0033e8g4w1')
	df_12.insert(0, 'episodes', 12)

	# 合并数据集
	df_all = pd.concat([df_1, df_2, df_3, df_4, df_5, df_6, df_7, df_8, df_9, df_10, df_11, df_12])

	# 写出数据
	df_all.to_csv('我是余欢水弹幕.csv', index=False)
