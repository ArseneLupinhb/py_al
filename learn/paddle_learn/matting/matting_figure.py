import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import paddlehub as hub


def show_initial_image():
	global test_img_path, img
	# 待预测图片
	# test_img_path = ["./meditation.jpg"]
	test_img_path = ["work/个人.jpg"]
	# test_img_path = ["./merge.jpg"]
	img = mpimg.imread(test_img_path[0])
	# 展示待预测图片
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.axis('off')
	plt.show()


def matting_image():
	global test_img_path, img
	module = hub.Module(name="deeplabv3p_xception65_humanseg")
	input_dict = {"image": test_img_path}
	# execute predict and print the result
	results = module.segmentation(data=input_dict)
	for result in results:
		print(result)
	# 预测结果展示
	# test_img_path = "./humanseg_output/meditation.png"
	test_img_path = "./humanseg_output/自拍.png"
	img = mpimg.imread(test_img_path)
	plt.figure(figsize=(10, 10))
	plt.imshow(img)
	plt.axis('off')
	plt.show()


if __name__ == '__main__':
	show_initial_image()
	matting_image()
