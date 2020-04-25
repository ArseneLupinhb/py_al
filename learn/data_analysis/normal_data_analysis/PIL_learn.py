import os

import matplotlib.pyplot as plt
from PIL import Image

source_path = os.getcwd() + r'/learn/data_analysis/normal_data_analysis/work/'

# 显示matplotlib生成的图形
# %matplotlib inline

# 读取图片
img = Image.open(source_path + r'person.jpg')

# 显示图片
# img.show() #自动调用计算机上显示图片的工具
plt.imshow(img)
plt.show()

# 获得图像的模式 RGB
img_mode = img.mode
print(img_mode)

# 获取图像的宽、高
width, height = img.size
print(width, height)

# 将图片旋转45度
img_rotate = img.rotate(45)
# 显示旋转后的图片
plt.imshow(img_rotate)
plt.show()

# 缩放
img2 = Image.open(source_path + r'person.jpg')
img2_resize_result = img2.resize((int(width * 0.6), int(height * 0.6)), Image.ANTIALIAS)
print(img2_resize_result.size)
# 保存图片
img2_resize_result.save(source_path + 'person_resize_result.jpg')
# 展示图片
plt.imshow(img2_resize_result)
plt.show()

#  镜像图片
img3 = Image.open(source_path + r'person.jpg')
# 左右镜像
img3_lr = img3.transpose(Image.FLIP_LEFT_RIGHT)
# 展示左右镜像图片
plt.imshow(img3_lr)
plt.show()
# 上下镜像
img3_bt = img3.transpose(Image.FLIP_TOP_BOTTOM)
# 展示上下镜像图片
plt.imshow(img3_bt)
plt.show()
