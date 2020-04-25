import os

import cv2
import paddlehub as hub

os.getcwd()
source_path = os.getcwd() + r'/learn/paddle_learn/paddle_hub/work/'
module = hub.Module(name="pyramidbox_lite_mobile_mask")

test_img_path = source_path + "hb.jpg"

# set input dict
input_dict = {"data": [cv2.imread(test_img_path)]}
results = module.face_detection(data=input_dict)
print(results)
