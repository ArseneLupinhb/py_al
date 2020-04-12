import os

import face_recognition

source_path = os.getcwd()

image = face_recognition.load_image_file(source_path + r"/idea/data/hb.jpg")
face_locations = face_recognition.face_locations(image)
face_locations
