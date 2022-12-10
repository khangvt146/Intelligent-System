import keras
from keras.layers import *
from keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import tensorflow as tf

model = keras.models.load_model('model.h5')
labels_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J', '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z', '26': 'del', '27': '', '28': ' '}


def predict(path_img):
    image = tf.keras.preprocessing.image.load_img(path_img, grayscale=False, color_mode="rgb", target_size=(64,64), interpolation="bilinear")
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    predict = model.predict(input_arr)
    prob = np.max(predict)
    result = labels_dict[str(np.argmax(predict))]
    return result,prob



print(predict('static\data\A_test.jpg'))
