import keras
import cv2
import numpy as np

model = keras.models.load_model('model.h5')
labels_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J', '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z', '26': 'del', '27': '', '28': ' '}


def predict(path_img):
    size = 64,64
    img = cv2.imread(path_img)
    temp = cv2.resize(img, size)
    img = temp.astype('float32')/255.0
    predict = model.predict(img.reshape(1,64,64,3))[0]
    prob = np.max(predict)
    prediction= np.argmax(predict)
    return labels_dict[str(prediction)], str(round(prob, 2))


print(predict('static\data\A.PNG'))
