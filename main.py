from flask import Flask, render_template
import keras
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates')

@app.route('/')
def render_home():
    return render_template('home.html')


@app.route('/application')
def render_application():
    return render_template('application.html')


@app.route('/contact')
def render_contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST'] )
def predict():
    labels_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J', '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z', '26': 'space', '27': 'del', '28': 'nothing'}

    img = load_img('data/real/D.PNG')
    model = keras.models.load_model('model.h5')
    prediction= np.argmax(model.predict(img.reshape(1,64,64,3))[0])
    return labels_dict[str(prediction)]



def load_img(img):
    size = 64,64
    temp = cv2.imread(img)
    temp = cv2.resize(temp, size)
    image = temp.astype('float32')/255.0
    return image


app.run(host="", port=3000, debug=True, use_reloader=True)