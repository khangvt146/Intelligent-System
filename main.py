from flask import Flask, render_template, request, Response
import keras
import cv2
import numpy as np

app = Flask(__name__, template_folder='templates')

import cv2

camera = cv2.VideoCapture(0)

global buffer_text
buffer_text = ''


def generate_frames():
    while True:

        ## read the camera frame
        success,frame=camera.read()
        temp = frame[150:350, 50:250]
        cv2.rectangle(frame, pt1=(50,150), pt2=(250,350), color=(0,255,0), thickness=10)

        if not success:
            break
        else:
            cv2.imwrite('static/data/crop.jpg', temp)

            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

        yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def render_home():
    return render_template('home.html')


@app.route('/application', methods=['POST', 'GET'])
def render_application():
    global buffer_text
    if request.method == 'POST':
        print('test post')
        if 'start_button' in request.form:
            path_image = r'img/webcam.png'
            path_gester = r'data/crop.jpg'
            # label_predict =  predict('static/'+path_gester)
            label_predict = 'A'
            buffer_text += label_predict
            return render_template('application.html', path_image = path_image , buffer = buffer_text, label_predict = label_predict, path_gester = path_gester)
        elif 'pause_button' in request.form:
            path_image = ''
            path_gester = ''
            # label_predict =  predict('static/'+path_gester)
            label_predict = 'B'
            buffer_text += label_predict
            return render_template('application.html', path_image = path_image , buffer = buffer_text, label_predict = label_predict, path_gester = path_gester)
        else:
            buffer_text = ''
            return render_template('application.html')
    else:
        return render_template('application.html')

@app.route('/contact')
def render_contact():
    return render_template('contact.html')


@app.route('/predict', methods=['POST'] )
def predict(path_image):
    print(path_image)
    labels_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J', '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z', '26': 'space', '27': 'del', '28': 'nothing'}

    img = load_img(path_image)
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