from flask import Flask, render_template, request, Response, jsonify
import keras
import cv2
import numpy as np
import json
app = Flask(__name__, template_folder='templates')

import cv2

camera = cv2.VideoCapture(0)
global start_button
start_button = False
global save_button
save_button = False

model = keras.models.load_model('model.h5')
labels_dict = {'0': 'A', '1': 'B', '2': 'C', '3': 'D', '4': 'E', '5': 'F', '6': 'G', '7': 'H', '8': 'I', '9': 'J', '10': 'K', '11': 'L', '12': 'M', '13': 'N', '14': 'O', '15': 'P', '16': 'Q', '17': 'R', '18': 'S', '19': 'T', '20': 'U', '21': 'V', '22': 'W', '23': 'X', '24': 'Y', '25': 'Z', '26': 'del', '27': 'nothing', '28': 'space'}

def generate_frames():
    i = 0
    global start_button
    while start_button:
        i += 1
        ## read the camera frame
        success,frame=camera.read()
        temp = frame[150:350, 50:250]
        cv2.rectangle(frame, pt1=(50,150), pt2=(250,350), color=(0,255,0), thickness=10)
        with open("static/data/log.json", "r") as jsonFile:
            data = json.load(jsonFile)

        
        data["predict_label"] = predict(temp)
        data["predict_prob"] = str(i/10000)
        data["buffer_text"] += data["predict_label"]

        with open("static/data/log.json", "w") as jsonFile:
            json.dump(data, jsonFile)

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

@app.route('/get_predict', methods = ['GET'])
def get_predict():
     with open("static/data/log.json", "r") as jsonFile:
            data = json.load(jsonFile)
     return jsonify({'predict_label': data["predict_label"],'predict_prob': data["predict_prob"], 'buffer_label': data["buffer_text"]})


@app.route('/')
def render_home():
    return render_template('home.html')

@app.route('/start_button', methods=['POST', 'GET'])
def toggle_start_button():
    global start_button
    global save_button
    start_button = True
    save_button = False
    return render_template('application.html')

@app.route('/stop_button', methods=['POST', 'GET'])
def toggle_stop_button():
    global start_button
    global save_button
    start_button = False
    save_button = False
    return render_template('application.html')

@app.route('/save_button', methods=['POST', 'GET'])
def toggle_save_button():
    with open("static/data/log.json", "r") as jsonFile:
                data = json.load(jsonFile)

    data["predict_label"] = ""
    data["predict_prob"] = ""
    data["buffer_text"] = ""

    with open("static/data/log.json", "w") as jsonFile:
        json.dump(data, jsonFile)
    global start_button
    global save_button
    save_button = True
    start_button =False
    return render_template('application.html')

@app.route('/application')
def render_application():
    return render_template('application.html')

@app.route('/contact')
def render_contact():
    return render_template('contact.html')


def predict(img):
    size = 64,64
    temp = cv2.imread(img)
    temp = cv2.resize(temp, size)
    img = temp.astype('float32')/255.0

    prediction= np.argmax(model.predict(img.reshape(1,64,64,3))[0])
    return labels_dict[str(prediction)]



app.run(host="", port=3000, debug=True, use_reloader=True)