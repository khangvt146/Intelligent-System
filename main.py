from flask import Flask, render_template

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

app.run(host="localhost", port=3000, debug=True, use_reloader=True)