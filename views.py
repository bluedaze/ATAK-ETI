from flask import Flask, render_template, request
from PIL import Image
import io, base64
from databaseController import DB
import csv

app = Flask(__name__)

@app.route("/")
@app.route("/index")
def index():
    '''
    This is the check in system. We use this to verify individuals have access.
    '''
    return render_template(
        "index.html",
    )


@app.route("/start")
def start():
    '''
    This is the check in system. We use this to verify individuals have access.
    '''
    return render_template(
        "start.html",
    )


@app.route('/register', methods=['GET', 'POST'])
def register():
    ''' Registers a user to the database '''
    if request.method == 'GET':
        return render_template('register.html')
    else:
        id_num = request.form.get("id")
        name = request.form.get("name")
        image_bytes = decode_base64()
        post = {"name": name, "id_num": id_num, "image": image_bytes}
        mydb = DB()
        mydb.send_to_log(post, 'registered_users')
    return "ok"

def decode_base64():
    ''' Decodes a base 64 image, and saves it as temp.png '''
    base64_string = request.form.get("image")
    b = io.BytesIO(base64.b64decode(base64_string))
    im = Image.open(b)
    image_bytes = io.BytesIO()
    im.save(image_bytes, format='PNG')
    im.save("./temp.png")
    return image_bytes.getvalue()

def check_recognition(image_bytes):
    mydb = DB()
    data = mydb.recognize_images(image_bytes)
    return data

@app.route('/verify', methods=['POST'])
def verify():
    ''' Checks for a face recognition and then returns a response based on if the face is recognized'''
    decoded_image =  decode_base64()
    data = check_recognition(decoded_image)
    if data:
        return data
    else:
        return "Record not found", 400

def transaction_template(database_name):
    mydb = DB()
    logs = mydb.get_logs(database_name)
    return render_template(
        "transactionLogs.html",
        logs=logs
    )

@app.route('/recognitions')
def recognitions():
    return transaction_template('recognized_faces')

@app.route('/registrants')
def registrants():
    return transaction_template('registered_users')

@app.route('/detections')
def detections():
    return transaction_template('detected_faces')