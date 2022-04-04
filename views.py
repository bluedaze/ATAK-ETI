from flask import Flask, render_template, request
from PIL import Image
import io, base64
from databaseController import DB

app = Flask(__name__)
db = DB()


def decode_base64():
    ''' Decodes a base 64 image, and saves it as unknown_image.png '''
    base64_string = request.form.get("image")
    b = io.BytesIO(base64.b64decode(base64_string))
    im = Image.open(b)
    image_bytes = io.BytesIO()
    im.save(image_bytes, format='PNG')
    im.save("./unknown_image.png")
    return image_bytes.getvalue()

def decode_image(base64_string, name):
    # Convert from base64 string to base64 bytes
    b64 = base64.b64decode(base64_string)
    # Convert to b64 binary object.
    base64_bytes = base64.b64encode(b64)
    with open(name, "wb") as fh:
        fh.write(base64.decodebytes(base64_bytes))
        return fh

@app.route('/verify', methods=['POST'])
def verify():
    ''' Checks for a face recognition and then returns a response based on if the face is recognized '''
    user_request = dict(zip(request.form.keys(), request.form.values()))
    data = db.process_request(user_request)
    if data:
        return data
    else:
        return "Record not found", 400

def transaction_template(database_name):
    logs = db.get_logs(database_name)
    return render_template(
        "transactionLogs.html",
        logs=logs
    )

@app.route('/save', methods=['POST'])
def save_new_user():
    ''' Registers a user to the database '''
    form_request = dict(zip(request.form.keys(), request.form.values()))
    # replacement method
    # base64_string = request.form.get("image")
    # decode_image(base64_string, "unknown_image.png")
    image_bytes = decode_base64()
    form_request["image"] = image_bytes
    db.send_to_log(form_request, 'registered_users')

@app.route("/start")
def checkin():
    '''
    This is the check in system. We use this to verify individuals have access.
    '''
    return render_template(
        "checkin.html",
    )

@app.route("/")
@app.route("/index")
def index():
    '''
    This is the check in system. We use this to verify individuals have access.
    '''
    return render_template(
        "index.html",
    )

@app.route('/register', methods=['GET'])
def register():
    ''' Registers a user to the database '''
    return render_template('register.html')

@app.route('/recognitions')
def recognitions():
    return transaction_template('recognized_faces')

@app.route('/registrants')
def registrants():
    return transaction_template('registered_users')

@app.route('/detections')
def detections():
    return transaction_template('detected_faces')