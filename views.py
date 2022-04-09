from flask import Flask, render_template, request
from PIL import Image
import io, base64
from databaseController import DB

app = Flask(__name__)
db = DB()


def decode_base64(base64_string):
    ''' Decodes a base 64 image, and saves it as unknown_image.png '''
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
    ''' Checks for a face recognition and then returns a response based on if the face is recognized'''
    db = DB()
    user_request = dict(zip(request.form.keys(), request.form.values()))

    # ImmutableMultiDict([('image', 'b64code), ('Time', '1:47:17'), ('Date', '2022-4-5'), ('BrowserCodeName', 'Mozilla'), ('BrowserName', 'Netscape'), ('BrowserVersion', '5.0 (X11)'), ('CookiesEnabled', 'true'),
    # decoded_image =  decode_base64(user_request["image"])
    try:
        data = db.recognize_images(user_request)
        if data:
            return data
    except:
        return "Record not found", 400

def transaction_template(database_name):
    header, rows = db.get_logs(database_name)
    return render_template(
        "transactionLogs.html",
        header=header, rows=rows
    )

@app.route('/save', methods=['POST'])
def save_new_user():
    ''' Registers a user to the database '''
    x = 1
    user_request = dict(zip(request.form.keys(), request.form.values()))
    # replacement method
    # base64_string = request.form.get("image")
    # decode_image(base64_string, "unknown_image.png")
    db.save_image(user_request)
    db.send_to_log(user_request, 'registered_users')
    return render_template(
        "index.html",
    )

@app.route("/checkin")
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