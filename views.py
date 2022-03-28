from flask import Flask, render_template, request
from PIL import Image
import io, base64


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

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
        id_num = request.form.get("id")
        name = request.form.get("name")
        base64_string = request.form.get("image")
        sbuf = io.StringIO()
        sbuf.write(base64_string)
        b = io.BytesIO(base64.b64decode(base64_string))
        pimg = Image.open(b)
        # pimg.show() Opens image in new window, good for testing.
    return "ok"
