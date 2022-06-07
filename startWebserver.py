#!/usr/bin/env python
from importlib import import_module
import os
from flask import Flask, render_template, Response, request



# import camera driver
import globalVars

if os.environ.get('CAMERA'):
    Camera = import_module('camera_' + os.environ['CAMERA']).Camera
else:
    from scripts.body_profile import Camera

# Raspberry Pi camera module (requires picamera package)
# from camera_pi import Camera

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    """Video streaming home page."""
    return render_template('index.html')

@app.route("/checkin")
def checkin():
    '''
    This is the check in system. We use this to verify individuals have access.
    '''
    return render_template(
        "checkin.html",
    )

@app.route('/keypresses', methods=['POST'])
def keypresses():
    ''' gets keypresses from the page '''
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            data = request.get_json()
            globalVars.keyPress = data["key"]
        print(globalVars.keyPress)
        status_code = Response(status=201)
        return status_code
    else:
        status_code = Response(status=400)
        return  status_code

def gen(camera):
    """Video streaming generator function."""
    yield b'--frame\r\n'
    while True:
        frame = camera.get_frame()
        yield b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n--frame\r\n'


@app.route('/video_feed', methods=['GET'])
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen(Camera()),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', threaded=True)
