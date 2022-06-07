# Functions for selecting/identifying cameras to run with the model

import cv2
import numpy
import os


#  Returns a list of available usb-cameras to capture from, identified by internal port value
def show_available_cams():
    valid = True
    port = 0
    buffer = 5
    available_cams = []
    while valid:
        test_cam = cv2.VideoCapture(port)
        if not test_cam.isOpened():
            valid = False
            buffer = buffer - 1
            if buffer <= 0:
                return available_cams
        else:
            available_cams.append(port)
            test_cam.release()
    return available_cams

# TODO
def set_cam_names(names: list, out: str):
    cams = show_available_cams()

# TODO
def load_cam_names(directory: str):
    if not os.path.exists(directory):
        print('No camera names file found. Use camera.set_cam_names() to create a camera names directory.')
    else:
        names = open(directory, 'r')
        return names
