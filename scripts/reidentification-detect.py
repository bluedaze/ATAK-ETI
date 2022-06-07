import argparse
import queue
import threading
import signal
from pathlib import Path
import os

import blobconverter
import cv2
import depthai as dai
import numpy as np
import modules.verify as verify

from imutils.video import FPS
labelmap = ['Jackson']

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# First, we want the Color camera as the output
cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(672, 384)  # 672x384 will be the preview frame size, available as 'preview' output of the node
cam_rgb.setInterleaved(False)

# Next, we want a neural network that will produce the confidence
face_nn = pipeline.createNeuralNetwork()
# Blob is the Neural Network file, compiled for MyriadX. It contains both the definition and weights of the model
# We're using a blobconverter tool to retreive the MobileNetSSD blob automatically from OpenVINO Model Zoo
face_nn.setBlobPath(blobconverter.from_zoo('face-detection-adas-0001', shaves=6))

# Next, we link the camera 'preview' output to the neural network detection input, so that it can produce confidence
cam_rgb.preview.link(face_nn.input)

# XLinkOut is a "way out" from the device. Any data you want to transfer to host need to be send via XLink
xout_rgb = pipeline.createXLinkOut()
# For the rgb camera output, we want the XLink stream to be named "rgb"
xout_rgb.setStreamName("rgb")
# Linking camera preview to XLink input, so that the frames will be sent to host
cam_rgb.preview.link(xout_rgb.input)

# The same XLinkOut mechanism will be used to receive nn results
xout_face = pipeline.createXLinkOut()
xout_face.setStreamName("face")
face_nn.out.link(xout_face.input)


# Pipeline is now finished, and we need to find an available device to run our pipeline
# we are using context manager here that will dispose the device after we stop using it
with dai.Device(pipeline) as device:
    # From this point, the Device will be in "running" mode and will start sending data via XLink

    # To consume the device results, we get two output queues from the device, with stream names we assigned earlier
    q_rgb = device.getOutputQueue("rgb")
    q_face = device.getOutputQueue("face")

    # Here, some of the default values are defined. Frame will be an image from "rgb" stream, confidence will contain nn results
    frame = None
    bboxes = []

    # Since the confidence returned by nn have values from <0..1> range, they need to be multiplied by frame width/height to
    # receive the actual position of the bounding box on the image
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    max = 10
    buffer = 120
    # Main host-side application loop
    fps = FPS()
    fps.start()
    while True:
        fps.update()
        # we try to fetch the data from nn/rgb queues. tryGet will return either the data packet or None if there isn't any
        in_rgb = q_rgb.tryGet()
        in_face = q_face.tryGet()

        if in_rgb is not None:
            # If the packet from RGB camera is present, we're retrieving the frame in OpenCV format using getCvFrame
            frame = in_rgb.getCvFrame()
            unaltered = frame

        if in_face is not None:
            # when data from nn is received, we take the confidence array that contains mobilenet-ssd results
            bboxes = np.array(in_face.getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]


        if frame is not None:
            for raw in bboxes:
                # for each bounding box, we first normalize it to match the frame size
                bbox = frameNorm(frame, raw)
                face_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                confidence = verify.face_match(face_frame,'X:/Programming/ETI projects/unclassed-output/tmp5k4a2cpz_cut.png')
                # and then draw a rectangle on the frame to show the actual result
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
                cv2.putText(frame,str(confidence * 100)+'%',(bbox[0], bbox[1]),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)

            # After all the drawing is finished, we show the frame on the screen
            cv2.imshow("preview", frame)


        # at any time, you can press "q" and exit the main loop, therefore exiting the program itself
        if cv2.waitKey(1) == ord('q'):
            fps.stop()
            print('FPS: {:.2f}'.format(fps.fps()))
            break