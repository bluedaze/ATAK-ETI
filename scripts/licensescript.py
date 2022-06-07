# first, import all necessary modules
import os
import pathlib


import blobconverter
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS
import tempfile
import modules.model as model
import datetime as dt
import copy

colors = []
label_map = {1 : 'vehicle', 2: 'license-plate'}
for i in range(10):
    colors.append(np.random.uniform(0, 255, 3))

os.chdir('..')
os.chdir('unclassed-output')
tempfile.tempdir = os.getcwd()
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(300, 300)
cam_rgb.setVideoSize(720, 720)
cam_rgb.setInterleaved(False)

license_nn = pipeline.createNeuralNetwork()
license_nn.setBlobPath(str(blobconverter.from_zoo('vehicle-license-plate-detection-barrier-0106', shaves=8)))

cam_rgb.preview.link(license_nn.input)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_license = pipeline.createXLinkOut()
xout_license.setStreamName("license")
license_nn.out.link(xout_license.input)


with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_license = device.getOutputQueue("license")

    frame = None
    bboxes = []

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    max = 10
    buffer = 10
    fps = FPS()
    fps.start()
    while True:
        fps.update()
        in_rgb = q_rgb.tryGet()
        in_license = q_license.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            unaltered = copy.copy(frame)

        l_frames = []
        if in_license is not None:
            bboxes = np.array(in_license.getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            confidence = bboxes[bboxes[:, 2] > 0.7][:, 2]
            labels = bboxes[bboxes[:, 2] > 0.7][:, 1]
            bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

        color_iter = 0
        if frame is not None:
            for raw in bboxes:
                bbox = frameNorm(frame, raw)
                license_frame = unaltered[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                l_frames.append(license_frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[color_iter], 2)
                color_iter = color_iter + 1
                buffer = buffer - 1
            cv2.imshow("preview", frame)
            if max > 0 and buffer <= 0 and len(confidence) > 0:
                cv2.imwrite(
                    f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}.png',
                    unaltered)
                for i in range(len(l_frames)):
                    model.det_to_json(label=label_map[labels[i]],
                                      time=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                      score=confidence[i],
                                      camera=device.getMxId(),
                                      dets=range(len(l_frames)))
                    cv2.imwrite(
                        f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}{i}_cut.png',
                        l_frames[i])
                max = max - 1
                l_frames = []
                buffer = 60

        if cv2.waitKey(1) == ord('q'):
            fps.stop()
            print('FPS: {:.2f}'.format(fps.fps()))
            break