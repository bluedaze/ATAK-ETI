# first, import all necessary modulesq
import os
import pathlib


import blobconverter
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS
import datetime as dt
import copy
from modules.model import frame_norm, to_planar

label_map = {1 : 'vehicle'}
os.chdir('..')
os.chdir('unclassed-output')
colors = []

for i in range(10):
    colors.append(np.random.uniform(0, 255, 3))

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(512, 512)
cam_rgb.setInterleaved(False)

vehicle_nn = pipeline.createNeuralNetwork()
vehicle_nn.setBlobPath(str(blobconverter.from_zoo('vehicle-detection-0202', shaves=6)))
license_nn = pipeline.createNeuralNetwork()
license_nn.setBlobPath(str(blobconverter.from_zoo('vehicle-license-plate-detection-barrier-0106', shaves=6)))

cam_rgb.preview.link(vehicle_nn.input)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_vehicle = pipeline.createXLinkOut()
xout_vehicle.setStreamName("vehicle")
vehicle_nn.out.link(xout_vehicle.input)

xin_license = pipeline.createXLinkIn()
xin_license.setStreamName('license_in')
xin_license.out.link(license_nn.input)
xout_license = pipeline.createXLinkOut()
xout_license.setStreamName("license_nn")
license_nn.out.link(xout_license.input)

with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_vehicle = device.getOutputQueue("vehicle", maxSize=1, blocking=False)
    q_license = device.getInputQueue("license_in")

    frame = None
    vehicle_frame = None
    bboxes = []
    l_bboxes = []
    max = 10
    buffer = 60
    fps = FPS()
    fps.start()

    while True:
        fps.update()
        in_rgb = q_rgb.tryGet()
        in_vehicle = q_vehicle.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            unaltered = copy.copy(frame)

        if in_vehicle is not None:
            bboxes = np.array(in_vehicle.getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            confidence = bboxes[bboxes[:, 2] > 0.7][:, 2]
            labels = bboxes[bboxes[:, 2] > 0.7][:, 1]
            bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]
        if vehicle_frame is not None:
            license_data = dai.NNData()
            license_data.setLayer('0', to_planar(vehicle_frame, (300, 300)))
            q_license.send(license_data)
            lic_nn = device.getOutputQueue("license_nn", maxSize=1, blocking=False)
            in_license = lic_nn.tryGet()
            if in_license is not None:
                l_bboxes = np.array(in_license.getFirstLayerFp16())
                l_bboxes = l_bboxes.reshape((l_bboxes.size // 7, 7))
                l_confidence = l_bboxes[l_bboxes[:, 2] > 0.7][:, 2]
                l_labels = l_bboxes[l_bboxes[:, 2] > 0.7][:, 1]
                l_bboxes = l_bboxes[l_bboxes[:, 2] > 0.7][:, 3:7]
        print(l_bboxes)
        v_frames = []
        color_iter = 0
        if frame is not None:
            for raw in bboxes:
                bbox = frame_norm(frame, raw)
                vehicle_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]

                v_frames.append(vehicle_frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[color_iter], 2)
                buffer = buffer - 1
            cv2.imshow("preview", frame)
            if max > 0 and buffer <= 0 and len(confidence) > 0:
                cv2.imwrite(f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}.png', unaltered)
                for i in range(len(v_frames)):
                    cv2.imwrite(f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}{i}_cut.png',
                                v_frames[i])
                max = max - 1
                v_frames = []
                buffer = 60
            color_iter = color_iter + 1


        if cv2.waitKey(1) == ord('q'):
            fps.stop()
            print('FPS: {:.2f}'.format(fps.fps()))
            break