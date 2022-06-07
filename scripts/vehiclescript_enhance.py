# first, import all necessary modulesq
import os
import pathlib

import blobconverter
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS
import datetime as dt
from modules.model import det_to_json, frame_norm
import copy

class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA
    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)


color_map = ['white', 'gray', 'yellow', 'red', 'green', 'blue', 'black']
type_map = ['car', 'bus', 'truck', 'van']
os.chdir('..')
os.chdir('unclassed-output')
colors = []

for i in range(10):
    colors.append(np.random.uniform(0, 255, 3))

pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Cam
cam_rgb = pipeline.createColorCamera()
cam_rgb.setIspScale(2,3)
cam_rgb.setInterleaved(False)
cam_rgb.setVideoSize(720, 720)
cam_rgb.setPreviewSize(720, 720)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.video.link(xout_rgb.input)

# Vehicle Manip
vehicle_manip = pipeline.createImageManip()
vehicle_manip.initialConfig.setResize(256, 256)
cam_rgb.preview.link(vehicle_manip.inputImage)

# Neural Net
vehicle_nn = pipeline.createMobileNetDetectionNetwork()
vehicle_nn.setConfidenceThreshold(0.3)
vehicle_nn.setBlobPath(str(blobconverter.from_zoo('vehicle-detection-0200', shaves=6)))
vehicle_manip.out.link(vehicle_nn.input)

config_xout = pipeline.createXLinkOut()
config_xout.setStreamName("vehicle_det")
vehicle_nn.out.link(config_xout.input)

image_manip_script = pipeline.create(dai.node.Script)
vehicle_nn.out.link(image_manip_script.inputs['nn_in'])
cam_rgb.preview.link(image_manip_script.inputs['frame'])
image_manip_script.setScript("""
import time
# def limit_roi(det):
#     if det.xmin <= 0: det.xmin = 0.001
#     if det.ymin <= 0: det.ymin = 0.001
#     if det.xmax >= 1: det.xmax = 0.999
#     if det.ymax >= 1: det.ymax = 0.999
# while True:
#     frame = node.io['frame'].get()
#     vehicle_dets = node.io['nn_in'].get().detections
#     # node.warn(f"Vehicles detected: {len(vehicle_dets)}")
#     for det in vehicle_dets:
#         limit_roi(det)
#         # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
#         cfg = ImageManipConfig()
#         cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
#         cfg.setResize(72, 72)
#         cfg.setKeepAspectRatio(False)
#         node.io['manip_cfg'].send(cfg)
#         node.io['manip_img'].send(frame)
#         # node.warn(f"1 from nn_in: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
# """)

# Manip to bounding boxes
manip_crop = pipeline.createImageManip()
image_manip_script.outputs['manip_img'].link(manip_crop.inputImage)
image_manip_script.outputs['manip_cfg'].link(manip_crop.inputConfig)
manip_crop.initialConfig.setResize(72, 72)
manip_crop.setWaitForConfigInput(True)

# Neural Net
attr_nn = pipeline.createNeuralNetwork()
attr_nn.setBlobPath(str(blobconverter.from_zoo('vehicle-attributes-recognition-barrier-0042', shaves=6)))
manip_crop.out.link(attr_nn.input)

xout_attr = pipeline.createXLinkOut()
xout_attr.setStreamName("attributes")
attr_nn.out.link(xout_attr.input)

print("Sending Pipeline to device...")
with dai.Device(pipeline) as device:
    print("Obtaining camera output..")
    q_rgb = device.getOutputQueue("rgb", maxSize=1, blocking=False)
    print("Interpreting vehicle detection network...")
    q_vehicle = device.getOutputQueue("vehicle_det", maxSize=4, blocking=False)
    print("Extracting attributes...")
    q_attr = device.getOutputQueue('attributes', maxSize=4, blocking=False)

    max = 10
    buffer = 60
    texthelper = TextHelper()
    print("Main program loop initialized.")
    while True:
        if q_vehicle.has():
            bboxes = np.array(q_vehicle.get().detections)
            frame = q_rgb.get().getCvFrame()
            for det in bboxes:
                bbox = frame_norm(frame, (det.xmin, det.ymin, det.xmax, det.ymax))
                # cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[0], 2)
                # attr_data = q_attr.tryGet()
                # color = np.array(attr_data.getFirstLayerFp16())
                # texthelper.putText(frame, f"{color_name}, {color_name}%", (bbox[0] + 10, bbox[1] + 20))
            cv2.imshow('vehicle', frame)
            if max > 0 >= buffer:
                cv2.imwrite(
                    f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")}.png',
                    unaltered)
                for i in range(len(v_frames)):
                    det_to_json(label=type_map[type_name],
                                time=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                camera=device.getMxId(),
                                dets=range(len(v_frames)))
                    cv2.imwrite(
                        f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")}{i}_cut.png',
                        v_frames[i])
                max = max - 1
                buffer = 60

            if cv2.waitKey(1) == ord('q'):
                break
