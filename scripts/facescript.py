# first, import all necessary modules
import os
import pathlib

import blobconverter
import cv2
import depthai as dai
import numpy as np
from imutils.video import FPS
import modules.verify as verify
import modules.model as model
import datetime as dt
import copy


label_map = {1 : 'face'}
landmark_reference = np.float32([[0.31556875000000000, 0.4615741071428571],
                                [0.68262291666666670, 0.4615741071428571],
                                [0.50026249999999990, 0.6405053571428571],
                                [0.34947187500000004, 0.8246919642857142],
                                [0.65343645833333330, 0.8246919642857142]])
os.chdir('..')
os.chdir('unclassed-output')
colors = []

for i in range(10):
    colors.append(np.random.uniform(0, 255, 3))
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

cam_rgb = pipeline.createColorCamera()
cam_rgb.setPreviewSize(672, 384)
cam_rgb.setVideoSize(2*672, 2*384)
cam_rgb.setInterleaved(False)


face_nn = pipeline.createNeuralNetwork()
face_nn.setBlobPath(blobconverter.from_zoo('face-detection-adas-0001', shaves=6))
cam_rgb.preview.link(face_nn.input)

xout_rgb = pipeline.createXLinkOut()
xout_rgb.setStreamName("rgb")
cam_rgb.preview.link(xout_rgb.input)

xout_face = pipeline.createXLinkOut()
xout_face.setStreamName("face")
face_nn.out.link(xout_face.input)




with dai.Device(pipeline) as device:
    q_rgb = device.getOutputQueue("rgb")
    q_face = device.getOutputQueue("face")

    frame = None
    bboxes = []

    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)


    max = 10
    buffer = 60
    fps = FPS()
    fps.start()
    while True:
        fps.update()
        in_rgb = q_rgb.tryGet()
        in_face = q_face.tryGet()

        if in_rgb is not None:
            frame = in_rgb.getCvFrame()
            unaltered = copy.copy(frame)

        if in_face is not None:
            bboxes = np.array(in_face.getFirstLayerFp16())
            bboxes = bboxes.reshape((bboxes.size // 7, 7))
            confidence = bboxes[bboxes[:, 2] > 0.7][:, 2]
            labels = bboxes[bboxes[:, 2] > 0.7][:, 1]
            bboxes = bboxes[bboxes[:, 2] > 0.7][:, 3:7]

        face_frames = []
        color_iter = 0
        if frame is not None:
            for raw in bboxes:
                bbox = frameNorm(frame, raw)
                face_frame = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                face_frames.append(face_frame)
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), colors[color_iter], 2)
                buffer = buffer - 1
            cv2.imshow("preview", frame)
            if max > 0 and buffer <= 0 and len(confidence) > 0:
                cv2.imwrite(
                    f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}.png',
                    unaltered)
                for i in range(len(face_frames)):
                    model.det_to_json(label=label_map[labels[i]],
                                      time=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                      score=confidence[i],
                                      camera=device.getMxId(),
                                      dets=range(len(face_frames)))
                    cv2.imwrite(
                        f'{str(pathlib.PureWindowsPath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y%m%d-%H%M%S")}{i}_cut.png',
                        face_frames[i])
                max = max - 1
                face_frames = []
                buffer = 60
            color_iter = color_iter + 1

        if cv2.waitKey(1) == ord('q'):
            fps.stop()
            print('FPS: {:.2f}'.format(fps.fps()))
            break