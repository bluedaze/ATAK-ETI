import os
import pathlib
import gtts
from playsound import playsound

import blobconverter
import numpy as np
import cv2
import depthai as dai
import modules.model as model
import copy
import datetime as dt

from modules.MultiMsgSync import TwoStageHostSeqSync


class TextHelper:
    def __init__(self) -> None:
        self.bg_color = (0, 0, 0)
        self.color = (255, 255, 255)
        self.text_type = cv2.FONT_HERSHEY_SIMPLEX
        self.line_type = cv2.LINE_AA

    def putText(self, frame, text, coords):
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.bg_color, 4, self.line_type)
        cv2.putText(frame, text, coords, self.text_type, 0.5, self.color, 1, self.line_type)


os.chdir('..')
os.chdir('classed-output')
openvinoVersion = "2021.4"
p = dai.Pipeline()
p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

cam = p.create(dai.node.ColorCamera)
cam.setVideoSize(720,720)
cam.setInterleaved(False)
cam.setBoardSocket(dai.CameraBoardSocket.RGB)
FPS = cam.getFps()

# Send color frames to the host via XLink
cam_xout = p.create(dai.node.XLinkOut)
cam_xout.setStreamName("color")
cam.video.link(cam_xout.input)

# Crop 720x720 -> 256x256
body_det_manip = p.create(dai.node.ImageManip)
body_det_manip.initialConfig.setResize(256, 256)
cam.preview.link(body_det_manip.inputImage)

monoLeft = p.create(dai.node.MonoCamera)
monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)

monoRight = p.create(dai.node.MonoCamera)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

stereo = p.create(dai.node.StereoDepth)
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

# Spatial Detection network if OAK-D
body_nn = p.create(dai.node.MobileNetSpatialDetectionNetwork)
body_nn.setBoundingBoxScaleFactor(0.8)
body_nn.setDepthLowerThreshold(100)
body_nn.setDepthUpperThreshold(5000)
stereo.depth.link(body_nn.inputDepth)
body_nn.setBlobPath(str(blobconverter.from_zoo("person-detection-0200", shaves=6, version=openvinoVersion)))

body_det_manip.out.link(body_nn.input)

# Send ImageManipConfig to host so it can visualize the landmarks
body_xout = p.create(dai.node.XLinkOut)
body_xout.setStreamName("detection")
body_nn.out.link(body_xout.input)

# Script node will take the output from the NN as an input, get the first bounding box
# and send ImageManipConfig to the manip_crop
image_manip_script = p.create(dai.node.Script)
body_nn.out.link(image_manip_script.inputs['nn_in'])
cam.preview.link(image_manip_script.inputs['frame'])
image_manip_script.setScript("""
import time
def limit_roi(det):
    if det.xmin <= 0: det.xmin = 0.001
    if det.ymin <= 0: det.ymin = 0.001
    if det.xmax >= 1: det.xmax = 0.999
    if det.ymax >= 1: det.ymax = 0.999
while True:
    frame = node.io['frame'].get()
    face_dets = node.io['nn_in'].get().detections
    # node.warn(f"Faces detected: {len(face_dets)}")
    for det in face_dets:
        limit_roi(det)
        # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
        cfg = ImageManipConfig()
        cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
        cfg.setResize(80, 160)
        cfg.setKeepAspectRatio(False)
        node.io['manip_cfg'].send(cfg)
        node.io['manip_img'].send(frame)
        # node.warn(f"1 from nn_in: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
""")

# This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
# face that was detected by the face-detection NN.
manip_crop = p.create(dai.node.ImageManip)
image_manip_script.outputs['manip_img'].link(manip_crop.inputImage)
image_manip_script.outputs['manip_cfg'].link(manip_crop.inputConfig)
manip_crop.initialConfig.setResize(80, 160)
manip_crop.setWaitForConfigInput(True)

# Second NN that detcts emotions from the cropped 64x64 face
rec_nn = p.createNeuralNetwork()
rec_nn.setBlobPath(
    str(blobconverter.from_zoo("person-attributes-recognition-crossroad-0238", shaves=6, version=openvinoVersion)))
manip_crop.out.link(rec_nn.input)

rec_nn_xout = p.createXLinkOut()
rec_nn_xout.setStreamName("recognition")
rec_nn.out.link(rec_nn_xout.input)


with dai.Device(p) as device:
    queues = {}
    for name in ["color", "detection", "recognition"]:
        queues[name] = device.getOutputQueue(name)
    sync = TwoStageHostSeqSync()

    textHelper = TextHelper()
    buffer = 60
    det_color = (255, 255, 255)
    while True:
        for name,q in queues.items():
            if q.has():
                sync.add_msg(q.get(), name)
        msgs = sync.get_msgs()
        if msgs is not None:
            detections = msgs['detection'].detections
            frame = msgs['color'].getCvFrame()
            unaltered = copy.copy(frame)
            if buffer <= 0:
                det_color = (255,0,0)
            for i, detection in enumerate(detections):
                bbox = model.frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), det_color, 2)
                # Each face detection will be sent to emotion estimation model. Wait for the result
                nndata = msgs['recognition']
                # [print(f"Layer name: {l.name}, Type: {l.dataType}, Dimensions: {l.dims}") for l in nndata.getAllLayers()]
                scores = nndata[i].getFirstLayerFp16()
                features = np.round(scores)
                sex = 'female' if features[0] < 0.5 else 'male'
                bag = features[1]
                hat = features[2]
                sleeves = features[3]
                pants = features[4]
                hair = features[5]
                coat = features[6]


                textHelper.putText(frame, f"sex:{sex}", (bbox[0] + 10, bbox[1] + 20))
                textHelper.putText(frame, f"pants:{bool(pants)}", (bbox[0] + 10, bbox[1] + 40))
                textHelper.putText(frame, f"long-sleeves:{bool(sleeves)}", (bbox[0] + 10, bbox[1] + 60))
                y = (bbox[1] + bbox[3]) // 2
                x = (bbox[1] + bbox[3]) // 2
                coords = "Z: {:.2f} m".format(detection.spatialCoordinates.z / 1000)
                textHelper.putText(frame, coords, (x, y + 35))
                textHelper.putText(frame, coords, (x, y + 35))


                if buffer <= 0:
                    cv2.imwrite(
                        f'{str(pathlib.PurePath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")}.png',
                        unaltered)
                    model.det_to_json(label='person',
                                      sex=sex,
                                      sleeves=bool(sleeves),
                                      pants=bool(pants),
                                      distance=f'{detection.spatialCoordinates.z / 1000}m',
                                      time=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                      camera=device.getMxId(),
                                      dets=range(len(detections)))
                    cv2.imwrite(f'{str(pathlib.PurePath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")}_vehicle_cut.png', frame)
                    buffer = 60
                    # if buffer < 0:
                    #     tts = gtts.gTTS(f'{len(detections)} vehicles detected.', lang='es')
                    #     tts.save(f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{len(detections)}.mp3")
                    #     playsound(f"{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}_{len(detections)}.mp3")
                    #     buffer = 60
                buffer = buffer - 1
                det_color = (255,255,255)
            cv2.imshow("frame", frame)
        # if frame is not None:

        if cv2.waitKey(1) == ord('q'):
            break
