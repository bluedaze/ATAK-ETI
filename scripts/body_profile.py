import globalVars
from scripts.base_camera import BaseCamera
import os
import pathlib

import blobconverter
import numpy as np
import cv2
import depthai as dai
import modules.model as model
import copy
import datetime as dt
import json
import pymongo

from scipy.spatial.distance import cosine
from modules.MultiMsgSync import TwoStageHostSeqSync

client = pymongo.MongoClient('45.79.221.195', 27017)
db = client['detections']

def create_pipeline():
    isExist = os.path.exists('classed-output')
    if isExist:
        os.chdir('classed-output')
    else:
        creating_directory = "Creating classed-output directory..."
        new_string = creating_directory.rjust(100)
        print("~" * 200 + "\n" + new_string + "\n" + "~" * 200 + "\n")
        os.makedirs('classed-output/vectors')
    openvinoVersion = "2021.4"
    p = dai.Pipeline()
    p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(720, 720)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    FPS = cam.getFps()

    # Send color frames to the host via XLink
    cam_xout = p.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")

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
    stereo.setDepthAlign(dai.CameraBoardSocket.RGB)
    stereo.setLeftRightCheck(True)
    stereo.setExtendedDisparity(False)
    monoLeft.out.link(stereo.left)
    monoRight.out.link(stereo.right)

    # Spatial Detection network if OAK-D
    body_nn = p.create(dai.node.MobileNetSpatialDetectionNetwork)
    stereo.depth.link(body_nn.inputDepth)
    body_nn.setBlobPath(str(blobconverter.from_zoo("person-detection-0200", shaves=4, version=openvinoVersion)))

    body_det_manip.out.link(body_nn.input)

    # Send ImageManipConfig to host so it can visualize the landmarks
    body_xout = p.create(dai.node.XLinkOut)
    body_xout.setStreamName("detection")

    # Script node will take the output from the NN as an input, get the first bounding box
    # and send ImageManipConfig to the manip_crop
    image_manip_script = p.create(dai.node.Script)
    body_nn.out.link(image_manip_script.inputs['nn_in'])
    cam.preview.link(image_manip_script.inputs['frame'])
    image_manip_script.setScript("""
    import time
    import json
    def limit_roi(det):
        if det.xmin <= 0: det.xmin = 0.001
        if det.ymin <= 0: det.ymin = 0.001
        if det.xmax >= 1: det.xmax = 0.999
        if det.ymax >= 1: det.ymax = 0.999
    delay = 120
    while True:
        frame = node.io['frame'].get()
        body_dets = node.io['nn_in'].get().detections
        for det in body_dets:
            limit_roi(det)
            # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(128, 256)
            cfg.setKeepAspectRatio(False)
            if delay <= 0:
                node.io['manip_cfg'].send(cfg)
                node.io['manip_img'].send(frame)
                delay = 120
            delay = delay - 1
    """)
    # This ImageManip will crop the mono frame based on the NN detections. Resulting image will be the cropped
    # face that was detected by the face-detection NN.
    manip_crop = p.create(dai.node.ImageManip)
    image_manip_script.outputs['manip_img'].link(manip_crop.inputImage)
    image_manip_script.outputs['manip_cfg'].link(manip_crop.inputConfig)
    manip_crop.initialConfig.setResize(128, 256)
    manip_crop.setWaitForConfigInput(True)

    # Second NN that detcts emotions from the cropped 64x64 face
    rec_nn = p.createNeuralNetwork()
    rec_nn.setBlobPath(
        str(blobconverter.from_zoo('person-reidentification-retail-0277', shaves=5, version=openvinoVersion)))
    manip_crop.out.link(rec_nn.input)

    rec_nn_xout = p.createXLinkOut()
    rec_nn_xout.setStreamName("recognition")
    rec_nn.out.link(rec_nn_xout.input)

    # Tracker Linking
    tracker = p.createObjectTracker()
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)

    trackerOut = p.create(dai.node.XLinkOut)
    trackerOut.setStreamName("tracklets")

    cam.preview.link(tracker.inputTrackerFrame)
    cam.preview.link(tracker.inputDetectionFrame)
    body_nn.out.link(tracker.inputDetections)
    tracker.out.link(trackerOut.input)
    tracker.passthroughTrackerFrame.link(cam_xout.input)
    tracker.passthroughDetections.link(body_xout.input)
    return p


device = dai.Device(create_pipeline())
class Camera(BaseCamera):

    @staticmethod
    def frames():
        queues = {}
        for name in ["color", "detection", "recognition"]:
            queues[name] = device.getOutputQueue(name)
        sync = TwoStageHostSeqSync()
        tracklets = device.getOutputQueue('tracklets')
        identities = {}

        buffer = 1000
        det_color = (255, 255, 255)
        target_color = (0, 0, 255)
        target_idx = 0
        target = None

        while True:
            # print(globalVars.keyPress)
            key = cv2.waitKey(1)
            for name, q in queues.items():
                if q.has():
                    sync.add_msg(q.get(), name)
            msgs = sync.get_msgs()
            track = tracklets.get()
            if msgs is not None:
                detections = msgs['detection'].detections
                frame = msgs['color'].getCvFrame()
                unaltered = copy.copy(frame)
                trackletsData = track.tracklets
                for i,t in enumerate(trackletsData):
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)
                    best_id = 1
                    if t.id not in identities.keys():
                        identity = 'unknown'
                        identities[t.id] = 'unknown'
                    if len(msgs['recognition']) != 0:
                        det_color = (255, 0, 0)
                        try:
                            score = msgs['recognition'][i].getFirstLayerFp16()
                            for file in os.listdir(f'{os.getcwd()}/vectors'):
                                if file.endswith('.json'):

                                    # One-liner used to open a JSON file to compare against the current embedding
                                    data = np.array(json.load(open(f'{os.getcwd()}/vectors/{file}'))['embed'])
                                    best_id = min(cosine(data, score), best_id)
                                    if cosine(data, score) == best_id:
                                        identity = file.removesuffix('.json')
                            if best_id > 0.4:
                                identities[t.id] = 'unknown'
                            else:
                                identities[t.id] = identity
                            cv2.imwrite(
                                f'{str(pathlib.PurePath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")}.png',
                                unaltered)
                            if identities[t.id] == 'unknown':
                                model.det_to_json(db,
                                                  label='person',
                                                  id=identity,
                                                  embed=score,
                                                  distance=f'{t.spatialCoordinates.z / 1000}',
                                                  angle=f'{np.arcsin((t.spatialCoordinates.x / 1000) / max(t.spatialCoordinates.z / 1000, 0.001))}',
                                                  time=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                                  camera=device.getMxId(),
                                                  postprocess=False,
                                                  dets=range(len(detections)))
                            cv2.imwrite(
                                f'{str(pathlib.PurePath(os.getcwd()))}/vectors/{identities[t.id]}_cut.png',
                                frame[y1:y2,x1:x2])
                        except IndexError:
                            print("Recognition missed, likely desynchronized")
                    if identities[t.id] == target:
                        det_color = target_color
                    cv2.putText(frame, identities[t.id], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), det_color, cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX,
                                0.5, 255)
                    det_color = (255,255,255)
                if len(trackletsData) == 0:
                    buffer = buffer - 1
                else:
                    buffer = 1000
                cv2.putText(frame, target, (frame.shape[1] - 10, frame.shape[0] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                # cv2.imshow("vision", frame)
                ret, buf = cv2.imencode('.jpg', frame)
                image = buf.tobytes()
                yield  (b'--frame\r\n'
                 b'Content-Type: image/jpeg\r\n\r\n' + image + b'\r\n')
            if buffer <= 0:
                key = ord('q')
            if globalVars.keyPress == "q":
                print(identities)
                globalVars.keyPress = None
                break
            elif key == ord(','):
                if target is not None:
                    id_keys = list(identities.keys())
                    target_idx = max(target_idx - 1, 0)
                    target = identities[id_keys[target_idx]]
                    print(f'Tracking identity {identities[target_idx]}')
            elif key == ord('.'):
                if target is not None:
                    id_keys = list(identities.keys())
                    target_idx = min(target_idx + 1, max(identities.keys()))
                    target = identities[id_keys[target_idx]]
                    print(f'Tracking identity {identities[target_idx]}')
            elif key == ord('t') and len(identities) != 0:
                if len(list(identities.keys())) != 0:
                    target = identities[min(identities.keys())]
