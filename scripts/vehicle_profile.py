import os
import pathlib

import blobconverter
import numpy as np
import cv2
import depthai as dai
import modules.model as model
import copy
import datetime as dt
import openvino
import json

from scipy.spatial.distance import cosine
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

class CTCCodec:
    """ Convert between text-label and text-index """

    def __init__(self, characters):
        # characters (str): set of the possible characters.
        dict_character = list(characters)

        self.dict = {}
        for i, char in enumerate(dict_character):
            self.dict[char] = i + 1

        self.characters = dict_character
        # print(self.characters)
        # input()

    def decode(self, preds):
        """ convert text-index into text-label. """
        texts = []
        index = 0
        # Select max probabilty (greedy decoding) then decode index to character
        preds = preds.astype(np.float16)
        preds_index = np.argmax(preds, 2)
        preds_index = preds_index.transpose(1, 0)
        preds_index_reshape = preds_index.reshape(-1)
        preds_sizes = np.array([preds_index.shape[1]] * preds_index.shape[0])

        for l in preds_sizes:
            t = preds_index_reshape[index:index + l]

            # NOTE: t might be zero size
            if t.shape[0] == 0:
                continue

            char_list = []
            for i in range(l):
                # removing repeated characters and blank.
                if not (i > 0 and t[i - 1] == t[i]):
                    if self.characters[t[i]] != '#':
                        char_list.append(self.characters[t[i]])
            text = ''.join(char_list)
            texts.append(text)

            index += l

        return texts


class vehicle_identification:
    characters = '0123456789abcdefghijklmnopqrstuvwxyz#'
    codec = CTCCodec(characters)


    os.chdir('..')
    os.chdir('classed-output')
    openvinoVersion = "2021.4"
    p = dai.Pipeline()
    p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

    cam = p.create(dai.node.ColorCamera)
    cam.setPreviewSize(720,720)
    cam.setInterleaved(False)
    cam.setBoardSocket(dai.CameraBoardSocket.RGB)
    FPS = cam.getFps()

    # Send color frames to the host via XLink
    cam_xout = p.create(dai.node.XLinkOut)
    cam_xout.setStreamName("color")


    # Crop 720x720 -> 256x256
    body_det_manip = p.create(dai.node.ImageManip)
    body_det_manip.initialConfig.setResize(300, 300)
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
    vehicle_nn = p.create(dai.node.MobileNetSpatialDetectionNetwork)
    vehicle_nn.setConfidenceThreshold(0.3)
    stereo.depth.link(vehicle_nn.inputDepth)
    vehicle_nn.setBlobPath(str(blobconverter.from_zoo("vehicle-license-plate-detection-barrier-0106", shaves=6, version=openvinoVersion)))

    body_det_manip.out.link(vehicle_nn.input)

    # Send ImageManipConfig to host so it can visualize the landmarks
    vehicle_xout = p.create(dai.node.XLinkOut)
    vehicle_xout.setStreamName("detection")


    # Script node will take the output from the NN as an input, get the first bounding box
    # and send ImageManipConfig to the manip_crop
    image_manip_script = p.create(dai.node.Script)
    vehicle_nn.out.link(image_manip_script.inputs['nn_in'])
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
        licenses = []
        frame = node.io['frame'].get()
        dets = node.io['nn_in'].get().detections
        for det in dets:
            if det.label == 2:
                licenses.append(det)
        for det in licenses:
            limit_roi(det)
            # node.warn(f"Detection rect: {det.xmin}, {det.ymin}, {det.xmax}, {det.ymax}")
            cfg = ImageManipConfig()
            cfg.setCropRect(det.xmin, det.ymin, det.xmax, det.ymax)
            cfg.setResize(120, 32)
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
    manip_crop.initialConfig.setResize(120, 32)
    manip_crop.setWaitForConfigInput(True)

    # Second NN that detcts emotions from the cropped 64x64 face
    rec_nn = p.createNeuralNetwork()
    rec_nn.setBlobPath(str(blobconverter.from_zoo("text-recognition-0012", shaves=6, version=openvinoVersion)))
    manip_crop.out.link(rec_nn.input)

    rec_nn_xout = p.createXLinkOut()
    rec_nn_xout.setStreamName("recognition")
    rec_nn.out.link(rec_nn_xout.input)

    # Tracker Linking
    tracker = p.createObjectTracker()
    tracker.setTrackerType(dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM)
    tracker.setTrackerIdAssignmentPolicy(dai.TrackerIdAssignmentPolicy.UNIQUE_ID)
    tracker.setDetectionLabelsToTrack([1]) # Vehicles only

    trackerOut = p.create(dai.node.XLinkOut)
    trackerOut.setStreamName("tracklets")

    cam.preview.link(tracker.inputTrackerFrame)
    cam.preview.link(tracker.inputDetectionFrame)
    vehicle_nn.out.link(tracker.inputDetections)
    tracker.out.link(trackerOut.input)
    tracker.passthroughTrackerFrame.link(cam_xout.input)
    tracker.passthroughDetections.link(vehicle_xout.input)

    with dai.Device(p, usb2Mode=True) as device:

        queues = {}
        for name in ["color", "detection", "recognition"]:
            queues[name] = device.getOutputQueue(name)
        sync = TwoStageHostSeqSync()
        tracklets = device.getOutputQueue('tracklets')
        licenses = {}

        textHelper = TextHelper()
        buffer = 120
        det_color = (255, 255, 255)
        target_color = (0, 0, 255)
        target_idx = 0
        target = None

        while True:
            key = cv2.waitKey(1)
            for name, q in queues.items():
                if q.has():
                    sync.add_msg(q.get(), name)
            msgs = sync.get_msgs()
            track = tracklets.get()
            if msgs is not None:
                detections = msgs['detection'].detections
                detected_licenses = [det for det in detections if det.label == 2]
                frame = msgs['color'].getCvFrame()
                unaltered = copy.copy(frame)
                trackletsData = track.tracklets
                for i,t in enumerate(trackletsData):
                    roi = t.roi.denormalize(frame.shape[1], frame.shape[0])
                    x1 = int(roi.topLeft().x)
                    y1 = int(roi.topLeft().y)
                    x2 = int(roi.bottomRight().x)
                    y2 = int(roi.bottomRight().y)
                    for lic in detected_licenses:
                        bbox = model.frame_norm(frame, (lic.xmin, lic.ymin, lic.xmax, lic.ymax))
                        w1 = bbox[0]
                        z1 = bbox[1]
                        w2 = bbox[2]
                        z2 = bbox[3]
                        if w1 > x1 and w2 < x2 and z1 > y1 and z2 < y2:
                            license = lic
                            licenses[t.id] = 'pending'
                    if t.id not in licenses.keys():
                        license = None
                        licenses[t.id] = 'unknown'
                    if len(msgs['recognition']) != 0 and license is not None:
                        det_color = (255, 0, 0)
                        try:
                            nndata = msgs['recognition'][i]
                            tensor = np.array(nndata.getFirstLayerFp16()).reshape(30,1,37)
                            plate = codec.decode(tensor)[0]
                            licenses[t.id] = plate
                            cv2.imwrite(
                                f'{str(pathlib.PurePath(os.getcwd()))}/{device.getMxId()}_{dt.datetime.now().strftime("%Y-%m-%d-%H%M%S")}.png',
                                unaltered)
                            model.det_to_json(label='vehicle',
                                              id=licenses[t.id],
                                              distance=f'{t.spatialCoordinates.z / 1000}',
                                              angle=f'{np.arcsin((t.spatialCoordinates.x / 1000) / (t.spatialCoordinates.z / 1000))}',
                                              time=dt.datetime.now().strftime("%Y%m%d-%H%M%S"),
                                              camera=device.getMxId(),
                                              postprocess=False,
                                              dets=range(len(detections)))
                            cv2.imwrite(
                                f'{str(pathlib.PurePath(os.getcwd()))}/vectors/{licenses[t.id]}_cut.png',
                                frame[y1:y2,x1:x2])
                        except IndexError:
                            print("Recognition missed, likely desynchronized")
                    if licenses[t.id] == target:
                        det_color = target_color
                    cv2.putText(frame, licenses[t.id], (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, f"ID: {[t.id]}", (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.putText(frame, t.status.name, (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), det_color, cv2.FONT_HERSHEY_SIMPLEX)
                    cv2.putText(frame, f"Z: {int(t.spatialCoordinates.z)} mm", (x1 + 10, y1 + 95), cv2.FONT_HERSHEY_TRIPLEX,
                                0.5, 255)
                    det_color = (255,255,255)
                cv2.putText(frame, target, (frame.shape[1] - 10, frame.shape[0] - 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
                cv2.imshow("vision", frame)

            if key == ord('q'):
                print(licenses)
                break
            elif key == ord(','):
                if target is not None:
                    id_keys = list(licenses.keys())
                    target_idx = max(target_idx - 1, 0)
                    target = licenses[id_keys[target_idx]]
                    print(f'Tracking identity {licenses[target_idx]}')
            elif key == ord('.'):
                if target is not None:
                    id_keys = list(licenses.keys())
                    target_idx = min(target_idx + 1, max(licenses.keys()))
                    target = licenses[id_keys[target_idx]]
                    print(f'Tracking identity {licenses[target_idx]}')
            elif key == ord('t') and len(licenses) != 0:
                if len(list(licenses.keys())) != 0:
                    target = licenses[min(licenses.keys())]
