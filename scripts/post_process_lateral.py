import json
import os
import pathlib
import time

import cv2
import depthai as dai
from modules.model import to_planar,frame_norm
import matplotlib.pyplot as plt
from tabulate import tabulate
import subprocess
import blobconverter
import numpy as np
import pandas as pd
from openvino.runtime import Core


def group_json_to_csv(show=True, **kwargs):
    # Nonsensical string hackery to extract latitude and longitude, will be replaced
    geospatial = str(subprocess.run('curl ipinfo.io', stdout=subprocess.PIPE).stdout).replace(' ', '').replace('\\n','')[3:].removesuffix("}'")
    lat = float(geospatial[geospatial.find('loc')+6:geospatial.find('loc')+13])
    long = float(geospatial[geospatial.find('org')-10:geospatial.find('org')-3])
    full_data = pd.DataFrame()
    for file in os.listdir(pathlib.Path(f'{os.getcwd()}/vectors')):
        if file.endswith('.json'):
            vector = json.load(open(f'{pathlib.Path(os.getcwd()).resolve()}/vectors/{file}'))
            reformed = {}
            for key,value in vector.items():
                reformed[key.upper()] = vector[key]
            row = pd.Series(reformed, name=reformed['TIME'])
            row['LAT'] = lat
            row['LONG'] = long
            full_data = full_data.append(row)
    if show == True:
        print(tabulate(full_data, headers=full_data.columns))
    with open(f'{pathlib.Path(os.getcwd()).resolve()}/tables/full_data.csv', 'w') as f:
        f.write(full_data.to_csv(index=False))

def clarify(**kwargs):
    if kwargs['label']=='person':
        openvinoVersion = "2021.4"
        p = dai.Pipeline()
        p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        img_xin = p.createXLinkIn()
        img_xin.setStreamName('Image')

        # Crop 720x720 -> 256x256
        body_det_manip = p.create(dai.node.ImageManip)
        body_det_manip.initialConfig.setResize(80, 160)
        img_xin.out.link(body_det_manip.inputImage)

        # Spatial Detection network if OAK-D
        body_nn = p.create(dai.node.NeuralNetwork)
        body_nn.setBlobPath(str(blobconverter.from_zoo("person-attributes-recognition-crossroad-0230", shaves=13, version=openvinoVersion)))

        body_det_manip.out.link(body_nn.input)



        # Send ImageManipConfig to host so it can visualize the landmarks
        body_xout = p.create(dai.node.XLinkOut)
        body_xout.setStreamName("detection")
        body_nn.out.link(body_xout.input)


        with dai.Device(p) as device:
            frames = device.getInputQueue('Image')
            data = device.getOutputQueue('detection')
            files = os.listdir(pathlib.Path(f'{os.getcwd()}/vectors'))
            for file in files:
                if file.endswith('.png') and file.removesuffix('_cut.png')+'.json' in files:
                    raw = cv2.imread(f'{pathlib.Path(os.getcwd()).resolve()}/vectors/{file}')
                    img = dai.ImgFrame()
                    img.setData(to_planar(raw, (80,160)))
                    img.setType(dai.RawImgFrame.Type.BGR888p)
                    img.setWidth(80)
                    img.setHeight(160)
                    frames.send(img)
                    scores = data.get().getFirstLayerFp16()
                    features = np.round(scores)
                    sex = 'female' if features[0] < 0.5 else 'male'
                    bag = features[1]
                    hat = features[2]
                    sleeves = features[3]
                    pants = features[4]
                    hair = features[5]
                    coat = features[6]
                    vector = json.load(open(f'{pathlib.Path(os.getcwd()).resolve()}/vectors/{file.removesuffix("_cut.png")+".json"}','r'))
                    vector['sex'] = sex
                    vector['bag'] = bool(bag)
                    vector['hat'] = bool(hat)
                    vector['sleeves'] = bool(sleeves)
                    vector['pants'] = bool(pants)
                    vector['hair'] = bool(hair)
                    vector['coat'] = bool(coat)
                    vector['sex_conf'] = scores[0]
                    vector['bag_conf'] = scores[1]
                    vector['hat_conf'] = scores[2]
                    vector['sleeves_conf'] = scores[3]
                    vector['pants_conf'] = scores[4]
                    vector['hair_conf'] = scores[5]
                    vector['coat_conf'] = scores[6]
                    vector['postprocess'] = True
                    with open(f'{pathlib.Path(os.getcwd()).resolve()}/vectors/{file.removesuffix("_cut.png")+".json"}', 'w') as out:
                        json.dump(vector, out, indent=4)
            group_json_to_csv()

    if kwargs['label']=='vehicle':
        color_map = ['white', 'gray', 'yellow', 'red', 'green', 'blue', 'black']
        type_map = ['car', 'bus', 'truck', 'van']
        openvinoVersion = "2021.4"
        p = dai.Pipeline()
        p.setOpenVINOVersion(version=dai.OpenVINO.Version.VERSION_2021_4)

        img_xin = p.createXLinkIn()
        img_xin.setStreamName('Image')

        # Crop 720x720 -> 256x256
        body_det_manip = p.create(dai.node.ImageManip)
        body_det_manip.initialConfig.setResize(72, 72)
        img_xin.out.link(body_det_manip.inputImage)

        # Spatial Detection network if OAK-D
        body_nn = p.create(dai.node.NeuralNetwork)
        body_nn.setBlobPath(str(blobconverter.from_zoo("vehicle-attributes-recognition-crossroad-0042", shaves=13, version=openvinoVersion)))

        body_det_manip.out.link(body_nn.input)



        # Send ImageManipConfig to host so it can visualize the landmarks
        body_xout = p.create(dai.node.XLinkOut)
        body_xout.setStreamName("detection")
        body_nn.out.link(body_xout.input)


        with dai.Device(p) as device:
            frames = device.getInputQueue('Image')
            data = device.getOutputQueue('detection')
            files = os.listdir(pathlib.Path(f'{os.getcwd()}/vectors'))
            for file in files:
                if file.endswith('.png') and file.removesuffix('_cut.png')+'.json' in files:
                    raw = cv2.imread(f'{pathlib.Path(os.getcwd()).resolve()}/vectors/{file}')
                    img = dai.ImgFrame()
                    img.setData(to_planar(raw, (72,72)))
                    img.setType(dai.RawImgFrame.Type.BGR888p)
                    img.setWidth(72)
                    img.setHeight(72)
                    frames.send(img)

                    result = data.get()
                    colors = result.getLayer('color')
                    forms = result.getLayer('color')
                    color = color_map[np.argmax(colors)]
                    form = type_map[np.argmax(forms)]

                    vector = json.load(open(f'{pathlib.Path(os.getcwd()).resolve()}/vectors/{file.removesuffix("_cut.png")+".json"}','r'))
                    vector['color'] = color
                    vector['type'] = form
                    vector['postprocess'] = True
                    with open(f'{pathlib.Path(os.getcwd()).resolve()}/vectors/{file.removesuffix("_cut.png")+".json"}', 'w') as out:
                        json.dump(vector, out, indent=4)
            group_json_to_csv()

