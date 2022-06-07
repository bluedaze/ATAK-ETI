# Functions for the purpose of verifying detected objects and exporting into a database (if detect, then ...)
import cv2
import pandas as pd
import depthai as dai
import blobconverter
import numpy as np
from scipy.spatial import distance
import os
from pathlib import Path



# TODO If class = PERSON, verify:
#   Face
#   Clothes
#   Iris
#   ...

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

def frameNorm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def face_normalize(face):
    if type(face) == str:
        rface = cv2.imread(face)
    else:
        rface = face
    landmark_reference = np.float32([[0.31556875000000000, 0.4615741071428571],
                                     [0.68262291666666670, 0.4615741071428571],
                                     [0.50026249999999990, 0.6405053571428571],
                                     [0.34947187500000004, 0.8246919642857142],
                                     [0.65343645833333330, 0.8246919642857142]])
    pipeline = dai.Pipeline()
    norm = pipeline.createNeuralNetwork()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

    norm.setBlobPath(blobconverter.from_zoo('landmarks-regression-retail-0009', shaves=6))

    xin_norm = pipeline.createXLinkIn()
    xin_norm.setStreamName('norm_in')
    xin_norm.out.link(norm.input)
    xout_norm = pipeline.createXLinkOut()
    xout_norm.setStreamName("norm_net")
    norm.out.link(xout_norm.input)
    height = rface.shape[0]
    width = rface.shape[1]

    with dai.Device(pipeline) as device:
        land_in = device.getInputQueue('norm_in')

        land_data = dai.NNData()
        land_data.setLayer("0", to_planar(rface, (48, 48)))
        land_in.send(land_data)

        land_nn = device.getOutputQueue('norm_net', maxSize=1)
        result = land_nn.get().getFirstLayerFp16()
        data = frameNorm(rface, result)
        zipped = zip(result[0:8:2], result[1:9:2])
        marks = []
        for x,y in list(zip(result[0:10:2], result[1:11:2])):
            marks.append([x,y])
        marks = np.float32(marks)*np.float32([rface.shape[1], rface.shape[0]])
        ref = landmark_reference*np.float32([rface.shape[1], rface.shape[0]])
        M = cv2.getAffineTransform(marks[0:3], ref[0:3])
        dst = cv2.warpAffine(rface, M, (rface.shape[1], rface.shape[0]))
    return dst



def face_match(face1, face2):

    facenorm1 = face_normalize(face1)
    facenorm2 = face_normalize(face2)
    # cv2.imshow('image1',facenorm1)
    # cv2.waitKey(0)
    # cv2.imshow('image2', facenorm2)
    # cv2.waitKey(0)
    pipeline = dai.Pipeline()
    pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)
    face1_id = pipeline.createNeuralNetwork()
    face2_id = pipeline.createNeuralNetwork()
    face1_id.setBlobPath(blobconverter.from_zoo('face-reidentification-retail-0095', shaves=6))
    face2_id.setBlobPath(blobconverter.from_zoo('face-reidentification-retail-0095', shaves=6))

    xin_face1 = pipeline.createXLinkIn()
    xin_face1.setStreamName('face1_in')
    xin_face1.out.link(face1_id.input)
    xin_face2 = pipeline.createXLinkIn()
    xin_face2.setStreamName('face2_in')
    xin_face2.out.link(face2_id.input)

    xout_face1 = pipeline.createXLinkOut()
    xout_face1.setStreamName("face1_out")
    face1_id.out.link(xout_face1.input)
    xout_face2 = pipeline.createXLinkOut()
    xout_face2.setStreamName("face2_out")
    face2_id.out.link(xout_face2.input)
    conf = 0
    with dai.Device(pipeline) as device:
        face1_in = device.getInputQueue('face1_in')
        face2_in = device.getInputQueue('face2_in')

        face1_data = dai.NNData()
        face1_data.setLayer("0", to_planar(facenorm1, (128, 128)))
        face1_in.send(face1_data)
        face2_data = dai.NNData()
        face2_data.setLayer("0", to_planar(facenorm2, (128, 128)))
        face2_in.send(face2_data)

        face1_nn = device.getOutputQueue('face1_out', maxSize=1)
        face2_nn = device.getOutputQueue('face2_out', maxSize=1)

        result_face1 = face1_nn.get().getFirstLayerFp16()
        result_face2 = face2_nn.get().getFirstLayerFp16()
        conf = 1 - distance.cosine(result_face1, result_face2)
        print(conf.__round__(4))
        return conf



# TODO If class = vehicle (or within vehicle), verify:
#   License plate
#   Make
#   Model

if __name__ == '__main__':
    face_match('X:/Programming/ETI projects/unclassed-output/tmpq33a7z1x_cut.png', 'X:/Programming/ETI projects/unclassed-output/tmp80v6wswt_cut.png')