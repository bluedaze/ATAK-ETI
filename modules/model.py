# Functions for the purpose of loading and running models
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
import pandas as pd
import copy
import json
import pathlib
import datetime as dt

def to_planar(arr: np.ndarray, shape: tuple) -> list:
    return [val for channel in cv2.resize(arr, shape).transpose(2, 0, 1) for y_col in channel for val in y_col]

def frame_norm(frame, bbox):
    normVals = np.full(len(bbox), frame.shape[0])
    normVals[::2] = frame.shape[1]
    return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

def get_label_map(directory: str):
    d = os.path.realpath(directory)

# Current detection features:
# Timestamp
# Class
# Confidence
# Camera Serial
def det_to_json(db, **kwargs):
    current_id = len(os.listdir(f"{os.getcwd()}/vectors"))
    df = pd.DataFrame()
    for i in kwargs['dets']:
        detection = {}
        for feature in kwargs.keys():
            detection[feature] = kwargs[feature]
        row = pd.Series(data=detection)
        row = row.drop('dets')
        df[i] = row
        if kwargs['label'] == 'person':
            with open(f'{os.getcwd()}/vectors/id_{current_id}.json',"w") as outfile:
                outfile.write(row.to_json())
            db['person'].insert_one(row.to_dict())
        if kwargs['label'] == 'vehicle':
            with open(f'{os.getcwd()}/vectors/{kwargs["license"]}.json',"w") as outfile:
                outfile.write(row.to_json())
            db['vehicle'].insert_one(row.to_dict())
    out_json = df.to_json(orient='split')
    with open(f'{os.getcwd()}/{kwargs["camera"]}_{kwargs["time"]}.json', "w") as outfile:
        outfile.write(out_json)


