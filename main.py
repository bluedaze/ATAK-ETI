#!/usr/bin/env python
from scripts.post_process_lateral import clarify
import argparse
from importlib import import_module
import os
from flask import Flask, render_template, Response, request


script = ''

parser = argparse.ArgumentParser()

parser.add_argument('-p', '--person', action='store_true', help='Run the person tracking protocol')
parser.add_argument('-v', '--vehicle', action='store_true', help='Run the vehicle tracking protocol')
parser.add_argument('-c', '--continuous', action='store_true', help='Disable post-processing')

args = parser.parse_args()
args.person = True

# Imports done from arguments to prevent code execution unless flag was specifically chosen.
# If code is imported outside of a conditional clause then it will automatically execute.
# This is quick and dirty so this should change.
if args.person:
        from scripts.body_profile import person_identification
        pi = person_identification()
        clarify(label='person')
elif args.vehicle:
        from scripts.vehicle_profile import vehicle_identification
        vp = vehicle_identification()
        clarify(label='vehicle')
