#!/usr/bin/python

import sys
from unittest import skip
import cv2
import matplotlib.pyplot as plt

# if len(sys.argv) < 2:
#     print('usage: ./main.py VIDEO_FILE')
#     sys.exit(1)

modelpath = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

video_filename = 0
if len(sys.argv) == 2:
    video_filename = sys.argv[1]
cap = cv2.VideoCapture(video_filename)

net = cv2.dnn_DetectionModel(modelpath, configpath)

half = 255.0 / 2.0

net.setInputSize(320, 320)
net.setInputScale(1.0 / half)
net.setInputMean((half, half, half))
net.setInputSwapRB(True)

labels = [
    'person',
    'bicycle',
    'car',
    'motorbike',
    'aeroplane',
    'bus',
    'train',
    'truck',
    'boat',
    'traffic light',
]

found: any
foundText: str
framesLeft = 0
skipFrames = 1
counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))

    if ret != True:
        break

    key = cv2.waitKey(1)

    if key == ord('s'):
        key = None
        while key not in [ord('q'), ord('s')]:
            key = cv2.waitKey(0)

    if key == ord('q'):
        break

    if key == ord('i'):
        if (skipFrames < 10):
            skipFrames += 1

    if key == ord('d'):
        if (skipFrames > 1):
            skipFrames -= 1

    if key == ord('r'):
        skipFrames = 1

    if counter % skipFrames == 0:
        class_ids, confidences, boxes = net.detect(frame)
        for classid, confidence, box in zip(class_ids, confidences, boxes):
            if classid == 10:
                if confidence >= 0.65:
                    found = box
                    framesLeft = 30 / skipFrames
                    text = labels[classid-1]
                    foundText = text
            if (framesLeft > 0):
                cv2.rectangle(frame, found, (255, 0, 0), thickness=2)
                cv2.putText(frame, foundText, found[:2],
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))
                framesLeft -= 1

        cv2.imshow(video_filename if video_filename != 0 else "Kamera", frame)
    counter += 1
