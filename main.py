#!/usr/bin/python

import sys
import cv2
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
    print('usage: ./main.py VIDEO_FILE')
    sys.exit(1)

modelpath = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

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

while cap.isOpened():
    ret, frame = cap.read()
    frame = cv2.resize(frame, (800, 600))

    if ret != True:
        break

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    class_ids, confidences, boxes = net.detect(frame)
    for classid, confidence, box in zip(class_ids, confidences, boxes):
        if classid == 10:
            if confidence >= 0.7:
                print(confidence)
                cv2.rectangle(frame, box, (255, 0, 0), thickness=2)
                text = labels[classid-1]
                cv2.putText(frame, text, box[:2],
                            cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

    cv2.imshow(video_filename, frame)
