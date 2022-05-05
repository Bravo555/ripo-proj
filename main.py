#!/usr/bin/python

import cv2
import matplotlib.pyplot as plt

modelpath = 'ssd_mobilenet_v3_large_coco_2020_01_14/frozen_inference_graph.pb'
configpath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'

image = cv2.imread('traffic.jpg')
image = cv2.resize(image, (800, 600))

rows = image.shape[0]
cols = image.shape[1]

net = cv2.dnn_DetectionModel(modelpath, configpath)

half = 255.0 / 2.0

net.setInputSize(320, 320)
net.setInputScale(1.0 / half)
net.setInputMean((half, half, half))
net.setInputSwapRB(True)

class_ids, confidences, boxes = net.detect(image)

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

# for detection in out[0, 0, :, :]:
#     score = float(detection[2])
#     if score > 0.3:
#         left = detection[3] * cols
#         top = detection[4] * rows
#         right = detection[5] * cols
#         bottom = detection[6] * rows
#         cv2.rectangle(image, (int(left), int(top)), (int(right),
#                                                      int(bottom)), (23, 230,
#                                                      210), thickness=2)

for classid, box in zip(class_ids, boxes):
    cv2.rectangle(image, box, (255, 0, 0), thickness=2)
    cv2.putText(image, labels[classid-1], box[:2],
                cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0))

cv2.imshow('RiPO', image)
cv2.waitKey()
