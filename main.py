import cv2
import numpy as np

cap = cv2.VideoCapture(0)
whT = 320

classesFile = 'vehicle_name'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
#print(classNames)

modelConfiguration = 'yolo3-320.cfg'
modelWeights = 'yolov3.weights'

net = cv2.dnn.readNetFromDarknet(modelConfiguration,modelWeights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img,1/255,(whT,whT),[0,0,0],1,crop=False)
    net.setInput(blob)

    layerNames = net.getLayerNames()
    outputNames = [layerNames[i[0]-1] for i in net.getUnconnectedOutLayers()]
    print(net.getUnconnectedOutLayers())
    print(outputNames)
    print(net.getUnconnectedOutLayers())


    cv2.imshow('cam', img)
    #if cv2.waitKey(1) & 0xFF == ord('q'):
        #break
    cv2.waitKey(0)
