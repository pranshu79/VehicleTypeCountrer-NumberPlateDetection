import cv2
import numpy as np

net = cv2.dnn.readNet("yolov3.weights", "yolov3-320.cfg")
classes = []

with open("vehicle_name", 'r') as f:
    classes = f.read().splitlines()

#print(classes)
offset = 8

detect = []
cap = cv2.VideoCapture("//Users//pranshukedia//Downloads//video3.mp4")
#img = cv2.imread("//Users//pranshukedia//object_detection//image//img.png")
#print(img)
count = 1
while True:
    _, img = cap.read()
    height, width = img.shape[:2]

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)

#for b in blob:
    #for n, img_blob in enumerate(b):
       # cv2.imshow(str(n), img_blob)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    #layerOutputs = net.forward(outs, getOutputsNames(this->net))
    boxes = []
    confidences = []
    class_ids = []
    #print(output_layers_names)

    tem = "person"
    for output in layerOutputs:
        for detection in output:
            #print(detection)
            scores = detection[5:]
            class_id = np.argmax(scores)

            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)


                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.FONT_HERSHEY_PLAIN
    colors = np.random.uniform(0, 255, size=(len(boxes), 3))

    if len(indexes) > 0:

        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            x1 = x + int(w/2)
            y1 = y + int(h/2)
            centre = x1, y1
            detect.append(centre)

            for (x, y) in detect:
                if y < (300 + offset) and y > (300 - offset):
                    count += 1
                    cv2.line(img, (0, 600), (1000, 600), (127, 255, 0), 2)

                detect.remove((x, y))

            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.circle(img, centre, 4, (127, 255, 0), 1)

            cv2.putText(img, "No. of Vehicle =  " + str(count), (40, 40), font, 3, (0, 127, 255), 2)
            cv2.putText(img, label + " " + confidence + " " + str(i), (x, y+20), font, 2, (255, 0, 0), 2)
            cv2.line(img, (0, 600), (1000, 600), (0, 255, 0), 2)
            #cv2.putText(img, "No. of types of vehicle  " + str(count), (40, 40), font, 2, (0, 255, 0), 2)

    h, w = img.shape[:2]
    img2 = cv2.resize(img, (1000, int(1000 * h / w)))

    print(count)

    cv2.imshow("Image", img2)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllwindows()
