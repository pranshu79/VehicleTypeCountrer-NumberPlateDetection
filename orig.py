import cv2
import numpy as np

net = cv2.dnn.readNet('yolov3.weights', 'yolov3-320.cfg')
classes = []
plateCascade = cv2.CascadeClassifier('haarcascade_russian_plate_number')
with open('vehicle_name', 'r') as f:
    classes = f.read().splitlines()

offset = 6
minArea = 500
detect = []
cap = cv2.VideoCapture("//Users//pranshukedia//Downloads//video3.mp4")

count = 0
car = 0
motorbike = 0
bus = 0
truck = 0
bicycle = 0

frame_change = 0
pre_data = []
while True:
    _, img = cap.read()
    height, width = img.shape[:2]

    frame_change = 0
    if cv2.waitKey(5) & 0xff == ord('x'):
        continue

    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    numberPlates = plateCascade.detectMultiScale(imgGray, 1.1, 4)

    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img, "LicensePlate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)

    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)
    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
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

            if y1 < (int(height * 2 / 3) + offset + 50) and y1 > (int(height * 2 / 3) - offset + 50):
                if label == "car":
                    car += 1
                elif label == "bicycle":
                    bicycle += 1
                elif label == "motorbike":
                    motorbike += 1
                elif label == "truck":
                    truck += 1
                elif label == "bus":
                    bus += 1

                for (x3, w3, h3) in pre_data:
                    if abs(x3 - x1) < 30 and abs(w3 - w) < 25 and abs(h3 - h) < 25:
                        if label == "car":
                            car -= 1
                        elif label == "bicycle":
                            bicycle -= 1
                        elif label == "motorbike":
                            motorbike -= 1
                        elif label == "truck":
                            truck -= 1
                        elif label == "bus":
                            bus -= 1

                cv2.line(img, (0, int(height * 2 / 3) + 50), (width, int(height * 2 / 3) + 50), (127, 255, 0), 2)

            if frame_change == 0:
                frame_change += 1
                pre_data.clear()
                if y1 < (int(height * 2 / 3) + offset + 50) and y1 > (int(height * 2 / 3) - offset + 50):
                    pre_data.append((x1, w, h))


            confidence = str(round(confidences[i], 2))
            color = colors[i]
            cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
            cv2.circle(img, centre, 4, (127, 255, 0), 1)

            count = bus + truck + bicycle + car + motorbike
            cv2.putText(img, "Detected Vehicle =  " + str(count), (30, 35), font, 2, (0, 127, 255), 2)
            cv2.putText(img, "Car =  " + str(car), (30, 58), font, 2, (0, 127, 255), 2)
            cv2.putText(img, "Motorbike =  " + str(motorbike), (30, 81), font, 2, (0, 127, 255), 2)
            cv2.putText(img, "Bus =  " + str(bus), (30, 104), font, 2, (0, 127, 255), 2)
            cv2.putText(img, "Truck =  " + str(truck), (30, 127), font, 2, (0, 127, 255), 2)
            cv2.putText(img, "Bicycle =  " + str(bicycle), (30, 150), font, 2, (0, 127, 255), 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 2, (255, 0, 0), 2)
            cv2.line(img, (0, int(height*2/3) + 50), (width, int(height*2/3) + 50), (0, 0, 255), 2)
            #cv2.putText(img, "No. of types of vehicle  " + str(count), (40, 40), font, 2, (0, 255, 0), 2)

    h, w = img.shape[:2]
    img2 = cv2.resize(img, (1000, int(1000 * h / w)))
    cv2.imshow("Image", img2)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllwindows()
