import cv2
import numpy as np
from gtts import gTTS
import os
import time

st_t = time.time()


language="en"
# Load Yolo
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# Loading image
cap=cv2.VideoCapture(0)

#img = cv2.imread("test.jpeg")
while True:
    _,img=cap.read()
    img = cv2.resize(img, None, fx=1, fy=1)
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    font = cv2.QT_NEW_BUTTONBAR
    st = ""
    l = []
    for i in range(len(boxes)):


        if i in indexes:

            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]
            #conf=(confidences[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
            cv2.putText(img, label, (x, y + 30), font, 2, (0,255,0), 3)
            '''end_t = time.time()
            exeTime = round(end_t - st_t, 2)
            t = str((exeTime))
            cv2.putText(img, t, (x+400, y + 30), font, 1, (0, 255, 0), 3)'''
            #cv2.putText(img, str(conf), (x, y + h), font, 3, color, 3)

            l.append(label)
            #print(*l,end="")
    for xy in l:
            st=st+" "+xy


    #s=""

    #for obj in l:
            #s=s+" "+obj
            #print(s)
    if len(l)!=0:
        speech=gTTS(text=st,lang=language,slow="False")
        speech.save("detect.mp3")
        os.system("detect.mp3")
        #os.kill("detect.mp3")
    cv2.imshow("Image", img)

    key = cv2.waitKey(1)
    if key == 27:
       break

cap.release()
cv2.destroyAllWindows()

