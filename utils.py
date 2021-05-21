import cv2
import easyocr
import pickle
import numpy as np

def extract_text(filepath):
    reader = easyocr.Reader(['en'])
    img = cv2.imread(filepath)
    text = reader.readtext(img, detail=0)
    text = ' '.join(text)
    return text

def detect_object(filepath):
    with open('file.pkb', 'rb') as f:
        labels_to_names = pickle.load(f)

    img = cv2.imread(filepath)

    net = cv2.dnn.readNet("yolov4.weights", "yolov4.cfg")
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0,0,0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    height, width, channels = img.shape

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
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

                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    objects = []

    for i in range(len(boxes)):
        if i in indexes:
            label = str(labels_to_names[class_ids[i]])
            confidence = confidences[i]
            objects.append((label, confidence))

    return objects
