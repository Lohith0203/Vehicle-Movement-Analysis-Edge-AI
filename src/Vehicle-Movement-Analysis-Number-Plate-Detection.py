import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import urllib.request

# URLs to YOLOv3 files
weights_url = 'https://pjreddie.com/media/files/yolov3.weights'
cfg_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg'
names_url = 'https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names'

# Download paths
weights_path = 'yolov3.weights'
cfg_path = 'yolov3.cfg'
names_path = 'coco.names'

# Download the files if they do not exist
if not os.path.exists(weights_path):
    urllib.request.urlretrieve(weights_url, weights_path)
if not os.path.exists(cfg_path):
    urllib.request.urlretrieve(cfg_url, cfg_path)
if not os.path.exists(names_path):
    urllib.request.urlretrieve(names_url, names_path)

print("YOLOv3 weights, config, and coco names have been downloaded.")

# Load YOLO
net = cv2.dnn.readNet(weights_path, cfg_path)
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open(names_path, "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Define the path to your dataset directory
dataset_path = 'D:\IntelProject//archive\dataset'  # Adjust this to the path where your images and XMLs are stored

# Function to parse XML annotations
def parse_annotations(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall('object'):
        name = obj.find('name').text
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        bboxes.append((name, b))
    return bboxes

# Process each image and corresponding XML
for file in os.listdir(dataset_path):
    if file.endswith(".jpg"):
        image_path = os.path.join(dataset_path, file)
        xml_path = os.path.join(dataset_path, file.replace(".jpg", ".xml"))

        # Load image
        img = cv2.imread(image_path)
        height, width, channels = img.shape

        # Parse annotations
        bboxes = parse_annotations(xml_path)

        # Detecting objects with YOLO
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # Showing information on the screen
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
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                cv2.putText(img, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw ground truth boxes
        for name, (xmin, ymin, xmax, ymax) in bboxes:
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)
            cv2.putText(img, name, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
