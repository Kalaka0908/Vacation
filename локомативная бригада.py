import cv2
import numpy as np
import time
import os
from collections import Counter
import logging
import tkinter as tk

def detect_objects(frame, net, out_layers, classes_to_look_for):
    detected_objects = []
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    layer_outputs = net.forward(out_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.4 and classes[class_id] in classes_to_look_for:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.4, nms_threshold=0.5)
    
    if isinstance(indices, tuple) and not indices:
         indices = []
    elif isinstance(indices, np.ndarray):
        indices = indices.astype(int)
        indices = indices.reshape(-1, 1).tolist()
    else:
        raise ValueError(f"Unexpected value in 'indices': {indices}")

    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        detected_objects.append(label)
        confidence = round(confidences[i], 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  
    return frame, detected_objects
yolo_config = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/модель/yolov3.cfg"
yolo_weights = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/модель/yolov3.weights"
net = cv2.dnn.readNet(yolo_weights, yolo_config)

class_file = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/data/coco.txt"
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers_indices = np.reshape(output_layers_indices, (output_layers_indices.size,))
out_layers = [layer_names[i - 1] for i in output_layers_indices]

classes_to_look_for = ["person", "cell phone", "laptop", "tv", "remote"]
# Настройка логирования
logging.basicConfig(filename='results.log', level=logging.INFO)

def load_video():
    return "/Users/kakotichi/Downloads/train_dataset_Бригады/Анализ бригад (телефон)/Есть телефон/02_07_39.mp4"

video_path = load_video()  # video_path теперь определён до его первого использования
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


class_file = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/data/coco.txt"
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


output_folder = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/result"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_video_path = os.path.join(output_folder, 'результат.mp4')

out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width,frame_height))

phone_in_hands = False
phone_detection_timer = None
phone_hold_time = 3.0

events = []  # список для сохранения всех событий

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        processed_frame, detected_objects = detect_objects(frame, net, out_layers, classes_to_look_for)
        detected_objects_count = Counter(detected_objects)

        detected_phone_count = detected_objects_count.get('cell phone', 0)
        detected_person_count = detected_objects_count.get('person', 0)

        if detected_phone_count > 0:
            if not phone_in_hands:
                phone_detection_timer = time.time()
                logging.info('Обнаружено событие: начало использования телефона')
            phone_in_hands = True
        else:
            if phone_in_hands:
                 logging.info('Обнаружено событие: окончание использования телефона')
            phone_in_hands = False


        if detected_person_count >= 2:
            if phone_in_hands and (time.time() - phone_detection_timer) >= phone_hold_time:
                event_msg = f"Телефон(ы) был(и) в руках двух человек более {phone_hold_time} секунд."
                events.append(event_msg)
        elif detected_person_count == 1:
            if phone_in_hands and (time.time() - phone_detection_timer) >= phone_hold_time:
                event_msg = "Телефон был в руках одного человека более 3-х секунд."
                events.append(event_msg)
       
        cv2.imshow('Detected Objects', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()

print("\nОбработка видео завершена. Обнаруженные события:")
for idx, event in enumerate(events, start=1):
    logging.info(f"Обнаружено событие {idx}: {event}")
    print(f"{idx}. {event}")