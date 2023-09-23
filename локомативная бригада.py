import cv2
import numpy as np
import time
import os


def detect_objects(frame, net, out_layers, classes_to_look_for):
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

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.5)

    detected_objects = []
    for i in indices:
        i = i[0]
        x, y, w, h = boxes[i]
        label = str(classes[class_ids[i]])
        detected_objects.append(label)
        confidence = round(confidences[i], 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f'{label} {confidence}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
    return frame, detected_objects

yolo_config = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/yolov3-tiny.cfg"
yolo_weights = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/yolov3.weights"
net = cv2.dnn.readNet(yolo_weights, yolo_config)

class_file = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/coco.txt"
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()
output_layers_indices = np.reshape(output_layers_indices, (output_layers_indices.size,))
out_layers = [layer_names[i - 1] for i in output_layers_indices]

classes_to_look_for = ["person", "cell phone", "laptop", "tv", "remote"]

video_path = "/Users/kakotichi/Downloads/train_dataset_Бригады/Анализ бригад (телефон)/Есть телефон/02_59_18.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')


class_file = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/coco.txt"
with open(class_file, 'r') as f:
    classes = [line.strip() for line in f.readlines()]


output_folder = "/Users/kakotichi/Documents/GitHub/sf_data_science/Vacation/result"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)
output_video_path = os.path.join(output_folder, '00_13_48.mp4')

out = cv2.VideoWriter(output_video_path, fourcc, 20.0, (frame_width,frame_height))

phone_in_hands = False
phone_detection_timer = None
phone_hold_time = 3.0

while cap.isOpened():
    ret, frame = cap.read()

    if ret:
        processed_frame, detected_objects = detect_objects(frame, net, out_layers, classes_to_look_for)

        if 'cell phone' in detected_objects:
            if not phone_in_hands:
                phone_detection_timer = time.time()
            phone_in_hands = True
        else:
            phone_in_hands = False

        person_in_cabin = 'person' in detected_objects

        if person_in_cabin:
            if phone_in_hands and (time.time() - phone_detection_timer) >= phone_hold_time:
                print("Телефон был в руках более 3-х секунд.")
                # Вставьте ваш код здесь, если хотите предпринять дополнительные действия после 3-х секунд
        else:
            print("Никого нет в кабине.")

        cv2.imshow('Detected Objects', processed_frame)
        out.write(processed_frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
