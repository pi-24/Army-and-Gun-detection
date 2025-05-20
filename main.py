import os
from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# Paths
image_folder = 'images/'
normal_folder = 'normal/'
alert_folder = 'alert/'
os.makedirs(normal_folder, exist_ok=True)
os.makedirs(alert_folder, exist_ok=True)

# Load models
gun_model = YOLO('best_m_gun.pt')          # Detects guns (class 0 = gun)
uniform_model = YOLO('/home/proeffico/Desktop/private/runs2/detect/train/weights/best.pt')      # Detects uniform (class 0) and normal dress (class 1)

# Helper: Calculate IoU
def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return 0.0
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea)

# Draw bounding boxes
def draw_boxes(image, detections, label, color=(0, 255, 0)):
    for det in detections:
        x1, y1, x2, y2 = map(int, det[:4])
        conf = det[4]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, f'{label} {conf:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

# Process images
for img_file in Path(image_folder).glob('*.*'):
    img = cv2.imread(str(img_file))
    if img is None:
        continue

    # Detect guns
    gun_results = gun_model(img)[0]
    gun_dets = [box.cpu().numpy() for box in gun_results.boxes.data if int(box[5]) == 0]

    if not gun_dets:
        continue  # No gun → skip

    # Detect people (uniformed and normal dress)
    uniform_results = uniform_model(img)[0]
    uniform_dets = [box.cpu().numpy() for box in uniform_results.boxes.data if int(box[5]) == 0]  # class 0 = uniform
    normal_dets = [box.cpu().numpy() for box in uniform_results.boxes.data if int(box[5]) == 1]   # class 1 = normal dress

    # Match guns to normal-dressed people
    alert_flag = False
    overlapped_normals = []  # Store only normal-dressed people overlapped with a gun
    for gun in gun_dets:
        gun_box = gun[:4]
        for person in normal_dets:
            person_box = person[:4]
            iou = calculate_iou(gun_box, person_box)
            if iou > 0.01:
                alert_flag = True
                overlapped_normals.append(person)
        # No break here: collect all overlapped normals for all guns

    # Save image accordingly
    if alert_flag:
        img_annotated = img.copy()
        draw_boxes(img_annotated, gun_dets, label='Gun', color=(0, 0, 255))           # Red
        draw_boxes(img_annotated, overlapped_normals, label='Militant', color=(0, 255, 255))   # Yellow
        cv2.imwrite(os.path.join(alert_folder, img_file.name), img_annotated)
    else:
        img_annotated = img.copy()
        draw_boxes(img_annotated, gun_dets, label='Gun', color=(0, 0, 255))           # Red
        draw_boxes(img_annotated, uniform_dets, label='Uniform', color=(0, 255, 0))   # Green
        cv2.imwrite(os.path.join(normal_folder, img_file.name), img_annotated)
