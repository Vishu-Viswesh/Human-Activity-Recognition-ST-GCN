import cv2
import mediapipe as mp
import numpy as np
import torch
from ultralytics import YOLO  # pip install ultralytics
from model import STGCN

# Configuration
WINDOW_SIZE = 30
NUM_CLASSES = 5
LABELS = ["Walking", "Running", "Sitting", "Standing", "Waving"]

# Initialize Models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
yolo_model = YOLO('yolov8n.pt')  # Uses standard object detection weights
stgcn_model = STGCN(in_channels=3, num_classes=NUM_CLASSES).to(device)
# stgcn_model.load_state_dict(torch.load('best_stgcn_weights.pth', map_location=device))
stgcn_model.eval()

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

sequence_buffer = []
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    
    # Step 1: Run YOLO detection to find people (Class 0 in COCO is person)
    yolo_results = yolo_model(frame, verbose=False)[0]
    boxes = yolo_results.boxes.data.cpu().numpy()
    
    # Filter boxes for 'person' with a solid confidence score
    person_boxes = [box for box in boxes if int(box[5]) == 0 and box[4] > 0.5]

    frame_landmarks = np.zeros((33, 3))

    if len(person_boxes) > 0:
        # Focus on the most prominent person found (largest bounding box area)
        main_person = max(person_boxes, key=lambda b: (b[2]-b[0]) * (b[3]-b[1]))
        x1, y1, x2, y2 = map(int, main_person[:4])
        
        # Add a slight padding margin to the crop area safely bounded by frame walls
        pad = 20
        crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
        crop_x2, crop_y2 = min(w, x2 + pad), min(h, y2 + pad)
        
        crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
        
        if crop.size > 0:
            # Step 2: Extract keypoints exclusively from the isolated bounding box
            rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            mp_results = pose.process(rgb_crop)
            
            if mp_results.pose_landmarks:
                # Map relative crop landmarks back to absolute master screen space coordinates
                for idx, lm in enumerate(mp_results.pose_landmarks.landmark):
                    abs_x = (lm.x * (crop_x2 - crop_x1) + crop_x1) / w
                    abs_y = (lm.y * (crop_y2 - crop_y1) + crop_y1) / h
                    frame_landmarks[idx] = [abs_x, abs_y, lm.visibility]
                
                # Draw local bounding framework on screen
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                mp_drawing.draw_landmarks(frame, mp_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    sequence_buffer.append(frame_landmarks)
    if len(sequence_buffer) > WINDOW_SIZE:
        sequence_buffer.pop(0)

    # Step 3: Run ST-GCN classification over the rolling window
    if len(sequence_buffer) == WINDOW_SIZE:
        data = np.array(sequence_buffer)
        data = np.transpose(data, (2, 0, 1)) # Shape: (3, 30, 33)
        data_tensor = torch.tensor(data, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = stgcn_model(data_tensor)
            prediction = torch.argmax(output, dim=1).item()
            confidence = torch.softmax(output, dim=1)[0][prediction].item()
        
        cv2.putText(frame, f"Action: {LABELS[prediction]} ({confidence*100:.1f}%)", 
                    (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.imshow('YOLO + ST-GCN Activity Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
