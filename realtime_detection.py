import cv2
import torch
from ultralytics import YOLO

# Load trained model
model = YOLO("runs/detect/train6/weights/best.pt")

# Open webcam
cap = cv2.VideoCapture(0)

# Approximate real human height (in meters)
REAL_HEIGHT = 1.7

# Approximate focal length (adjust if needed)
FOCAL_LENGTH = 800

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    for r in results:
        boxes = r.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            conf = float(box.conf[0])

            # Bounding box height in pixels
            box_height = y2 - y1

            if box_height > 0:
                distance = (REAL_HEIGHT * FOCAL_LENGTH) / box_height
            else:
                distance = 0

            # Draw rectangle
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)

            label = f"Victim {conf:.2f} | {distance:.2f}m"
            cv2.putText(frame, label, (int(x1), int(y1)-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Disaster Victim Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
