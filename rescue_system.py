from ultralytics import YOLO
import cv2
import requests
import pygame
import os

# ----------------------------
# Initialize Alarm System
# ----------------------------
pygame.mixer.init()

# Load alarm sound
if os.path.exists("alarm.wav"):
    pygame.mixer.music.load("alarm.wav")
else:
    print("⚠ alarm.wav not found")

# ----------------------------
# Load YOLO Model
# ----------------------------
model = YOLO("runs/detect/train6/weights/best.pt")

# ----------------------------
# Distance Settings
# ----------------------------
FOCAL_LENGTH = 350
REAL_HEIGHT = 1.7

# ----------------------------
# Get Automatic Location
# ----------------------------
try:
    location_data = requests.get("http://ip-api.com/json").json()
    latitude = location_data['lat']
    longitude = location_data['lon']
except:
    latitude = "Unknown"
    longitude = "Unknown"

# ----------------------------
# Open IP Camera
# ----------------------------
cap = cv2.VideoCapture("http://10.176.184.127:8080/video")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    victim_detected = False  # Reset every frame

    for r in results:
        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])

            if conf > 0.6:
                victim_detected = True

                box_height = y2 - y1
                distance = (REAL_HEIGHT * FOCAL_LENGTH) / box_height if box_height > 0 else 0

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show confidence + distance
                label = f"Victim {conf:.2f} | {distance:.2f}m"
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 255, 0), 2)

                # Show Latitude & Longitude
                location_text = f"Lat: {latitude}  Lon: {longitude}"
                cv2.putText(frame, location_text, (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 0, 255), 2)

    # ----------------------------
    # Alarm Control Logic
    # ----------------------------
    if victim_detected:
        if not pygame.mixer.music.get_busy():
            pygame.mixer.music.play()
    else:
        pygame.mixer.music.stop()

    cv2.imshow("Smart Rescue AI System", frame)

    if cv2.waitKey(1) == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.music.stop()
