AI-Based Real-Time Victim Detection and Distance Estimation for Disaster Response using YOLOv8

## Project Overview
This project uses Artificial Intelligence and Computer Vision to detect victims during disaster situations and assist rescue teams.

It uses a custom-trained YOLOv8 model to identify victims in real-time video streams.

## Features
- Real-time victim detection using YOLOv8
- Custom dataset-based model training
- Confidence score display
- Approximate victim distance estimation
- IP camera integration using a smartphone
- Automatic latitude and longitude display (IP-based location)
- Audio alarm when a victim is detected

## Technologies Used
- Python
- YOLOv8
- PyTorch
- OpenCV
- CUDA
- NVIDIA RTX 3050 GPU
- Pygame
- DroidCam / IP Camera

## Project Structure
```text
AI-Disaster-Victim-Detection/
│
├── rescue_system.py
├── weights/
│   └── best.pt
├── alarm.wav
├── README.md
└── .gitignore
```

## Installation

Install the required packages:

```bash
pip install ultralytics opencv-python requests pygame
```

## Run the Project

```bash
python rescue_system.py
```

Make sure your smartphone camera stream is connected and the camera URL is configured correctly.

## Important Notes
- Distance estimation is approximate and requires camera calibration.
- IP-based location provides approximate network location, not precise victim GPS.
- This is an academic prototype and should not be relied upon as the sole source for real emergency rescue decisions.

## Future Enhancements
- Accurate GPS integration
- SMS and email alerts
- Automatic victim image capture
- Rescue team dashboard
- Multiple camera integration
-
