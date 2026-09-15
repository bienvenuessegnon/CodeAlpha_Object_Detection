# CodeAlpha_Object_Detection
👁️ CodeAlpha — Object Detection and Tracking

«Real-time object detection, tracking, and counting using YOLO and OpenCV.»

"Python" (https://img.shields.io/badge/Python-3.x-blue?logo=python)
"OpenCV" (https://img.shields.io/badge/OpenCV-Computer%20Vision-green?logo=opencv)
"YOLO" (https://img.shields.io/badge/YOLO-Object%20Detection-orange)
"CodeAlpha" (https://img.shields.io/badge/CodeAlpha-AI%20Internship-purple)

---

📌 Project Overview

This project was developed as part of the CodeAlpha Artificial Intelligence Internship.

The objective is to build a real-time Object Detection and Tracking system capable of detecting objects in a video stream, assigning tracking IDs, and counting detected objects.

The system uses YOLO for object detection and tracking, combined with OpenCV for video processing and real-time visualization.

---

🎯 Objectives

The project implements the main requirements of the CodeAlpha Object Detection and Tracking task:

- 🎥 Process real-time video from a webcam or video file.
- 👁️ Detect objects using a pre-trained YOLO model.
- 📦 Draw bounding boxes around detected objects.
- 🏷️ Display object class labels.
- 🆔 Assign tracking IDs to detected objects.
- 🔢 Count objects currently visible in the frame.
- 📊 Count unique objects tracked during the session.
- ⚡ Display the results in real time.

---

🧠 Technologies Used

Technology| Purpose
Python| Main programming language
YOLO| Object detection and tracking
OpenCV| Video processing and visualization
Ultralytics| YOLO implementation
Computer Vision| Object detection and tracking

---

📂 Project Structure

CodeAlpha_Object_Detection/
│
├── object_detection.py
├── requirements.txt
├── README.md
└── output/

"object_detection.py"

Contains the complete object detection, tracking, counting, and visualization pipeline.

"requirements.txt"

Contains the Python dependencies required to run the project.

"output/"

Can be used to store generated output videos or screenshots.

---

⚙️ Installation

1. Clone the repository

git clone https://github.com/bienvenuessegnon/CodeAlpha_Object_Detection.git

2. Navigate to the project directory

cd CodeAlpha_Object_Detection

3. Install dependencies

pip install -r requirements.txt

---

📦 Requirements

The project requires:

ultralytics
opencv-python

You can also install them manually:

pip install ultralytics opencv-python

---

▶️ Run the Project

Start the application with:

python object_detection.py

By default, the program uses the computer's webcam.

video_source = 0

To use a video file instead, change it to:

video_source = "video.mp4"

---

🔍 How It Works

The application follows this pipeline:

┌─────────────────────┐
│   Webcam / Video    │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│       OpenCV        │
│   Read Video Frame  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│        YOLO         │
│  Object Detection   │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│      Tracking       │
│   Tracking IDs      │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Counting & Labels  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Real-Time Display  │
└─────────────────────┘

---

🎯 Object Detection

YOLO analyzes every video frame and identifies objects using a pre-trained model.

For every detected object, the system can obtain:

- Object class
- Confidence score
- Bounding box
- Tracking ID

Example:

Object: person
ID: 3
Confidence: 0.91

---

🆔 Object Tracking

Detection alone identifies objects in individual frames.

Tracking goes one step further by assigning an ID to detected objects and attempting to maintain that identity across consecutive frames.

Example:

Frame 1          Frame 2          Frame 3

Person           Person           Person
ID: 1      →     ID: 1      →     ID: 1

This allows the system to follow an object as it moves through the video.

---

🔢 Object Counting

The application provides two types of counts.

Objects in the current frame

This represents the number of objects currently detected.

Objects in frame: 3

Total unique objects tracked

This represents the number of unique tracking IDs detected during the session.

Total tracked: 7

For example, if seven different people pass in front of the camera one after another:

Objects in frame: 1
Total tracked: 7

---

📊 Real-Time Information

The video window displays information such as:

Objects in frame: 2
Total tracked: 5

Bounding boxes and labels are also displayed around detected objects.

---

💻 Main Implementation

The YOLO model is loaded using Ultralytics:

from ultralytics import YOLO

model = YOLO("yolo11n.pt")

Tracking is performed using:

results = model.track(
    frame,
    persist=True,
    conf=0.5,
    verbose=False
)

The "persist=True" option allows tracking information to be maintained across consecutive frames.

---

🛠️ Configuration

The confidence threshold can be adjusted:

conf=0.5

A higher value can reduce low-confidence detections, while a lower value can allow more detections.

The video source can also be changed:

video_source = 0

For a video file:

video_source = "video.mp4"

---

✨ Key Features

- ✅ Real-time object detection
- ✅ YOLO-based detection
- ✅ Object tracking
- ✅ Tracking IDs
- ✅ Bounding boxes
- ✅ Object labels
- ✅ Confidence scores
- ✅ Current object counting
- ✅ Unique object tracking count
- ✅ Webcam support
- ✅ Video file support
- ✅ Real-time visualization

---

🚀 Possible Improvements

Future versions could include:

- 📈 FPS monitoring
- 👥 Person-specific counting
- 🚶 Entry and exit counting
- 📹 Automatic output video recording
- 🗺️ Region-based counting
- 🚧 Line-crossing detection
- 📊 Detection statistics
- 🧠 Custom-trained YOLO models
- 🔄 Advanced tracking algorithms such as ByteTrack or Deep SORT
- 🌐 Web-based interface

---

🎥 Demo

A demonstration video or screenshots can be added here after testing the project.

Example:

Input Video
     ↓
YOLO Detection
     ↓
Tracking IDs
     ↓
Object Counting
     ↓
Real-Time Visualization

---

🎓 CodeAlpha Internship

Program: CodeAlpha Artificial Intelligence Internship

Task: Task 4 — Object Detection and Tracking

This project was developed as part of the CodeAlpha Artificial Intelligence Internship.

---

👨‍💻 Author

Bienvenu Essegnon

«Aspiring AI Engineer | Data Science Enthusiast | Full-Stack Developer»

GitHub: "@bienvenuessegnon" (https://github.com/bienvenuessegnon)

---

📄 License

This project was developed for educational purposes as part of the CodeAlpha Artificial Intelligence Internship.