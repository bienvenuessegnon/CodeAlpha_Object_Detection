import cv2
from ultralytics import YOLO


# ============================================================
# 1. LOAD YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 2. OPEN VIDEO SOURCE
# ============================================================

# Use 0 for the default webcam
# Replace 0 with a video file path if needed
video_source = 0

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open the video source.")
    exit()


# ============================================================
# 3. OBJECT DETECTION AND TRACKING
# ============================================================

while True:

    success, frame = cap.read()

    if not success:
        print("Video ended or frame could not be read.")
        break

    # Run YOLO tracking
    results = model.track(
        frame,
        persist=True,
        conf=0.5,
        verbose=False
    )

    # Draw detections and tracking information
    annotated_frame = results[0].plot()

    # Display the result
    cv2.imshow(
        "CodeAlpha - Object Detection and Tracking",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 4. RELEASE RESOURCES
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("Object detection and tracking stopped.")