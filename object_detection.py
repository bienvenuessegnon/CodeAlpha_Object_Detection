import cv2
from ultralytics import YOLO


# ============================================================
# 1. LOAD YOLO MODEL
# ============================================================

model = YOLO("yolo11n.pt")


# ============================================================
# 2. VIDEO SOURCE
# ============================================================

# Use 0 for the default webcam.
# Replace 0 with a video file path if needed.
video_source = 0

cap = cv2.VideoCapture(video_source)

if not cap.isOpened():
    print("Error: Could not open the video source.")
    exit()


# ============================================================
# 3. TRACKING AND COUNTING
# ============================================================

# Store IDs that have already been detected
tracked_ids = set()


# ============================================================
# 4. PROCESS VIDEO FRAMES
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

    result = results[0]

    # --------------------------------------------------------
    # Process detected objects
    # --------------------------------------------------------

    if result.boxes is not None:

        boxes = result.boxes

        # Get tracking IDs
        if boxes.id is not None:

            track_ids = boxes.id.int().cpu().tolist()
            class_ids = boxes.cls.int().cpu().tolist()
            confidences = boxes.conf.cpu().tolist()

            for track_id, class_id, confidence in zip(
                track_ids,
                class_ids,
                confidences
            ):

                # Add ID to the set
                tracked_ids.add(track_id)

                # Get object class name
                class_name = model.names[class_id]

                # Display information
                print(
                    f"Object: {class_name} | "
                    f"ID: {track_id} | "
                    f"Confidence: {confidence:.2f}"
                )

    # --------------------------------------------------------
    # Draw bounding boxes and tracking IDs
    # --------------------------------------------------------

    annotated_frame = result.plot()

    # --------------------------------------------------------
    # Display statistics
    # --------------------------------------------------------

    current_objects = 0

    if result.boxes is not None:
        current_objects = len(result.boxes)

    total_tracked = len(tracked_ids)

    cv2.putText(
        annotated_frame,
        f"Objects in frame: {current_objects}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.putText(
        annotated_frame,
        f"Total tracked: {total_tracked}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Display frame
    # --------------------------------------------------------

    cv2.imshow(
        "CodeAlpha - Object Detection and Tracking",
        annotated_frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ============================================================
# 5. RELEASE RESOURCES
# ============================================================

cap.release()
cv2.destroyAllWindows()

print("\nObject detection and tracking stopped.")
print(f"Total unique objects tracked: {len(tracked_ids)}")