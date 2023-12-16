import cv2
import numpy as np
from pathlib import Path
import supervision as sv
from ultralytics import YOLO

# Initialize YOLO model and other components
MODEL = "yolov8x.pt"
model = YOLO(MODEL)
byte_tracker = sv.ByteTrack()
annotator = sv.BoxAnnotator()

# Function to process each frame
def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model.predict(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    labels = [
        f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
        for _, _, confidence, class_id, tracker_id
        in detections
    ]
    return annotator.annotate(scene=frame.copy(), detections=detections, labels=labels)

# Read the video
CWD = Path().absolute()
VIDEO_PATH = CWD / "single.mp4"
cap = cv2.VideoCapture(str(VIDEO_PATH))

# Process and display the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    processed_frame = process_frame(frame)
    cv2.imshow('Processed Video', processed_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close windows
cap.release()
cv2.destroyAllWindows()
