import cv2
from detector import HygieneDetector

source = "static/clips/clip_20260903073608_Staff_1.webm"
capture = cv2.VideoCapture(source)
ok, frame = capture.read()
capture.release()
if not ok:
    raise RuntimeError(f"Unable to read {source}")

detector = HygieneDetector()
annotated, detections = detector.process_frame(frame)
print(f"model_classes={detector.class_names}")
print(f"violation_classes={sorted(detector.violation_classes)}")
print(f"first_frame_detections={detections}")
print(f"annotated_shape={annotated.shape}")
