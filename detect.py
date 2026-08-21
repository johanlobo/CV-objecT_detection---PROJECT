import cv2
import time
from ultralytics import YOLO
from config_loader import load_config

config = load_config()

model = YOLO(config.model_path)
cap = cv2.VideoCapture(config.video_source)

seen_person_ids = set()
seen_vehicle_ids = set()
vehicle_classes = set(config.vehicle_classes)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=config.confidence_threshold)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.tolist()
        classes = results[0].boxes.cls.tolist()

        for track_id, cls_id in zip(ids, classes):
            class_name = model.names[int(cls_id)]
            if class_name == "person":
                seen_person_ids.add(track_id)
            elif class_name in vehicle_classes:
                seen_vehicle_ids.add(track_id)

    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    cv2.putText(annotated_frame, f"People: {len(seen_person_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"Vehicles: {len(seen_vehicle_ids)}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Tracking + Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()