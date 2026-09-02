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

first_seen_time = {}
zone = (200, 150, 500, 400)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.track(frame, persist=True, conf=config.confidence_threshold)

    zone_count = 0
    active_dwell_times = []  # NEW: dwell times of people currently on screen

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.tolist()
        classes = results[0].boxes.cls.tolist()
        boxes = results[0].boxes.xyxy.tolist()

        current_time = time.time()

        for track_id, cls_id, box in zip(ids, classes, boxes):
            class_name = model.names[int(cls_id)]
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

            if class_name == "person":
                seen_person_ids.add(track_id)

                if track_id not in first_seen_time:
                    first_seen_time[track_id] = current_time

                # NEW: how long this specific person has been visible so far
                dwell = current_time - first_seen_time[track_id]
                active_dwell_times.append(dwell)

                if zone[0] <= cx <= zone[2] and zone[1] <= cy <= zone[3]:
                    zone_count += 1

            elif class_name in vehicle_classes:
                seen_vehicle_ids.add(track_id)

    # NEW: average dwell time across everyone currently visible
    avg_dwell = sum(active_dwell_times) / len(active_dwell_times) if active_dwell_times else 0

    annotated_frame = results[0].plot()

    cv2.rectangle(annotated_frame, (zone[0], zone[1]), (zone[2], zone[3]), (255, 0, 0), 2)

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time else 0
    prev_time = curr_time

    # Right-side stats (unchanged position), with new fonts/colors per stat
    cv2.putText(annotated_frame, f"People: {len(seen_person_ids)}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)  # green, standard font

    cv2.putText(annotated_frame, f"Vehicles: {len(seen_vehicle_ids)}", (10, 70),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 200, 255), 2)  # orange, duplex font

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 110),
                cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 0), 1)  # cyan, triplex font

    cv2.putText(annotated_frame, f"In Zone: {zone_count}", (10, 150),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 255), 2)  # pink/magenta

    # NEW — Left side of screen: Average Dwell Time
    h, w = annotated_frame.shape[:2]
    cv2.putText(annotated_frame, f"Avg Dwell: {avg_dwell:.1f}s", (w - 320, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)  # yellow, top-right actually

    cv2.imshow("Tracking + Counting", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()