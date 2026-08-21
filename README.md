# Object Detection & Tracking with Analytics

A computer vision pipeline that detects and tracks people and vehicles in video, with live analytics like people count, vehicle count, and FPS.

## Tech Stack
Python, OpenCV, Ultralytics YOLOv8, ByteTrack, Pydantic, YAML

## How to Run
1. `pip install ultralytics opencv-python pydantic pyyaml`
2. Edit `config.yaml` for model/video source
3. `python detect.py`