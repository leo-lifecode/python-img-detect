# Object Detection with YOLOv5 and Image Cropping

## Overview
This project demonstrates how to use the YOLOv5 model for object detection in images. The steps involved are:
1. **Initial Object Detection**: Detect objects in the image and draw bounding boxes around them.
2. **Image Cropping**: Allow the user to select and crop a specific area of the image.
3. **Display Results**: Show two images — one cropped and one with bounding boxes drawn on the cropped image.

## Requirements
Before running the project, you need to set up a Python environment and install the required dependencies.

### Step 1: Create and Activate a Virtual Environment

To ensure the project runs with the correct dependencies, it's recommended to use a virtual environment.

#### On Windows:

```bash
python -m venv venv
venv\Scripts\activate

pip install opencv-python ultralytics matplotlib numpy

python object_detection.py


