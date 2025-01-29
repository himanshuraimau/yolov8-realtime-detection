# Real-time Object Detection with YOLOv8

A Python application for real-time object detection in videos using YOLOv8. The program processes video files, detects objects, and generates annotated output with detection results.

## Features

- Real-time object detection using YOLOv8
- Progress tracking during video processing
- Annotated output video generation
- Detection report in text format
- Support for multiple video formats
- Live preview during processing

## Requirements

- Python 3.8+
- OpenCV
- Ultralytics YOLOv8
- CUDA (optional, for GPU acceleration)

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/object-detection.git
cd object-detection

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install ultralytics opencv-python
```

## Usage

Run the script with a video file:
```bash
python yolov8_video.py path/to/your/video.mp4
```

### Output

The program generates:
- Annotated video with detected objects
- Text file containing detection results
- Live preview window during processing

## Files

- `yolov8_video.py`: Main script for video processing
- `output_video_[timestamp].mp4`: Processed video output
- `video_detections_[timestamp].txt`: Detection results

## Controls

- Press 'q' to stop processing and exit
- Close the preview window to continue processing without preview


