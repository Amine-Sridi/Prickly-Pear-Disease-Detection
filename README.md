# Prickly Pear Disease Detection

A machine learning project for automated detection and classification of diseases in prickly pear plants using deep learning models. The system identifies healthy and diseased plants in real-time using computer vision techniques.

## Overview

This project implements disease detection for prickly pear crops using state-of-the-art object detection models. It supports both real-time detection via webcam and batch processing through trained neural networks.

## Features

- Real-time disease detection using webcam input
- Multiple model architectures: YOLOv8 and MobileNet
- Pre-trained weights for immediate inference
- Training pipelines for model customization
- Bounding box visualization with confidence scores
- Binary classification: Healthy vs Diseased

## Requirements

- Python 3.8 or higher
- PyTorch or TensorFlow
- OpenCV
- Ultralytics YOLO
- See requirements.txt for complete dependencies

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Prickly-Pear-Disease-Detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download or place pre-trained weights in the appropriate directories

## Usage

### Real-Time Detection

Run real-time detection using your webcam:
```bash
python realtime.py
```

Press 'q' to quit the detection window.

### Training

Train a new model using the provided notebooks:

- **YOLOv8 Training**: Open `YoLo.ipynb` in Jupyter and run all cells
- **MobileNet Training**: Open `Mobilenet.ipynb` in Jupyter and run all cells

## Project Structure

```
Prickly-Pear-Disease-Detection/
├── realtime.py                 # Real-time detection script
├── YoLo.ipynb                  # YOLOv8 training notebook
├── Mobilenet.ipynb             # MobileNet training notebook
├── requirements.txt            # Project dependencies
├── yolov8n.pt                  # Pre-trained YOLOv8 nano weights
└── datasets/
    └── runs/detect/
        ├── train/              # Initial training results
        ├── train2/             # Second training iteration
        ├── train3/             # Third training iteration
        └── train4/             # Final training iteration
            └── weights/        # Model weights for each epoch
```

## Models

### YOLOv8
- Architecture: YOLOv8 nano variant for resource-efficient inference
- Input: Images or video frames
- Output: Bounding boxes with class labels and confidence scores
- Classes: Healthy, Diseased

### MobileNet
- Architecture: Mobile-optimized neural network
- Suitable for deployment on edge devices
- Alternative approach for disease classification

## Model Performance

Training results are stored in the datasets/runs/detect/ directory with:
- Training metrics (results.csv)
- TensorBoard event files for visualization
- Trained weights for each epoch
- Best model checkpoint

## Input/Output

**Input**: Video frames from webcam or image files  
**Output**: 
- Bounding boxes around detected plants
- Class labels (Healthy/Diseased)
- Confidence scores
- Color-coded boxes (green for healthy, red for diseased)

