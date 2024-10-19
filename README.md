# README: Object Detection and Depth Estimation Pipeline

This README file provides a step-by-step guide to the object detection and depth estimation process using YOLO and a depth model. The code detects objects in an image, estimates their depth, and visualizes both the bounding boxes and depth information.

## Prerequisites

Before running this code, ensure that you have the following libraries installed:

- `opencv-python`
- `numpy`
- `torch`
- `ultralytics` (for YOLO)
- `depth_pro` (a custom library for depth estimation, or use any alternative depth estimation model)

Install the necessary dependencies with the following command:
```bash
pip install opencv-python numpy torch ultralytics
```

Make sure you have a YOLO model checkpoint (`yolo11s.pt`) and a compatible depth estimation model.

## Project Structure

```plaintext
project/
│
├── yolo11s.pt                 # Pre-trained YOLO model
├── image.jpg                  # Input image for object detection
├── depth_model.pth            # Pre-trained depth model
├── detection_with_depth.jpg    # Output image with object detection and depth info
├── inverted_depth_map.jpg      # Inverted depth map visualization
├── yolo_output.jpg             # YOLO detection output image
├── script.py                  # Main script for object detection and depth estimation
└── README.md                  # This README file

```
## Architecture 

![Alt text](Model_ML_depth_estimation_yolo/arch.png)

## Flow Diagram
```
Start
  |
  V
Load YOLO Model
  |
  V
Load Input Image (cv2.imread)
  |
  V
Perform YOLO Inference (yolo_model)
  |
  V
Extract Results from YOLO Output
  | 
  V
Iterate Through Detected Objects
  |   (For each object: extract bounding box, class name)
  +---------------------------------------+
  |                                       |
  V                                       V
Draw Bounding Boxes on Image         Save Bounding Box Coordinates 
                                      and Class Names for Later Use
  |                                       |
  V                                       |
Display Image with Bounding Boxes <-------+
  |
  V
Load Depth Model and Preprocess Image for Depth Estimation
  |
  V
Perform Depth Estimation Inference
  |
  V
Obtain Depth Values from Depth Map (Depth as numpy array)
  |
  V
For Each Detected Object:
  |   (Calculate the center of each bounding box)
  +----------------------------+
  |                            |
  V                            V
Retrieve Depth Value     Draw Depth Information 
at Bounding Box Center   Next to Bounding Box
  |
  V
Display Image with Depth Information
  |
  V
Save Final Image with Bounding Boxes and Depth Data
  |
  V
Normalize and Invert Depth Map
  |
  V
Display Inverted Depth Map
  |
  V
Save Inverted Depth Map
  |
  V
End

```
## Code Overview

1. **Loading the YOLO Model:**
   - The YOLO model is loaded using the `ultralytics` library, which allows for object detection on the input image.
   
   ```python
   yolo_model = YOLO('yolo11s.pt')
   ```

2. **Activate conda Env:**
   ```
   conda create -n env_name -y python=3.9
   conda activate env_name
   ```
3. **Running from python:**
   ```
   python script.py --flags 
   
   ```