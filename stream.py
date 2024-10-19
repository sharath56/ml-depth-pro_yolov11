import depth_pro
import cv2
import numpy as np
from ultralytics import YOLO

# Load YOLO model
yolo_model = YOLO('yolo11s.pt')

# Load depth model and preprocessing transform
depth_model, transform = depth_pro.create_model_and_transforms()
depth_model.eval()

# Initialize video capture (0 for webcam, or provide video file path)
cap = cv2.VideoCapture(2)  # Use "video_path.mp4" for video file

while cap.isOpened():
    ret, frame = cap.read()  # Read the current frame from the video stream
    if not ret:
        break

    # YOLO object detection on the frame
    results = yolo_model(frame)

    # Initialize list to hold object boxes and class names
    object_boxes = []

    # Iterate through the YOLO results
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class indices

        # Iterate through detected objects
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = result.names[int(cls)]  # Get the class name
            object_boxes.append((x1, y1, x2, y2, class_name))  # Store the box and class name

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display class name
            label = f"{class_name}"
            font = cv2.FONT_HERSHEY_COMPLEX
            font_scale = 1.0
            font_thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, font_thickness)[0]

            text_x = x1
            text_y = y1 - 10
            rect_x1 = text_x - 5
            rect_y1 = text_y - text_size[1] - 10
            rect_x2 = text_x + text_size[0] + 5
            rect_y2 = text_y + 5
            cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
            cv2.putText(frame, label, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Process the frame for depth estimation
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert the frame to RGB
    f_px = None  # Set f_px if required by depth model

    # Transform the image for depth model input
    depth_input = transform(image)

    # Perform depth inference
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth = prediction["depth"]

    # Convert depth output to numpy array
    depth_np = depth.squeeze().cpu().numpy()

    # Annotate depth information for each detected object
    for x1, y1, x2, y2, class_name in object_boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        # Get depth value at the center of the bounding box
        depth_value = depth_np[center_y, center_x]
        text = f'{class_name} Depth: {depth_value:.2f}m'

        # Define font parameters
        font = cv2.FONT_HERSHEY_COMPLEX
        font_scale = 1.2
        font_thickness = 2
        text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]

        text_x = x1
        text_y = y1 - 10
        rect_x1 = text_x - 5
        rect_y1 = text_y - text_size[1] - 10
        rect_x2 = text_x + text_size[0] + 5
        rect_y2 = text_y + 5

        # Draw a black background rectangle and put the depth text
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

    # Show the frame with YOLO detections and depth info
    cv2.imshow('YOLO and Depth Estimation', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and destroy windows
cap.release()
cv2.destroyAllWindows()
