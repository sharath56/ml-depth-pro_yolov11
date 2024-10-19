import argparse
import cv2
import depth_pro
import torch
import numpy as np
from ultralytics import YOLO
from pyfiglet import Figlet

# Function to display a banner
def display_banner(text):
    figlet = Figlet(font='big')
    banner = figlet.renderText(text)
    print(banner)

# Function to process image input
def process_image(image_path, yolo_model, depth_model, transform, device):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Could not read image {image_path}")
        return

    run_detection_and_depth(frame, yolo_model, depth_model, transform, device)
    cv2.imwrite('output_image_with_detections.jpg', frame)
    print("Image saved as output_image_with_detections.jpg")

# Function to process video input
def process_video(video_path, yolo_model, depth_model, transform, device):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_video.mp4', fourcc, 30.0, (int(cap.get(3)), int(cap.get(4))))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        run_detection_and_depth(frame, yolo_model, depth_model, transform, device)
        out.write(frame)

        # cv2.imshow('YOLO and Depth Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video saved as output_video.mp4")

# Function to process camera input
def process_camera(yolo_model, depth_model, transform, device):
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        run_detection_and_depth(frame, yolo_model, depth_model, transform, device)
        cv2.imshow('YOLO and Depth Estimation', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to run YOLO and depth estimation on a frame
def run_detection_and_depth(frame, yolo_model, depth_model, transform, device):
    # YOLO object detection
    results = yolo_model(frame)

    object_boxes = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy()

        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box[:4])
            class_name = result.names[int(cls)]
            object_boxes.append((x1, y1, x2, y2, class_name))
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

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

    # Depth estimation
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth_input = transform(image).to(device)
    f_px = None  # Set f_px if required by depth model
    prediction = depth_model.infer(depth_input, f_px=f_px)
    depth = prediction["depth"].squeeze().cpu().numpy()

    # Annotate depth information for each detected object
    for x1, y1, x2, y2, class_name in object_boxes:
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        depth_value = depth[center_y, center_x]
        text = f'{class_name} Depth: {depth_value:.2f}m'

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

        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), (0, 0, 0), -1)
        cv2.putText(frame, text, (text_x, text_y), font, font_scale, (255, 255, 255), font_thickness)

# Main function to parse arguments and run the appropriate mode
def main():
    parser = argparse.ArgumentParser(description="YOLO and Depth Estimation Script")
    parser.add_argument('--image', type=str, help='Path to the input image')
    parser.add_argument('--video', type=str, help='Path to the input video')
    parser.add_argument('--camera', action='store_true', help='Use the camera for live video input')
    args = parser.parse_args()

    # Check if GPU is available, otherwise use CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    if device.type == "cuda":
        display_banner("Using GPU")
    else:
        display_banner("Using CPU")

    # Load YOLO and depth models
    yolo_model = YOLO('yolo11s.pt')
    depth_model, transform = depth_pro.create_model_and_transforms()
    depth_model = depth_model.to(device)
    depth_model.eval()

    if args.image:
        process_image(args.image, yolo_model, depth_model, transform, device)
    elif args.video:
        process_video(args.video, yolo_model, depth_model, transform, device)
    elif args.camera:
        process_camera(yolo_model, depth_model, transform, device)
    else:
        print("Please provide an input (--image, --video, or --camera).")

if __name__ == "__main__":
    main()
