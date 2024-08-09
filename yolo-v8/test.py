import os
import cv2
import numpy as np
import yaml
from ultralytics import YOLO
import argparse

def pad_to_square(image, color=(255, 255, 255)):
    h, w = image.shape[:2]
    size = max(h, w)
    padded_image = np.full((size, size, 3), color, dtype=np.uint8)
    padded_image[(size - h) // 2:(size - h) // 2 + h, (size - w) // 2:(size - w) // 2 + w] = image
    return padded_image

def resize_image(image, size=(640, 640)):
    return cv2.resize(image, size)

def draw_segmentation(image, results):
    for result in results:
        if result.masks is not None:
            masks = result.masks.data.cpu().numpy()  # Segmentation masks
            for mask in masks:
                mask = mask.squeeze()
                mask = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 255, 0), 2)
    return image

def save_detection_info(results, label_names, output_path):
    detection_info = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding box coordinates
        classes = result.boxes.cls.cpu().numpy()  # Class labels
        for box, cls in zip(boxes, classes):
            x1, y1, x2, y2 = map(int, box)
            label = label_names[int(cls)]
            detection_info.append(f"{label} {x1} {y1} {x2} {y2}")

    with open(output_path, 'w') as f:
        for info in detection_info:
            f.write(f"{info}\n")

def process_image(model, image_path, seg_output_dir, det_output_dir, label_names, color=(255, 255, 255)):
    original_image = cv2.imread(image_path)
    if original_image is None:
        print(f"Could not read image: {image_path}")
        return
    
    padded_image = pad_to_square(original_image, color)
    resized_image = resize_image(padded_image)

    results = model(resized_image)

    # Draw segmentation on image
    segmented_image = draw_segmentation(resized_image.copy(), results)

    # Save segmentation image
    image_name = os.path.basename(image_path)
    seg_output_path = os.path.join(seg_output_dir, f"output_{image_name}")
    cv2.imwrite(seg_output_path, segmented_image)
    print(f"Saved segmentation output to: {seg_output_path}")

    # Save detection info
    det_output_path = os.path.join(det_output_dir, f"detection_{os.path.splitext(image_name)[0]}.txt")
    save_detection_info(results, label_names, det_output_path)
    print(f"Saved detection output to: {det_output_path}")

def main(args):
    model = YOLO(args.model_path)

    seg_output_dir = os.path.join(args.output_dir, "segmentation")
    det_output_dir = os.path.join(args.output_dir, "detection")

    if not os.path.exists(seg_output_dir):
        os.makedirs(seg_output_dir)
    if not os.path.exists(det_output_dir):
        os.makedirs(det_output_dir)

    with open(args.yaml_path, 'r') as file:
        yaml_data = yaml.safe_load(file)
        label_names = yaml_data['names']

    if os.path.isdir(args.image_path):
        for image_name in os.listdir(args.image_path):
            image_path = os.path.join(args.image_path, image_name)
            if os.path.isfile(image_path):
                process_image(model, image_path, seg_output_dir, det_output_dir, label_names)
    else:
        process_image(model, args.image_path, seg_output_dir, det_output_dir, label_names)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Script")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the YOLOv8 model")
    parser.add_argument("--image_path", type=str, required=True, help="Path to an image or directory of images")
    parser.add_argument("--output_dir", type=str, default="output/test", help="Directory to save output images")
    parser.add_argument("--yaml_path", type=str, required=True, help="Path to the YAML file with label names")

    args = parser.parse_args()

    main(args)
