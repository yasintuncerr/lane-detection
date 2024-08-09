import logging
from ultralytics import YOLO
import os
import argparse
import torch

# Parse command line arguments
parser = argparse.ArgumentParser(description="YOLOv8 Segmentation Training Script")
parser.add_argument("--yaml_file_path", type=str, required=True, help="Path to the YAML file")
parser.add_argument("--runs_directory", type=str, required=True, help="Directory to save training runs")
parser.add_argument("--pretrained_model", type=str, default="yolov8s-seg.pt", help="Path to the pretrained model or yolo pretrained model name (default: yolov8s-seg.pt, other: yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt) ")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
parser.add_argument("--gpu_count", type=int, default=1, help="Number of GPUs to use for training")
parser.add_argument("--img_size", type=int, default=640, help="Image size for training (default: 640)")
args = parser.parse_args()

# Define the logfile path
logfile_path = "training_log.txt"

# Initialize logging
logging.basicConfig(
    filename=logfile_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Log start of training
logging.info("Starting YOLOv8 segmentation training...")

# Check GPU availability
available_gpus = torch.cuda.device_count()
if args.gpu_count > available_gpus:
    logging.error(f"Requested {args.gpu_count} GPUs, but only {available_gpus} are available.")
    raise ValueError(f"Requested {args.gpu_count} GPUs, but only {available_gpus} are available.")


# Initialize the YOLO model (using a pre-trained YOLOv8 segmentation model if provided)
model = YOLO(args.pretrained_model)

# Create a string of GPU devices
gpu_devices = ",".join(str(i) for i in range(args.gpu_count))

try:
    # Train the model
    model.train(
        data=args.yaml_file_path,        # Path to the dataset YAML file
        epochs=50,                       # Number of epochs to train
        imgsz=args.img_size,                       # Image size for training (width and height will be resized to 640)
        batch=args.batch_size,           # Batch size for training
        name='yolov8_segmentation',      # Name of the training run (for logging)
        device=gpu_devices,              # GPU devices (use '0,1,2' to utilize multiple GPUs)
        rect=False,                      # Ensure aspect ratio is not maintained with padding for multi-GPU compatibility
        project=args.runs_directory      # Directory where runs will be saved
    )
    
    # Evaluate the model
    metrics = model.val()
    
    # Log the evaluation metrics
    logging.info("Evaluation Metrics: %s", metrics)

    print(f"Training complete. Metrics logged to {logfile_path}")

except Exception as e:
    logging.error("An error occurred during training: %s", str(e))
    print(f"An error occurred: {str(e)}")
