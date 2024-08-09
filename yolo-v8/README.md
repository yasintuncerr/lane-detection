---
license: mit
---

# YOLOv8 Segmentation Training Script

This script is designed to train a YOLOv8 segmentation model using a specified dataset and configuration. It supports training on multiple GPUs, logs the training process, and evaluates the model's performance.

## Requirements

- Python 3.6 or higher
- PyTorch
- Ultralytics YOLOv8

Install the required packages using pip:

```bash
pip install torch ultralytics
```

## Usage

### Command-Line Arguments

- `dataset_yaml`: **(Required)** Path to the YAML file describing the dataset.
- `runs_directory`: **(Required)** Directory where training runs and logs will be saved.
- `pretrained_model`: **(Optional)** Path to a pretrained model or one of the YOLOv8 pretrained models 
  (e.g., yolov8s-seg.pt, yolov8m-seg.pt, yolov8l-seg.pt, yolov8x-seg.pt). Default is yolov8s-seg.pt.
- `batch_size`: **(Required)** Batch size for training.
- `gpu_count`: **(Optional)** Number of GPUs to use for training. Default is 1.
- `img_size`: **(Optional)** Image size for training. Default is 640.

### Example Usage

```bash
python train.py --dataset_yaml path/to/dataset.yaml --runs_directory runs/ --batch_size 16 --gpu_count 2
```

## Dataset

The [Zenseact Open Dataset](https://zod.zenseact.com/) includes a variety of autonomous driving data. A minimal version for lane-marking, [Zod Mini 2D Road Scenes](https://huggingface.co/datasets/8bits-ai/ZOD-Mini-2D-Road-Scenes), can be used for training.

## Logging

The training process and evaluation metrics are logged in `training_log.txt` in the root directory. The log file includes information about the start of training, GPU availability, and any errors encountered during training.

## Training on Multiple GPUs

The script allows for multi-GPU training by specifying the number of GPUs with the `--gpu_count` argument. If more GPUs are requested than are available, the script will raise an error and log it.

## Output

The trained model and logs will be saved in the specified `--runs_directory`. Each training run will be saved in a separate subdirectory named `yolov8_segmentation`.

## Test Usage

The following examples demonstrate how to use the script for testing with both a single image and a directory of images.

### Example 1: Processing a Single Image

To process a single image using the YOLOv8 segmentation model, you can run the script as follows:

```bash
python test.py --model_path yolov8s-seg.pt --image_path path/to/your/image.jpg --output_dir output/results --yaml_path path/to/dataset.yaml
```

- `model_path`: Path to the YOLOv8 model. If you don't have a custom model, you can use a pretrained one like yolov8s-seg.pt.
- `image_path`: Path to the image you want to process.
- `output_dir`: Directory where the output images and detection results will be saved. Default is `output/test`.
- `yaml_path`: Path to the YAML file containing the class names.

### Example 2: Processing a Directory of Images

If you want to process all images in a directory, run the following command:

```bash
python test.py --model_path yolov8s-seg.pt --image_dir path/to/your/images/ --output_dir output/results --yaml_path path/to/dataset.yaml
```

- `image_dir`: Path to the directory containing the images you want to process.
