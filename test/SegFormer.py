import os
import cv2
import numpy as np
from PIL import Image
import time

import torch
from torchvision.transforms.functional import to_tensor

from transformers import SegformerForSemanticSegmentation
import argparse

def load_model(model_path, device):
    # Load the trained model
    model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')
    model.config.num_labels = 2
    model.load_state_dict(torch.load(model_path))
    model.to(device)
    model.eval()
    return model

# Inference function
def predict(image_path, model, device):
    image = Image.open(image_path).convert("RGB")
    image = to_tensor(image).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model(image).logits
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    inference_time = time.time() - start_time

    return pred, inference_time

def stitch_images(original_image, prediction):
    # Convert prediction to an image
    pred_image = Image.fromarray((prediction * 255).astype(np.uint8)).convert("L")
    pred_image = pred_image.resize(original_image.size, Image.NEAREST)
    
    # Convert original and prediction images to numpy arrays
    original_np = np.array(original_image)
    pred_np = np.array(pred_image)
    
    # Stack them horizontally
    stitched_image = np.hstack((original_np, np.stack((pred_np,)*3, axis=-1)))
    
    return Image.fromarray(stitched_image)

def main():
    parser = argparse.ArgumentParser(description="Segmentation Model Inference Script")
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id (default: 0)')

    args = parser.parse_args()

    # Set device
    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    model = load_model(args.model_path, device)
    
    # Run prediction
    prediction, inference_time = predict(args.image_path, model, device)
    
    # Load original image
    original_image = Image.open(args.image_path).convert("RGB")
    
    # Stitch images
    stitched_image = stitch_images(original_image, prediction)
    
    # Save the result
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_image_path = os.path.join(script_dir, os.path.splitext(os.path.basename(args.image_path))[0] + '_stitched_prediction.png')
    stitched_image.save(output_image_path)
    
    print(f"Stitched prediction saved to {output_image_path}")
    print(f"Inference time: {inference_time:.4f} seconds")

if __name__ == "__main__":
    main()
