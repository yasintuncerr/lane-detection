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
    """
    Load the Segformer model.
    
    Args:
        model_path (str): Path to the trained model file. If None, load the default pretrained model.
        device (str): Device to load the model on ('cuda' or 'cpu').
    
    Returns:
        model (torch.nn.Module): Loaded model.
    """
    if model_path:
        model = SegformerForSemanticSegmentation.from_pretrained(model_path)
        model.config.num_labels = 2
    else:
        model = SegformerForSemanticSegmentation.from_pretrained('nvidia/segformer-b2-finetuned-ade-512-512')
        model.config.num_labels = 2
    
    model.to(device)
    model.eval()
    return model

def predict(image_path, model, device):
    """
    Perform segmentation prediction on the given image using the loaded model.
    
    Args:
        image_path (str): Path to the input image file.
        model (torch.nn.Module): Loaded model.
        device (str): Device to perform inference on ('cuda' or 'cpu').
    
    Returns:
        pred (np.ndarray): Prediction mask as a numpy array.
        inference_time (float): Time taken for inference in seconds.
    """
    image = Image.open(image_path).convert("RGB")
    image = to_tensor(image).unsqueeze(0).to(device)

    start_time = time.time()
    with torch.no_grad():
        output = model(image).logits
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    inference_time = time.time() - start_time

    return pred, inference_time

def stitch_images(original_image, prediction):
    """
    Stitch the original image and the prediction mask side by side.
    
    Args:
        original_image (PIL.Image.Image): Original input image.
        prediction (np.ndarray): Prediction mask as a numpy array.
    
    Returns:
        stitched_image (PIL.Image.Image): Stitched image combining original and prediction.
    """
    pred_image = Image.fromarray((prediction * 255).astype(np.uint8)).convert("L")
    pred_image = pred_image.resize(original_image.size, Image.NEAREST)
    
    original_np = np.array(original_image)
    pred_np = np.array(pred_image)
    
    stitched_image = np.hstack((original_np, np.stack((pred_np,) * 3, axis=-1)))
    
    return Image.fromarray(stitched_image)

def main():
    """
    Main function to perform segmentation inference and save the stitched result.
    """
    parser = argparse.ArgumentParser(description="Segmentation Model Inference Script")
    parser.add_argument('--model_path', type=str, help='Path to the trained model file (optional)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id (default: 0)')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    model = load_model(args.model_path, device)
    
    prediction, inference_time = predict(args.image_path, model, device)
    
    original_image = Image.open(args.image_path).convert("RGB")
    
    stitched_image = stitch_images(original_image, prediction)
    
    output_image_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                     os.path.splitext(os.path.basename(args.image_path))[0] + '_stitched_prediction.png')
    stitched_image.save(output_image_path)
    
    print(f"Stitched prediction saved to {output_image_path}")
    print(f"Inference time: {inference_time:.4f} seconds")

if __name__ == "__main__":
    main()
