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
    
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    image = transform(image).unsqueeze(0).to(device)
    start_time = time.time()
    with torch.no_grad():
        output = model(image).logits
    pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    inference_time = time.time() - start_time

    return pred, inference_time

def stitch_images(original_image, prediction):
    
    pred_image = Image.fromarray((prediction * 255).astype(np.uint8)).convert("L")
    pred_image = pred_image.resize(original_image.size, Image.NEAREST)
    
    original_np = np.array(original_image)
    pred_np = np.array(pred_image)
    
    stitched_image = np.hstack((original_np, np.stack((pred_np,) * 3, axis=-1)))
    
    return Image.fromarray(stitched_image)

def process_image(image_path, model, device, output_dir):
    
    prediction, inference_time = predict(image_path, model, device)
    
    original_image = Image.open(image_path).convert("RGB")
    
    stitched_image = stitch_images(original_image, prediction)
    
    output_image_path = os.path.join(output_dir, os.path.splitext(os.path.basename(image_path))[0] + '_stitched_prediction.png')
    os.makedirs(output_dir, exist_ok=True)
    stitched_image.save(output_image_path)
    
    print(f"Stitched prediction saved to {output_image_path}")
    print(f"Inference time: {inference_time:.4f} seconds")

def main():
    """
    Main function to perform segmentation inference and save the stitched result.
    """
    parser = argparse.ArgumentParser(description="Segmentation Model Inference Script")
    parser.add_argument('--model_path', type=str, help='Path to the trained model file (optional)')
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image file or directory')
    parser.add_argument('--gpu', type=str, default='0', help='GPU device id (default: 0)')

    args = parser.parse_args()

    device = f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu'
    
    model = load_model(args.model_path, device)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_base_dir = os.path.join(script_dir, 'output', 'test')

    if os.path.isdir(args.image_path):
        image_dir = args.image_path
        output_dir = os.path.join(output_base_dir, os.path.basename(image_dir))
        for image_file in os.listdir(image_dir):
            if image_file.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(image_dir, image_file)
                process_image(image_path, model, device, output_dir)
    else:
        image_path = args.image_path
        output_dir = os.path.join(output_base_dir, os.path.basename(os.path.dirname(image_path)))
        process_image(image_path, model, device, output_dir)

if __name__ == "__main__":
    main()
