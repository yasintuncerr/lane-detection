import os
import sys
import torch
import torch.nn.functional as F
from transformers import get_scheduler, SegformerForSemanticSegmentation
import logging
import argparse
from tqdm import tqdm
from PIL import Image


from utils.metrics import mean_iou
from torch.utils.data import DataLoader
from torchvision import transforms
from utils.datasets import Bdd100KDataset
from utils.utils import InvertMaskColors


class Trainer:
    def __init__(self, model, device, optimizer, lr_scheduler, train_loader, val_loader, epochs, save_dir, log_name):
        self.model = model
        self.device = device
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.save_dir = save_dir
        self.best_iou = 0
        self.log_name = log_name
        logging.basicConfig(
            filename=f'./{log_name}',
            filemode='a',
            format='%(message)s',
            level=logging.INFO
        )
        
        self.model.to(self.device)

    def evaluate(self):
        self.model.eval()
        total_iou = 0
        num_batches = 0

        with tqdm(total=len(self.val_loader), desc="Evaluating", unit="batch") as pbar:
            for batch in self.val_loader:
                images, masks = batch
                images = images.to(self.device)
                masks = masks.to(self.device).long()

                with torch.no_grad():
                    outputs = self.model(pixel_values=images, return_dict=True)
                    outputs = F.interpolate(outputs["logits"], size=masks.shape[-2:], mode="bilinear", align_corners=False)
                    preds = torch.argmax(outputs, dim=1)
                    preds = torch.unsqueeze(preds, dim=1)

                preds = preds.view(-1)
                masks = masks.view(-1)

                iou = mean_iou(preds, masks, self.model.config.num_labels)
                total_iou += iou
                num_batches += 1
                pbar.update(1)

        epoch_iou = total_iou / num_batches
        return epoch_iou

    def train(self):
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0.0
            with tqdm(total=len(self.train_loader), desc=f"Epoch {epoch + 1}/{self.epochs}", unit="batch") as pbar:
                for batch_idx, batch in enumerate(self.train_loader):
                    images, masks = batch
                    images = images.to(self.device)
                    masks = masks.to(self.device).squeeze(1)

                    self.optimizer.zero_grad()
                    outputs = self.model(pixel_values=images, labels=masks, return_dict=True)
                    loss = outputs["loss"]
                    loss.backward()
                    self.optimizer.step()
                    self.lr_scheduler.step()

                    outputs = F.interpolate(outputs["logits"], size=masks.shape[-2:], mode="bilinear", align_corners=False)

                    total_loss += loss.item()
                    pbar.set_postfix(loss=loss.item())
                    pbar.update(1)

            average_loss = total_loss / len(self.train_loader)
            logging.info(f"Epoch {epoch + 1}/{self.epochs}, Loss: {average_loss:.4f}")

            epoch_iou = self.evaluate()
            logging.info(f"Epoch {epoch + 1}/{self.epochs}, Mean IoU: {epoch_iou:.4f}")
            
            if epoch_iou > self.best_iou:
                logging.info(f"IoU improved from {self.best_iou:.4f} to {epoch_iou:.4f}")
                self.best_iou = epoch_iou
                logging.info("Saving the best model")
                self.model.save_pretrained(self.save_dir)
            logging.info("---")
            logging.info("learning rate: " + str(self.optimizer.param_groups[0]['lr']))

def setup(args):
    train_transform = transforms.Compose([
        transforms.Resize((360, 640)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((360, 640), interpolation=Image.NEAREST),
        InvertMaskColors()
    ])

    train_dataset = Bdd100KDataset(args.dataset_dir, split="train", image_transform=train_transform, mask_transform=mask_transform)
    val_dataset = Bdd100KDataset(args.dataset_dir, split="val", image_transform=train_transform, mask_transform=mask_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.from_pretrained:
        model = SegformerForSemanticSegmentation.from_pretrained(args.from_pretrained, ignore_mismatched_sizes=True)
    else:
        model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b2-finetuned-ade-512-512", ignore_mismatched_sizes=True)
    
    model.config.num_labels = 2
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    num_training_steps = args.num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=args.num_warmup_steps,
        num_training_steps=num_training_steps
    )

    trainer = Trainer(model, device, optimizer, lr_scheduler, train_loader, val_loader, args.num_epochs, args.save_dir, args.log_name)
    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Segformer model on BDD100K dataset.")
    parser.add_argument("--dataset_dir", type=str, help="Path to the dataset directory")
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for training")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate for the optimizer")
    parser.add_argument("--num_warmup_steps", type=int, default=500, help="Number of warmup steps for the learning rate scheduler")
    parser.add_argument("--save_dir", type=str, default='./segformer_model', help="Directory to save the best model")
    parser.add_argument("--from_pretrained", type=str, help="Path to the pretrained model directory")
    parser.add_argument("--log_name", type=str, default="segformer_training.log", help="Name of the log file")
    args = parser.parse_args()

    setup(args)    


    
    
