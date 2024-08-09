import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF


class Bdd100KDataset(Dataset):
    def __init__(self, dataset_dir, split="train", image_transform=None, mask_transform=None):
        self.dataset_dir = dataset_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        
        images_dir = os.path.join(dataset_dir, "bdd100k/images/100k", split)
        masks_dir = os.path.join(dataset_dir, "bdd100k/labels/lane/masks", split)
        
        self.images = [os.path.join(images_dir, img) for img in os.listdir(images_dir) if img.endswith(('.jpg', '.png'))]
        
        self.masks = []
        for img_path in self.images:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            mask_path = os.path.join(masks_dir, base_name + '.png')
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Masked image not found for {img_path}")
            self.masks.append(mask_path)
            
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            mask = self.mask_transform(mask)
            
        mask = TF.to_tensor(mask)
        mask = (mask > 0).long()
    
        return image, mask

# Example usage:
# image_transform = transforms.Compose([...])  # Define your image transforms
# mask_transform = transforms.Compose([...])  # Define your mask transforms
# dataset = Bdd100KDataset(dataset_dir="/path/to/dataset", split="train" or "val", image_transform=image_transform, mask_transform=mask_transform)
