import torchvision.transforms.functional as F
import torchvision.transforms as T
import random
from PIL import Image, ImageEnhance, ImageFilter  
import numpy as np
import torch

class CustomTransform:
    def __init__(self, image_size=(224, 224), model_input_size=(224, 224), mean=None, std=None):
        self.image_size = image_size  # Final resized dimensions
        self.model_input_size = model_input_size   # Final cropped dimensions
        self.mean = mean if mean else [0.485, 0.456, 0.406]  # Default mean for ImageNet
        self.std = std if std else [0.229, 0.224, 0.225]    # Default std for ImageNet

    def __call__(self, img, mask, split):
        
        train_mode = (split == 'train') 
        
        img = F.resize(img, self.model_input_size)
        mask = F.resize(mask, self.model_input_size)
        
        if train_mode:

            # Random horizontal flip
            if random.random() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)

            # Random brightness and contrast adjustments
            if random.random() > 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))  # Brightness adjustment
            if random.random() > 0.5:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(random.uniform(0.8, 1.2))  # Contrast adjustment

            # Color jittering
            if random.random() > 0.5:
                img = F.adjust_hue(img, random.uniform(-0.05, 0.05))  # Slight hue change
                img = F.adjust_saturation(img, random.uniform(0.8, 1.2))  # Saturation change

           

        
        # Convert to tensor
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)  # Keeps the mask in [0, 1] range (binary)

        # Normalize image (mask should NOT be normalized)
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, mask
    

class CustomTransformWithLogits:
    def __init__(self, image_size=(224, 224), model_input_size=(224, 224), mean=None, std=None):
        self.image_size = image_size
        self.model_input_size = model_input_size
        self.mean = mean if mean else [0.485, 0.456, 0.406]
        self.std = std if std else [0.229, 0.224, 0.225]

    def __call__(self, img, mask, target_logits, split):
        
        train_mode = (split == 'train')
        # Resize all to model_input_size
        img = F.resize(img, self.model_input_size)
        mask = F.resize(mask, self.model_input_size)
        target_logits = F.resize(target_logits, self.model_input_size)  # torch tensor [C,H,W]

        if train_mode:
            if random.random() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)
                target_logits = torch.flip(target_logits, dims=[2])

            if random.random() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)
                target_logits = torch.flip(target_logits, dims=[1])

            if random.random() > 0.5:
                img = ImageEnhance.Brightness(img).enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                img = ImageEnhance.Contrast(img).enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                img = F.adjust_hue(img, random.uniform(-0.05, 0.05))
                img = F.adjust_saturation(img, random.uniform(0.8, 1.2))

        img = F.to_tensor(img)
        mask = F.to_tensor(mask)
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, mask, target_logits
