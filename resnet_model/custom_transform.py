import torchvision.transforms.functional as F
import random

class CustomTransform:
    def __init__(self, image_size=(224, 224), mean=None, std=None):
        self.image_size = image_size
        self.mean = mean if mean else [0.485, 0.456, 0.406]  # Default mean for ImageNet
        self.std = std if std else [0.229, 0.224, 0.225]    # Default std for ImageNet

    def __call__(self, img, mask, train_mode):
        if train_mode:
            # Random horizontal flip
            if random.random() > 0.5:
                img = F.hflip(img)
                mask = F.hflip(mask)

            # Random vertical flip
            if random.random() > 0.5:
                img = F.vflip(img)
                mask = F.vflip(mask)

        # Resize image and mask
        img = F.resize(img, self.image_size)
        mask = F.resize(mask, self.image_size)

        # Convert to tensor
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)  # Keeps the mask in [0, 1] range (binary)

        # Normalize image (mask should NOT be normalized)
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, mask
