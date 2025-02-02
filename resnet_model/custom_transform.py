import torchvision.transforms.functional as F
import torchvision.transforms as T
import random
from PIL import Image, ImageEnhance, ImageFilter  
import numpy as np

class CustomTransform:
    def __init__(self, image_size=(480, 854), model_input_size=(448, 800), mean=None, std=None):
        self.image_size = image_size  # Final resized dimensions
        self.model_input_size = model_input_size   # Final cropped dimensions
        self.mean = mean if mean else [0.485, 0.456, 0.406]  # Default mean for ImageNet
        self.std = std if std else [0.229, 0.224, 0.225]    # Default std for ImageNet

    def __call__(self, img, mask, train_mode):
        if train_mode:
            # Resize to larger size for cropping
            img = F.resize(img, self.image_size)
            mask = F.resize(mask, self.image_size)

            # Random cropping occurs randomly.
            if random.random() > 0.5: 
                i, j, h, w = T.RandomCrop.get_params(img, output_size=self.model_input_size)
                img = F.crop(img, i, j, h, w)
                mask = F.crop(mask, i, j, h, w)

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

            # Apply Gaussian blur
            # if random.random() > 0.5:
            #     img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 1.5)))
            
            # apply random noise    
            # if random.random() > 0.5:
            #     img = self.add_noise(img)    
            
            # # add specular reflection
            # if random.random() > 0.5:
            #     img = self.add_specular_reflections(img) 
                       
        # Resize image and mask
        img = F.resize(img, self.model_input_size)
        mask = F.resize(mask, self.model_input_size) 
        
        # Convert to tensor
        img = F.to_tensor(img)
        mask = F.to_tensor(mask)  # Keeps the mask in [0, 1] range (binary)

        # Normalize image (mask should NOT be normalized)
        img = F.normalize(img, mean=self.mean, std=self.std)

        return img, mask

    @staticmethod
    def add_noise(img, max_intensity=0.3):
        """Apply random noise."""
        random_intensity = np.random.uniform(0, max_intensity)    
        img_np = np.array(img).astype(np.float32) / 255.0  # Normalize
        glare = np.random.uniform(0.0, random_intensity, img_np.shape)  # Create glare pattern
        reflection = np.minimum(img_np + glare, 1.0)  # Apply and clip
        return Image.fromarray((reflection * 255).astype(np.uint8))

    @staticmethod
    def add_specular_reflections(img, intensity=0.5, num_reflections=5, size_range=(1, 5)):
        """
        Simulates concentrated specular reflections in specific regions.
        
        Args:
            img (PIL.Image): Input image.
            intensity (float): Intensity of the specular reflections (0 to 1).
            num_reflections (int): Number of specular reflection spots to add.
            size_range (tuple): Range of sizes for the reflection spots (min_size, max_size).
        
        Returns:
            PIL.Image: Image with added specular reflections.
        """
        img_np = np.array(img).astype(np.float32) / 255.0  # Normalize the image
        h, w, _ = img_np.shape  # Get image dimensions

        # Create an empty mask for specular reflections
        reflection_mask = np.zeros((h, w), dtype=np.float32)

        for _ in range(num_reflections):
            # Randomly generate the position and size of each reflection
            center_x = random.randint(30, w - 30)
            center_y = random.randint(30, h - 30)
            radius_x = random.randint(size_range[0], size_range[1])  # Horizontal radius
            radius_y = random.randint(size_range[0], size_range[1])  # Vertical radius

            # Draw an elliptical reflection
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x) ** 2 / radius_x ** 2) + ((y - center_y) ** 2 / radius_y ** 2) <= 1
            reflection_mask[mask] += intensity  # Add intensity to the reflection region

        # Add the reflection mask to the image
        reflection_mask = np.clip(reflection_mask, 0, 1)  # Ensure valid range
        img_np += reflection_mask[:, :, None]  # Add mask to all channels
        img_np = np.clip(img_np, 0, 1)  # Ensure valid range after addition

        # Convert back to PIL.Image
        return Image.fromarray((img_np * 255).astype(np.uint8))