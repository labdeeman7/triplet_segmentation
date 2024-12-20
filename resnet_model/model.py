import torch
import torch.nn as nn
from torchvision import models

# Multitask ResNet
class MultiTaskResNet(nn.Module):
    def __init__(self, num_instruments, num_verbs, num_targets):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer to accept 4 channels
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(
            in_channels=4,  # 3 for RGB + 1 for mask
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )

        # Copy the pretrained weights for the RGB channels
        self.resnet.conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        # Initialize the weights for the mask channel to zeros
        self.resnet.conv1.weight.data[:, 3:, :, :] = 0.0
        
        
        # Save the number of input features of the original fc layer
        num_features = self.resnet.fc.in_features
        
        self.resnet.fc = nn.Identity()  # Remove the default classification head

        # Learnable embedding for instruments
        self.instrument_embedding = nn.Embedding(num_instruments, 64)

        # Fully connected layers for verbs and targets
        self.fc_verb = nn.Sequential(
            nn.Linear(num_features  + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_verbs)  # Output logits for verbs
        )

        self.fc_target = nn.Sequential(
            nn.Linear(num_features  + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_targets)  # Output logits for targets
        )

    def forward(self, img, mask, instrument_id):
       # Concatenate the image and mask along the channel dimension
        combined_input = torch.cat((img, mask), dim=1)  # Result: [batch_size, 4, H, W]
        
        img_features = self.resnet(combined_input)

        # Embed the instrument ID
        instrument_features = self.instrument_embedding(instrument_id)

        # Concatenate image features with instrument features
        combined_features = torch.cat((img_features, instrument_features), dim=1)

        # Predict verbs and targets
        action_preds = self.fc_verb(combined_features)
        target_preds = self.fc_target(combined_features)

        return action_preds, target_preds
