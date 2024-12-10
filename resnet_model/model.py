import torch
import torch.nn as nn
from torchvision import models

# Multitask ResNet
class MultiTaskResNet(nn.Module):
    def __init__(self, num_instruments, num_actions, num_targets):
        super(MultiTaskResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)

        # Freeze backbone layers if needed
        for param in self.resnet.parameters():
            param.requires_grad = True

        # Add input for instrument annotation
        self.fc_instrument = nn.Linear(256, 64)  # Adjust the input size if needed
        self.fc_concat = nn.Linear(self.resnet.fc.in_features + 64, 512)

        # Two separate task heads
        self.fc_action = nn.Linear(512, num_actions)
        self.fc_target = nn.Linear(512, num_targets)

    def forward(self, x, instrument_annotation):
        # Extract features from ResNet
        features = self.resnet(x)
        
        # Process instrument annotation
        instrument_features = self.fc_instrument(instrument_annotation)

        # Concatenate image features and instrument features
        combined_features = torch.cat((features, instrument_features), dim=1)

        # Process combined features
        combined_features = self.fc_concat(combined_features)

        # Task-specific predictions
        action_output = self.fc_action(combined_features)
        target_output = self.fc_target(combined_features)

        return action_output, target_output