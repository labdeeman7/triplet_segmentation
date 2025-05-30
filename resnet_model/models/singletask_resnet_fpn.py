import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

# Multitask ResNet
class SingleTaskResNetFPN(nn.Module):
    def __init__(self, config, num_instruments, num_task_class):
        
        super(SingleTaskResNetFPN, self).__init__()
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
        
        # Extract intermediate layers for FPN
        self.backbone = nn.ModuleDict({
            "conv1": self.resnet.conv1,
            "bn1": self.resnet.bn1,
            "relu": self.resnet.relu,
            "maxpool": self.resnet.maxpool,            
            "layer1": self.resnet.layer1,
            "layer2": self.resnet.layer2,
            "layer3": self.resnet.layer3,
            "layer4": self.resnet.layer4
        })
        
        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[
                256,  # layer1 output channels
                512,  # layer2 output channels
                1024, # layer3 output channels
                2048  # layer4 output channels
            ],
            out_channels=256  # FPN output channels for each level
        )
        
        # Global pooling layer to aggregate FPN features
        self.global_pool = nn.AdaptiveAvgPool2d(1)            
        
        self.instrument_embedding = nn.Embedding(num_instruments, 64)

        # Fully connected layers for verbs and targets
        self.fc_task = nn.Sequential(
            nn.Linear( (256*4)  + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_task_class)  # Output logits for task
        )
       

    def forward(self, img, mask, instrument_id):
       # Concatenate the image and mask along the channel dimension
        combined_input = torch.cat((img, mask), dim=1)  # Result: [batch_size, 4, H, W]
             
        out_conv_1 = self.backbone["conv1"](combined_input)
        out_bn1 = self.backbone["bn1"](out_conv_1)
        out_relu = self.backbone["bn1"](out_bn1)
        out_maxpool = self.backbone["maxpool"](out_relu)
        
        layer_1_output = self.backbone["layer1"](out_maxpool)
        layer_2_output = self.backbone["layer2"](layer_1_output)
        layer_3_output = self.backbone["layer3"](layer_2_output)
        layer_4_output = self.backbone["layer4"](layer_3_output)
 
        features = OrderedDict(
            layer1 = layer_1_output,
            layer2 = layer_2_output,
            layer3 = layer_3_output,
            layer4 = layer_4_output
        )
            
        # Generate FPN features
        fpn_features = self.fpn(features)
                
        # Aggregate FPN features into a single feature vector
        pooled_features = [self.global_pool(fpn_features[level]) for level in fpn_features.keys()]
        
        aggregated_features = torch.cat([feat.view(feat.size(0), -1) for feat in pooled_features], dim=1)
        
        # Embed the instrument ID
        instrument_features = self.instrument_embedding(instrument_id)

        # Concatenate image features with instrument features
        combined_features = torch.cat((aggregated_features, instrument_features), dim=1)        

        # Predict verbs and targets
        task_preds = self.fc_task(combined_features)

        return task_preds
