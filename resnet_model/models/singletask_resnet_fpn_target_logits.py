import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class SingleTaskResNetFPNTargetLogits(nn.Module):
    def __init__(self, 
                 config, 
                 num_instruments,
                 num_task_class):
        super(SingleTaskResNetFPNTargetLogits, self).__init__()
        
        num_target_logits = config.num_target_logits  # Extract from config
        self.num_target_logits = num_target_logits
        
        self.resnet = models.resnet50(pretrained=True)
        original_conv1 = self.resnet.conv1
        
        # New input channels: RGB(3) + mask(1) + target_logits(num_target_logits)
        total_input_channels = 3 + 1 + num_target_logits
        self.resnet.conv1 = nn.Conv2d(
            in_channels=total_input_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias
        )
        
        # Initialize weights
        self.resnet.conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        nn.init.kaiming_normal_(self.resnet.conv1.weight[:, 3:, :, :], mode='fan_out', nonlinearity='relu')

        # Backbone layers
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
            in_channels_list=[256, 512, 1024, 2048],
            out_channels=256
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.instrument_embedding = nn.Embedding(num_instruments, 64)

        self.fc_task = nn.Sequential(
            nn.Linear((256*4) + 64, 256),
            nn.ReLU(),
            nn.Linear(256, num_task_class)
        )

    def forward(self, img, mask, target_logits, instrument_id):
        # Concatenate image, mask, and target logits along the channel dimension
        combined_input = torch.cat((img, mask, target_logits), dim=1)
        
        out_conv_1 = self.backbone["conv1"](combined_input)
        out_bn1 = self.backbone["bn1"](out_conv_1)
        out_relu = self.backbone["relu"](out_bn1)
        out_maxpool = self.backbone["maxpool"](out_relu)
        
        layer_1_output = self.backbone["layer1"](out_maxpool)
        layer_2_output = self.backbone["layer2"](layer_1_output)
        layer_3_output = self.backbone["layer3"](layer_2_output)
        layer_4_output = self.backbone["layer4"](layer_3_output)

        features = OrderedDict(
            layer1=layer_1_output,
            layer2=layer_2_output,
            layer3=layer_3_output,
            layer4=layer_4_output
        )

        fpn_features = self.fpn(features)
        pooled_features = [self.global_pool(fpn_features[level]) for level in fpn_features.keys()]
        aggregated_features = torch.cat([feat.view(feat.size(0), -1) for feat in pooled_features], dim=1)
        
        instrument_features = self.instrument_embedding(instrument_id)
        combined_features = torch.cat((aggregated_features, instrument_features), dim=1)

        task_preds = self.fc_task(combined_features)
        return task_preds
