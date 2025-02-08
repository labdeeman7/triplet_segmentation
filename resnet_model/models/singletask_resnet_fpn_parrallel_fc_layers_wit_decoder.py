import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class SingleTaskResNetFPNWithTransformersAndParrallelFCLayers(nn.Module):
    def __init__(self, 
                 num_instruments,
                 instrument_to_task_classes,
                 embed_dim=64,
                 decoder_hidden_dim=256,
                 num_decoder_layers=4):
        super(SingleTaskResNetFPNWithTransformersAndParrallelFCLayers, self).__init__()

        # ResNet Backbone with 4-channel input
        self.resnet = models.resnet50(pretrained=True)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(4, 
                                      original_conv1.out_channels, 
                                      original_conv1.kernel_size,
                                      original_conv1.stride, 
                                      original_conv1.padding, 
                                      bias=original_conv1.bias)
        self.resnet.conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        self.resnet.conv1.weight.data[:, 3:, :, :] = 0.0

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

        # Feature Pyramid Network
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)

        # Downsampling masks for each FPN level
        self.mask_downsamplers = nn.ModuleDict({
            "layer1": nn.AdaptiveAvgPool2d((56, 56)),
            "layer2": nn.AdaptiveAvgPool2d((28, 28)),
            "layer3": nn.AdaptiveAvgPool2d((14, 14)),
            "layer4": nn.AdaptiveAvgPool2d((7, 7)),
        })

        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Embeddings
        self.instrument_class_embedding = nn.Embedding(num_instruments, embed_dim)
        self.instrument_embedding = nn.Sequential(
            nn.Linear((256 * 4) + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, decoder_hidden_dim)
        )
        self.background_image_embedding = nn.Sequential(
            nn.Linear(256 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, decoder_hidden_dim)
        )

        # Per-instrument transformer decoders
        self.transformer_decoders = nn.ModuleDict({
            str(instr_id): nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=decoder_hidden_dim, nhead=8),
                num_layers=num_decoder_layers
            )
            for instr_id in instrument_to_task_classes.keys()
        })

        # Fully connected layers for tasks
        self.fc_task_dict = nn.ModuleDict({
            str(instr_id): nn.Sequential(
                nn.Linear(decoder_hidden_dim, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, len(local_task_classes))
            )
            for instr_id, local_task_classes in instrument_to_task_classes.items()
        })
        
    def pad_tensor(self, tensor, max_size):
        """
        Pads tensor with -inf values so all outputs have the same shape.
        """
        pad_size = max_size - tensor.size(1)
        if pad_size > 0:
            padding = torch.full((tensor.size(0), pad_size), float('-inf'), device=tensor.device)
            return torch.cat((tensor, padding), dim=1)
        return tensor      

    def forward(self, img, mask, instrument_id):
        combined_input = torch.cat((img, mask), dim=1)

        # Backbone forward pass
        out = self.backbone["conv1"](combined_input)
        out = self.backbone["bn1"](out)
        out = self.backbone["relu"](out)
        out = self.backbone["maxpool"](out)

        layer1 = self.backbone["layer1"](out)
        layer2 = self.backbone["layer2"](layer1)
        layer3 = self.backbone["layer3"](layer2)
        layer4 = self.backbone["layer4"](layer3)

        features = OrderedDict(layer1=layer1, layer2=layer2, layer3=layer3, layer4=layer4)
        fpn_features = self.fpn(features)

        # Extract instrument and background features
        instrument_features = []
        background_features = []
        for level, downsampler in self.mask_downsamplers.items():
            fpn_feature = fpn_features[level]
            soft_mask = downsampler(mask)
            instrument_features.append(self.global_pool(fpn_feature * soft_mask))
            background_features.append(self.global_pool(fpn_feature * (1 - soft_mask)))

        aggregated_instrument_features = torch.cat([feat.view(feat.size(0), -1) for feat in instrument_features], dim=1)
        aggregated_background_features = torch.cat([feat.view(feat.size(0), -1) for feat in background_features], dim=1)

        # Background embedding
        background_embed = self.background_image_embedding(aggregated_background_features)
        background_memory = background_embed.unsqueeze(0)

        # Instrument embedding
        instrument_class_features = self.instrument_class_embedding(instrument_id)
        instrument_with_class_context_features = torch.cat((aggregated_instrument_features, instrument_class_features), dim=1)
        instrument_embed = self.instrument_embedding(instrument_with_class_context_features).unsqueeze(0)

        # Normalize embeddings
        instrument_embed = instrument_embed / (instrument_embed.norm(dim=-1, keepdim=True) + 1e-6)
        background_memory = background_memory / (background_memory.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Find max number of output classes across all instruments
        max_output_size = max(fc[-1].out_features for fc in self.fc_task_dict.values())

        # Process each instrument using its transformer decoder
        batch_preds = []
        for i in range(img.size(0)):
            instr_id = str(instrument_id[i].item())
            if instr_id in self.transformer_decoders:
                instrument_single = instrument_embed[:, i:i+1, :]  # Shape: [1, 1, decoder_hidden_dim]
                background_memory_single = background_memory[:, i:i+1, :]  # Shape: [1, 1, decoder_hidden_dim]
                
                decoder_output = self.transformer_decoders[instr_id](instrument_single, background_memory_single)
        
                # Pass through the corresponding fully connected layer
                task_pred = self.fc_task_dict[instr_id](decoder_output.squeeze(0))  # Remove sequence dimension
                task_pred = self.pad_tensor(task_pred, max_output_size)  # Pad output for consistency across instruments
                batch_preds.append(task_pred)
            else:
                raise ValueError(f"Instrument ID {instr_id} not found.")

        return torch.cat(batch_preds, dim=0)
