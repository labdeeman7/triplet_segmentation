import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class ThreeTaskResNetFPNWithMoEDecodersAndSoftmaxInputs(nn.Module):
    def __init__(self, 
                 config,
                 instrument_to_verb_classes, 
                 instrument_to_target_classes, 
                 instrument_to_verbtarget_classes, 
                 embed_dim=64, 
                 decoder_hidden_dim=256, 
                 num_decoder_layers=1):
        super(ThreeTaskResNetFPNWithMoEDecodersAndSoftmaxInputs, self).__init__()

        self.num_instruments = len(instrument_to_verb_classes)  # Number of instruments
        # ResNet Backbone
        self.resnet = models.resnet50(pretrained=True)
        original_conv1 = self.resnet.conv1
        self.resnet.conv1 = nn.Conv2d(4, original_conv1.out_channels, original_conv1.kernel_size,
                                      original_conv1.stride, original_conv1.padding, bias=original_conv1.bias)
        self.resnet.conv1.weight.data[:, :3, :, :] = original_conv1.weight.data
        self.resnet.conv1.weight.data[:, 3:, :, :] = 0.0  # Initialize mask input weights to zero

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

        # Feature Pyramid Network (FPN)
        self.fpn = FeaturePyramidNetwork([256, 512, 1024, 2048], 256)

        # Assuming model_input_size is (H, W)
        self.model_input_size = config.model_input_size  # (256, 448)

        # Compute the downsampled sizes dynamically
        H, W = self.model_input_size
        self.mask_downsamplers = nn.ModuleDict({
            "layer1": nn.AdaptiveAvgPool2d((H // 4, W // 4)),
            "layer2": nn.AdaptiveAvgPool2d((H // 8, W // 8)),
            "layer3": nn.AdaptiveAvgPool2d((H // 16, W // 16)),
            "layer4": nn.AdaptiveAvgPool2d((H // 32, W // 32)),
        })

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Embeddings
        self.instrument_class_embedding = nn.Embedding(self.num_instruments, embed_dim)
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

        # Transformer decoders for each instrument
        self.transformer_decoders = nn.ModuleDict({
            str(instr_id): nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=decoder_hidden_dim, nhead=8),
                num_layers=num_decoder_layers
            )
            for instr_id in instrument_to_verb_classes.keys()
        })

        # Fully Connected layers per instrument for all three tasks
        self.fc_dict = nn.ModuleDict({
            str(instr_id): nn.ModuleDict({
                "verb": nn.Sequential(
                    nn.Linear(decoder_hidden_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(instrument_to_verb_classes[instr_id]))
                ),
                "target": nn.Sequential(
                    nn.Linear(decoder_hidden_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(instrument_to_target_classes[instr_id]))
                ),                
                "verbtarg": nn.Sequential(
                    nn.Linear(decoder_hidden_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, len(instrument_to_verbtarget_classes[instr_id]))
                )
            })
            for instr_id in instrument_to_verb_classes.keys()
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

        # Backbone
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
        max_output_size_verbtarg = max(fc["verbtarg"][-1].out_features for fc in self.fc_dict.values())
        max_output_size_verb = max(fc["verb"][-1].out_features for fc in self.fc_dict.values())
        max_output_size_target = max(fc["target"][-1].out_features for fc in self.fc_dict.values())

        # Per-instrument transformer decoding and FC predictions
        verbtarg_preds, verb_preds, target_preds = [], [], []
        for i in range(img.size(0)):
            instr_id = str(instrument_id[i].item())
            if instr_id in self.transformer_decoders:
                instrument_single = instrument_embed[:, i:i+1, :]
                background_memory_single = background_memory[:, i:i+1, :]
                
                decoder_output = self.transformer_decoders[instr_id](instrument_single, background_memory_single)
                decoder_output = decoder_output.squeeze(0)  # Remove sequence dim
                
                # Predict verbtarget, verb, and target **independently**
                verbtarg_pred = self.fc_dict[instr_id]["verbtarg"](decoder_output)
                verb_pred = self.fc_dict[instr_id]["verb"](decoder_output)
                target_pred = self.fc_dict[instr_id]["target"](decoder_output)

                # Pad outputs for consistency
                verbtarg_preds.append(self.pad_tensor(verbtarg_pred, max_output_size_verbtarg))
                verb_preds.append(self.pad_tensor(verb_pred, max_output_size_verb))
                target_preds.append(self.pad_tensor(target_pred, max_output_size_target))
                
            else:
                raise ValueError(f"Instrument ID {instr_id} not found in fc_dict.")

        return (
            torch.cat(verb_preds, dim=0),
            torch.cat(target_preds, dim=0),
            torch.cat(verbtarg_preds, dim=0)            
        )
