import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

# Multitask ResNet
class MultiTaskResNetFPNLearnableEmbeddings(nn.Module):
    def __init__(self, 
                 num_instruments, 
                 num_verbs, 
                 num_targets,
                 embed_dim=64, 
                 decoder_hidden_dim=64, 
                 num_decoder_layers=3):
        
        super(MultiTaskResNetFPNLearnableEmbeddings, self).__init__()
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
        
                  
        # embeddings. 
        self.instrument_class_embedding =nn.Embedding(num_instruments, 64)
        self.instrument_embedding = nn.Sequential(
            nn.Linear((256*4) + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)  # Final embedding dimension
        )        
        self.full_image_embedding =  nn.Sequential(
            nn.Linear((256*4), 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)  # Final embedding dimension
        ) 
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)        

        # Fully connected layers for verbs and targets
        self.fc_verb = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_verbs)  # Output logits for verbs
        )

        self.fc_target = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_targets)  # Output logits for targets
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
        
        # for i,level in enumerate(fpn_features.keys()):
        #     print(f'{level} - shape {fpn_features[level].shape} - pool - {pooled_features[i].shape}  ')
        
             
        aggregated_features = torch.cat([feat.view(feat.size(0), -1) for feat in pooled_features], dim=1)
        # print(f'aggregated_features.shape {aggregated_features.shape}')
        
        #Encoder memory
        full_image_embed = self.full_image_embedding(aggregated_features)
        image_memory = full_image_embed.unsqueeze(0)
        
        # print(f'full_image_embed.shape {full_image_embed.shape}')
        
        # Embed the instrument ID
        instrument_class_features = self.instrument_class_embedding(instrument_id)
        instrument_with_context_features = torch.cat((aggregated_features, instrument_class_features), dim=1)
        instrument_embed = self.instrument_embedding(instrument_with_context_features).unsqueeze(0)
        
        # print(f'instrument_with_context_features.shape {instrument_with_context_features.shape}')
        # print(f'instrument_embed.shape {full_image_embed.shape}')
        # print(f'image_memory.shape {image_memory.shape}')

        decoder_output = self.transformer_decoder(instrument_embed, image_memory)
        
        action_preds = self.fc_verb(decoder_output.squeeze(0))
        target_preds = self.fc_target(decoder_output.squeeze(0))     

        return action_preds, target_preds
