import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

# Multitask ResNet
class SingleTaskResNetFPNTransformerDecoder(nn.Module):
    def __init__(self,
                 config, 
                 num_instruments, 
                 num_task_class,
                 embed_dim=64, 
                 decoder_hidden_dim=64, 
                 num_decoder_layers=3):
        
        super(SingleTaskResNetFPNTransformerDecoder, self).__init__()
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
        
        # Global pooling layer to aggregate FPN features
        self.global_pool = nn.AdaptiveAvgPool2d(1)  
        
                  
        # embeddings. 
        self.instrument_class_embedding =nn.Embedding(num_instruments, 64)
        self.instrument_embedding = nn.Sequential(
            nn.Linear((256*4) + embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim)  # Final embedding dimension
        )        
        self.background_image_embedding =  nn.Sequential(
            nn.Linear((256*4), 128),
            nn.ReLU(),
            nn.Linear(128, decoder_hidden_dim)  # directly predicting from target information. 
        ) 
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=decoder_hidden_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)        

        # Fully connected layers for verbs and targets
        self.fc_task = nn.Sequential(
            nn.Linear(decoder_hidden_dim, 256),
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
        
        instrument_features = []
        background_features = []
        # Soft binary masks
        for level, downsampler in self.mask_downsamplers.items():
            fpn_feature = fpn_features[level]
            soft_mask = downsampler(mask) # Downsampled soft mask
            
            instrument_feature = fpn_feature * soft_mask
            background_feature = fpn_feature * (1 - soft_mask)
            
            instrument_features.append(self.global_pool(instrument_feature))
            background_features.append(self.global_pool(background_feature))        
       
        # Aggregated embeddings      
        aggregated_instrument_features = torch.cat([feat.view(feat.size(0), -1) for feat in instrument_features], dim=1)
        aggregated_background_features = torch.cat([feat.view(feat.size(0), -1) for feat in background_features], dim=1)
          
        #Background information
        background_embed = self.background_image_embedding(aggregated_background_features) 
        background_memory = background_embed.unsqueeze(0)
                
        # Embed the instrument ID
        instrument_class_features = self.instrument_class_embedding(instrument_id)
        instrument_with_class_context_features = torch.cat((aggregated_instrument_features, instrument_class_features), dim=1)
        instrument_embed = self.instrument_embedding(instrument_with_class_context_features).unsqueeze(0)
        
        #Normalize before transformer
        instrument_embed = instrument_embed / (instrument_embed.norm(dim=-1, keepdim=True) + 1e-6)
        background_memory = background_memory / (background_memory.norm(dim=-1, keepdim=True) + 1e-6)

        # Decoder for task prediction.
        decoder_output = self.transformer_decoder(instrument_embed, background_memory)   
        
        #predict the task     
        task_preds = self.fc_task(decoder_output.squeeze(0))
       
        
        return task_preds
