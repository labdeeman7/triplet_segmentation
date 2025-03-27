import torch
import torch.nn as nn
from torchvision import models
from torchvision.ops import FeaturePyramidNetwork
from collections import OrderedDict

class FourTaskResNetFPNWithMoEDecodersAndSoftmaxInputs(nn.Module):
    def __init__(self, 
                 config,
                 instrument_to_verb_classes, 
                 instrument_to_target_classes, 
                 instrument_to_verbtarget_classes, 
                 instrument_to_triplet_classes,
                 num_instruments=6,
                 num_verbs=10,
                 num_targets=15,
                 num_verbtargets=56,
                 num_triplets=100, 
                 embed_dim=64, 
                 decoder_hidden_dim=256, 
                 num_decoder_layers=1):
        super(FourTaskResNetFPNWithMoEDecodersAndSoftmaxInputs, self).__init__()

        self.num_instruments = num_instruments  # Number of instruments
        self.num_verbs = num_verbs  # Number of instruments
        self.num_targets = num_targets  # Number of instruments
        self.num_verbtargets = num_verbtargets  # Number of instruments
        self.num_instrumetverbtargets = num_triplets  # Number of instruments
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
        self.instrument_class_embedding = nn.Embedding(self.num_instruments, embed_dim) # Not sure if this is correct.
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
            for instr_id in range(self.num_instruments)
        })
        
        verb_proj = self.build_all_projection_matrices(instrument_to_verb_classes, num_verbs)
        target_proj = self.build_all_projection_matrices(instrument_to_target_classes, num_targets)
        verbtarg_proj = self.build_all_projection_matrices(instrument_to_verbtarget_classes, num_verbtargets) 
        ivt_proj = self.build_all_projection_matrices(instrument_to_triplet_classes, num_triplets)  # IVT = 100 fixed
        
        for instr_id in range(self.num_instruments):
            self.register_buffer(f"verb_proj_{instr_id}", verb_proj[instr_id] )
            self.register_buffer(f"target_proj_{instr_id}", target_proj[instr_id] )
            self.register_buffer(f"verbtarg_proj_{instr_id}", verbtarg_proj[instr_id] )
            self.register_buffer(f"ivt_proj_{instr_id}", ivt_proj[instr_id] )
            


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
            for instr_id in range(self.num_instruments)
        })

    
    def create_projection_matrix(self, local_ids, global_size):
        """
        Returns a matrix of shape (len(local_ids), global_size)
        that maps local logits to their global positions.
        """
        num_local = len(local_ids)
        proj_matrix = torch.zeros(num_local, global_size, dtype=torch.float32)
        for local_idx, global_idx in enumerate(local_ids):
            proj_matrix[local_idx, global_idx] = 1.0
        return proj_matrix  # shape (num_local, global_size)
    
    
    def build_all_projection_matrices(self, instr_to_classes_dict, global_size):
        proj_dict = {}
        for instr_id, global_ids in instr_to_classes_dict.items():
            proj_dict[instr_id] = self.create_projection_matrix(global_ids, global_size)
        return proj_dict

 
        
    def pad_tensor(self, tensor, max_size):
        """
        Pads tensor with -inf values so all outputs have the same shape.
        """
        pad_size = max_size - tensor.size(1)
        if pad_size > 0:
            padding = torch.full((tensor.size(0), pad_size), float('-inf'), device=tensor.device)
            return torch.cat((tensor, padding), dim=1)
        return tensor  

    def forward(self, img, mask, instrument_softmax):
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
        
        # Soft-instrument embedding: use softmax over embedding table
        instrument_class_embeds = self.instrument_class_embedding.weight  # (6, embed_dim)
        instrument_class_features = torch.matmul(instrument_softmax, instrument_class_embeds) # (B, embed_dim)


        # Instrument embedding        
        instrument_with_class_context_features = torch.cat((aggregated_instrument_features, instrument_class_features), dim=1)
        instrument_embed = self.instrument_embedding(instrument_with_class_context_features).unsqueeze(0)

        # Normalize embeddings
        instrument_embed = instrument_embed / (instrument_embed.norm(dim=-1, keepdim=True) + 1e-6)
        background_memory = background_memory / (background_memory.norm(dim=-1, keepdim=True) + 1e-6)
        
        # Store all global logits from each expert
        final_verb_logits = []
        final_target_logits = []
        final_verbtarg_logits = []
        final_ivt_logits = []
        
        for i in range(img.size(0)):  # Loop over each sample in the batch
            instrument_embed_i = instrument_embed[:, i:i+1, :]         # (1, 1, D)
            background_memory_i = background_memory[:, i:i+1, :]       # (1, 1, D)

            verb_logits_i = torch.zeros(self.num_verbs, device=img.device)
            target_logits_i = torch.zeros(self.num_targets, device=img.device)
            verbtarg_logits_i = torch.zeros(self.num_verbtargets, device=img.device)
            ivt_logits_i = torch.zeros(self.num_instrumetverbtargets, device=img.device) 
            
            for instr_id in range(self.num_instruments):  
                decoder = self.transformer_decoders[str(instr_id)]
                decoder_out = decoder(instrument_embed_i, background_memory_i)
                decoder_out = decoder_out.view(-1) 
                
                # FC heads → (num_local,)
                verb_local = self.fc_dict[str(instr_id)]["verb"](decoder_out)
                target_local = self.fc_dict[str(instr_id)]["target"](decoder_out)
                verbtarg_local = self.fc_dict[str(instr_id)]["verbtarg"](decoder_out)  
                
                
                # Projection matrices (num_local, global)
                verb_proj = getattr(self, f"verb_proj_{instr_id}")     # (num_local, num_verbs)
                target_proj = getattr(self, f"target_proj_{instr_id}")
                verbtarg_proj = getattr(self, f"verbtarg_proj_{instr_id}")
                ivt_proj = getattr(self, f"ivt_proj_{instr_id}")       # IVT = 100

                # Project to global space
                verb_global = torch.matmul(verb_local, verb_proj)   # (num_verbs,)
                target_global = torch.matmul(target_local, target_proj)  
                verbtarg_global = torch.matmul(verbtarg_local, verbtarg_proj) 
                ivt_global = torch.matmul(verbtarg_local , ivt_proj)    
                
                # Weight by softmax score for this sample
                weight = instrument_softmax[i, instr_id]  
                
                verb_logits_i += weight * verb_global
                target_logits_i += weight * target_global
                verbtarg_logits_i += weight * verbtarg_global
                ivt_logits_i += weight * ivt_global
            
            # Append final logits for sample i
            final_verb_logits.append(verb_logits_i)
            final_target_logits.append(target_logits_i)
            final_verbtarg_logits.append(verbtarg_logits_i)
            final_ivt_logits.append(ivt_logits_i)    
            
        # Stack results: (B, num_classes)
        final_verb_logits = torch.stack(final_verb_logits, dim=0)
        final_target_logits = torch.stack(final_target_logits, dim=0)
        final_verbtarg_logits = torch.stack(final_verbtarg_logits, dim=0)
        final_ivt_logits = torch.stack(final_ivt_logits, dim=0)    
            
        return final_verb_logits, final_target_logits, final_verbtarg_logits, final_ivt_logits    
            
            
            
       
