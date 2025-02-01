import torch.nn as nn
from model_utils import get_verbtarget_to_verb_and_target_matrix
import numpy as np

class MultiTaskLoss(nn.Module):
    def __init__(self, weight):
        super(MultiTaskLoss, self).__init__()
        
        if weight:
            verbtarget_to_verb_matrix, verbtarget_to_target_matrix = get_verbtarget_to_verb_and_target_matrix()
            
            weights = weight.reshape(-1, 1) # Convert to Nx1 (should be 56x1) 
            weights_verbs  = np.matmul(verbtarget_to_verb_matrix, weights)
            weights_targets  = np.matmul(verbtarget_to_target_matrix, weights)
            
            self.criterion_verb = nn.CrossEntropyLoss(weights_verbs)
            self.criterion_target = nn.CrossEntropyLoss(weights_targets)
        else:
            self.criterion_verb = nn.CrossEntropyLoss()
            self.criterion_target = nn.CrossEntropyLoss()    

    def forward(self, verb_preds, target_preds, verb_labels, target_labels):
        loss_verb = self.criterion_verb(verb_preds, verb_labels)
        loss_target = self.criterion_target(target_preds, target_labels)
        return loss_verb + loss_target
    
