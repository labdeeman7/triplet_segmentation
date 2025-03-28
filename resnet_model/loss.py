import sys
sys.path.append('../')

import torch.nn as nn
import torch
from model_utils import get_verbtarget_to_verb_and_target_matrix
from utils.general.dataset_variables import TripletSegmentationVariables
import numpy as np
import pprint
import torch.nn.functional as F


verb_dict =  TripletSegmentationVariables.categories['verb']
target_dict = TripletSegmentationVariables.categories['target']
verbtarget_dict = TripletSegmentationVariables.categories['verbtarget']

class MultiTaskLoss(nn.Module):
    def __init__(self, config):
        super(MultiTaskLoss, self).__init__()
        if config.use_wce:
            print('using weighted cross entropy')
            assert hasattr(config, "task_class_frequencies"), 'task frequencies are required'
            #get frequency per class, this was not necessary. It is cool, but not necessary. Wasted hours. 
            verbtarget_to_verb_matrix, verbtarget_to_target_matrix = get_verbtarget_to_verb_and_target_matrix()
            
            
            class_to_idx_zero_index = {value: int(key)-1 for key, value in verbtarget_dict.items()}
            frequency_arranged_by_index = [config.task_class_frequencies[cls] 
                                           for cls in sorted(class_to_idx_zero_index, 
                                                key=class_to_idx_zero_index.get)]
            frequency_arranged_by_index = np.array(frequency_arranged_by_index,  dtype=float).reshape(-1, 1) # make 56x1
            
            
            frequency_arranged_by_index_verbs =  np.matmul(verbtarget_to_verb_matrix,  frequency_arranged_by_index).reshape(-1)
            frequency_arranged_by_index_targets = np.matmul(verbtarget_to_target_matrix,  frequency_arranged_by_index).reshape(-1)  
                        
             
            loss_verb_class_weights = [(1 / freq ** config.dataset_weight_scaling_factor) if freq > 0 else 0    
                                       for freq in frequency_arranged_by_index_verbs] 
            loss_target_class_weights = [(1 / freq ** config.dataset_weight_scaling_factor) if freq > 0 else 0    
                                       for freq in frequency_arranged_by_index_targets] 
            
            total_weight_verb = sum(loss_verb_class_weights)
            total_weight_target = sum(loss_target_class_weights)
            
            loss_verb_class_weights_normalized = [weight/total_weight_verb  for weight in loss_verb_class_weights] 
            loss_target_class_weights_normalized = [weight/total_weight_target  for weight in loss_target_class_weights] 
            
            print('loss_verb_class_weights_normalized')
            print(loss_verb_class_weights_normalized)
            print('loss_target_class_weights_normalized')
            print(loss_target_class_weights_normalized)
            
            loss_verb_class_weights_normalized = torch.tensor(loss_verb_class_weights_normalized,dtype=torch.float, device='cuda')
            loss_target_class_weights_normalized = torch.tensor(loss_target_class_weights_normalized,dtype=torch.float, device='cuda')
            
            
            self.criterion_verb = nn.CrossEntropyLoss(loss_verb_class_weights_normalized)
            self.criterion_target = nn.CrossEntropyLoss(loss_target_class_weights_normalized)
            

            
        else:
            print('using normal cross entropy')
            self.criterion_verb = nn.CrossEntropyLoss()
            self.criterion_target = nn.CrossEntropyLoss()    

    def forward(self, verb_preds, target_preds, verb_labels, target_labels):
        loss_verb = self.criterion_verb(verb_preds, verb_labels)
        loss_target = self.criterion_target(target_preds, target_labels)
        return loss_verb + loss_target



class MultiTaskLossThreeTasks(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossThreeTasks, self).__init__()
        if config.use_wce:
            pass
        else:
            self.criterion_verb = nn.CrossEntropyLoss()
            self.criterion_target = nn.CrossEntropyLoss()    
            self.criterion_verbtarget = nn.CrossEntropyLoss()
            
            self.verb_multitask_weight = 1.0 if not hasattr(config, "verb_multitask_weight") else config.verb_multitask_weight
            self.target_multitask_weight = 1.0 if not hasattr(config, "target_multitask_weight") else config.target_multitask_weight
            self.verbtarget_multitask_weight = 1.0 if not hasattr(config, "verbtarget_multitask_weight") else config.verbtarget_multitask_weight
            

    def forward(self, verb_preds, target_preds, verbtarget_preds, verb_labels, target_labels, verbtarget_labels):
        
        loss_verb = self.criterion_verb(verb_preds, verb_labels)
        loss_target = self.criterion_target(target_preds, target_labels)
        loss_verbtarget = self.criterion_verbtarget(verbtarget_preds, verbtarget_labels)
        
        return (self.verb_multitask_weight*loss_verb) + (self.target_multitask_weight*loss_target) + (self.verbtarget_multitask_weight*loss_verbtarget)  
    


class MultiTaskLossFourTasks(nn.Module):
    def __init__(self, config):
        super(MultiTaskLossFourTasks, self).__init__()
        if config.use_wce:
            pass
        else:
            self.criterion_verb = nn.CrossEntropyLoss()
            self.criterion_target = nn.CrossEntropyLoss()    
            self.criterion_verbtarget = nn.CrossEntropyLoss()
            self.criterion_ivt = nn.CrossEntropyLoss()
            
            
            self.verb_multitask_weight = 1.0 if not hasattr(config, "verb_multitask_weight") else config.verb_multitask_weight
            self.target_multitask_weight = 1.0 if not hasattr(config, "target_multitask_weight") else config.target_multitask_weight
            self.verbtarget_multitask_weight = 1.0 if not hasattr(config, "verbtarget_multitask_weight") else config.verbtarget_multitask_weight
            self.ivt_multitask_weight = 1.0 if not hasattr(config, "ivt_multitask_weight") else config.verbtarget_multitask_weight
            

    def forward(self, verb_preds, target_preds, verbtarget_preds, ivt_preds, verb_labels, target_labels, verbtarget_labels, ivt_labels):
        
        loss_verb = self.criterion_verb(verb_preds, verb_labels)
        loss_target = self.criterion_target(target_preds, target_labels)
        loss_verbtarget = self.criterion_verbtarget(verbtarget_preds, verbtarget_labels)
        loss_ivt = self.criterion_ivt(ivt_preds, ivt_labels)
        
        total_loss = (self.verb_multitask_weight*loss_verb) + (self.target_multitask_weight*loss_target) + (self.verbtarget_multitask_weight*loss_verbtarget) + (self.ivt_multitask_weight*loss_ivt)
        
        return  total_loss, {"loss_verb": loss_verb.item(),
                            "loss_target": loss_target.item(),
                            "loss_verbtarget": loss_verbtarget.item(),
                            "loss_ivt": loss_ivt.item() }     
    
