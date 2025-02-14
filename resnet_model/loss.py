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
            assert hasattr(config, "task_class_frequencies"), 'task frequencies are required'
            
            #get frequency per class, this was not necessary. It is cool, but not necessary. Wasted hours. 
            verbtarget_to_verb_matrix, verbtarget_to_target_matrix = get_verbtarget_to_verb_and_target_matrix()
            
            
            class_to_idx_zero_index = {value: int(key)-1 for key, value in verbtarget_dict.items()}
            frequency_arranged_by_index = [config.task_class_frequencies[cls] 
                                           for cls in sorted(class_to_idx_zero_index, 
                                                key=class_to_idx_zero_index.get)]
            frequency_arranged_by_index = np.array(frequency_arranged_by_index,  dtype=float).reshape(-1, 1) # make 56x1
            
            
            frequency_verbs =  np.matmul(verbtarget_to_verb_matrix,  frequency_arranged_by_index).reshape(-1)
            frequency_targets = np.matmul(verbtarget_to_target_matrix,  frequency_arranged_by_index).reshape(-1)  
            frequency_verbtargets = frequency_arranged_by_index.reshape(-1) # reshape back to 56. 
                   
                        

            loss_verbtarget_class_weights = [(1 / freq ** config.dataset_weight_scaling_factor) if freq > 0 else 0   
                                       for freq in frequency_verbtargets] 
            loss_verb_class_weights = [(1 / freq ** config.dataset_weight_scaling_factor if freq > 0 else 0)  
                                       for freq in frequency_verbs] 
            loss_target_class_weights = [(1 / freq ** config.dataset_weight_scaling_factor if freq > 0 else 0)  
                                         for freq in frequency_targets]  
            
            total_weight_verbtarget = sum(loss_verbtarget_class_weights)
            total_weight_verb = sum(loss_verb_class_weights)
            total_weight_target = sum(loss_target_class_weights)
            
            loss_verbtarget_class_weights_normalized = [weight/total_weight_verbtarget  for weight in loss_verbtarget_class_weights] 
            loss_verb_class_weights_normalized = [weight/total_weight_verb  for weight in loss_verb_class_weights] 
            loss_target_class_weights_normalized = [weight/total_weight_target  for weight in loss_target_class_weights] 
            
            print('loss_verb_class_weights_normalized')
            print(loss_verb_class_weights_normalized)
            print('loss_target_class_weights_normalized')
            print(loss_target_class_weights_normalized)
            print('loss_verbtarget_class_weights_normalized')
            print(loss_verbtarget_class_weights_normalized)
            
            loss_verbtarget_class_weights_normalized = torch.tensor(loss_verbtarget_class_weights_normalized,dtype=torch.float, device='cuda')
            loss_verb_class_weights_normalized = torch.tensor(loss_verb_class_weights_normalized,dtype=torch.float, device='cuda')
            loss_target_class_weights_normalized = torch.tensor(loss_target_class_weights_normalized,dtype=torch.float, device='cuda')
            
            self.criterion_verbtarget = nn.CrossEntropyLoss(loss_verbtarget_class_weights_normalized)
            self.criterion_verb = nn.CrossEntropyLoss(loss_verb_class_weights_normalized)
            self.criterion_target = nn.CrossEntropyLoss(loss_target_class_weights_normalized)
            
            raise ValueError('just stay')
            
        else:
            self.criterion_verb = nn.CrossEntropyLoss()
            self.criterion_target = nn.CrossEntropyLoss()    
            self.criterion_verbtarget = nn.CrossEntropyLoss()

    def forward(self, verb_preds, target_preds, verbtarget_preds, verb_labels, target_labels, verbtarget_labels):
        
        num_classes_verbtarg = verbtarget_preds.shape[1]  # Number of classes in verbtarg prediction
        num_classes_verb = verb_preds.shape[1]  # Number of classes in verb prediction
        num_classes_target = target_preds.shape[1]  # Number of classes in target prediction

        # print(f"Verbtarg Classes: {num_classes_verbtarg}, Verb Classes: {num_classes_verb}, Target Classes: {num_classes_target}")
        
        
        # print("Max verb label:", verb_labels.max().item(), "Expected max:", num_classes_verb - 1)
        # print("Max target label:", target_labels.max().item(), "Expected max:", num_classes_target - 1)
        # print("Max verbtarg label:", verbtarget_labels.max().item(), "Expected max:", num_classes_verbtarg - 1)

        assert verb_labels.max().item() < num_classes_verb, "verb_gt_ids contains out-of-range class indices!"
        assert target_labels.max().item() < num_classes_target, "target_gt_ids contains out-of-range class indices!"
        assert verbtarget_labels.max().item() < num_classes_verbtarg, "verbtarg_gt_ids contains out-of-range class indices!"
        
        
        loss_verbtarget = self.criterion_verbtarget(verbtarget_preds, verbtarget_labels)
        loss_verb = self.criterion_verb(verb_preds, verb_labels)
        loss_target = self.criterion_target(target_preds, target_labels)
        return loss_verbtarget +  loss_verb + loss_target
    
