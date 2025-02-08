import sys
sys.path.append('../')

from torch.utils.data import  Dataset
import os
from os.path import join
from PIL import Image
from utils.general.dataset_variables import TripletSegmentationVariables
from torchvision.transforms import functional as TF

INSTRUMENT_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['instrument']
VERB_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['verb']
TARGET_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['target']
VERBTARGET_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['verbtarget']

INSTRUMENT_CLASS_TO_ID_DICT = {instrument_class: instrument_id for instrument_id, instrument_class in INSTRUMENT_ID_TO_CLASS_DICT.items()}
VERB_CLASS_TO_ID_DICT = {verb_class: verb_id for verb_id, verb_class in VERB_ID_TO_CLASS_DICT.items()}
TARGET_CLASS_TO_ID_DICT = {target_class: target_id for target_id, target_class in TARGET_ID_TO_CLASS_DICT.items()}
VERBTARGET_CLASS_TO_ID_DICT = {verbtarget_class: verbtarget_id for verbtarget_id, verbtarget_class in VERBTARGET_ID_TO_CLASS_DICT.items()}


class SurgicalSingletaskDataset(Dataset):
    def __init__(self, 
                 config,
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True ):
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        self.task_name = config.task_name
        self.class_name = 'SurgicalSingletaskDataset'
        
        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir) 
        self.img_paths = os.listdir(self.img_dir) 

    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        ann_for_second_stage_name = self.ann_for_second_stage_names[idx]
        ann_for_second_stage_path = join(self.ann_for_second_stage_dir, 
                                         ann_for_second_stage_name)
 
        ann_for_second_stage_name_base = ann_for_second_stage_name.split('.')[0]
        img_name, instrument_name, instance_id, verb_name, target_name = ann_for_second_stage_name_base.split(',')
        img_path = join(self.img_dir, f'{img_name}.png')
        
        if self.task_name == 'verb':
            task_id = int(VERB_CLASS_TO_ID_DICT[verb_name])-1
        elif self.task_name == 'target': 
            task_id = int(TARGET_CLASS_TO_ID_DICT[target_name])-1
        elif self.task_name == 'verbtarget': 
            task_id = int(VERBTARGET_CLASS_TO_ID_DICT[f'{verb_name},{target_name}'])-1               
        
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name])-1
        
        img = Image.open(img_path).convert("RGB")  # Use PIL for transformations
        mask = Image.open(ann_for_second_stage_path).convert("L")  # Load mask as grayscale
        
        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  # Apply custom transformation


        return img, mask, instrument_id, instance_id, task_id, ann_for_second_stage_name_base
    
    
class SurgicalSingletaskDataset(Dataset):
    def __init__(self, 
                 config,
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True ):
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        self.task_name = config.task_name
        self.class_name = 'SurgicalSingletaskDataset'
        
        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir) 
        self.img_paths = os.listdir(self.img_dir) 

    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        ann_for_second_stage_name = self.ann_for_second_stage_names[idx]
        ann_for_second_stage_path = join(self.ann_for_second_stage_dir, 
                                         ann_for_second_stage_name)
 
        ann_for_second_stage_name_base = ann_for_second_stage_name.split('.')[0]
        img_name, instrument_name, instance_id, verb_name, target_name = ann_for_second_stage_name_base.split(',')
        img_path = join(self.img_dir, f'{img_name}.png')
        
        if self.task_name == 'verb':
            task_id = int(VERB_CLASS_TO_ID_DICT[verb_name])-1
        elif self.task_name == 'target': 
            task_id = int(TARGET_CLASS_TO_ID_DICT[target_name])-1
        elif self.task_name == 'verbtarget': 
            task_id = int(VERBTARGET_CLASS_TO_ID_DICT[f'{verb_name},{target_name}'])-1               
        
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name])-1
        
        img = Image.open(img_path).convert("RGB")  # Use PIL for transformations
        mask = Image.open(ann_for_second_stage_path).convert("L")  # Load mask as grayscale
        
        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  # Apply custom transformation


        return img, mask, instrument_id, instance_id, task_id, ann_for_second_stage_name_base    
    

import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class SurgicalSingletaskDatasetForParallelFCLayers(Dataset):
    def __init__(self, 
                 config,
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True):
        
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        self.task_name = config.task_name
        self.class_name = 'SurgicalSingletaskDatasetForParallelFCLayers'

        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir) 
        self.img_paths = os.listdir(self.img_dir) 
        

        # Store instrument-specific mappings
        self.instrument_to_task_classes = config.instrument_to_task_classes
        self.task_class_mappings = self._generate_task_class_mappings()      

    def _generate_task_class_mappings(self):
        """
        Creates a mapping from (instrument, global task class) -> local task class index.
        """
        task_mappings = {}
        for instr_id, valid_task_classes in self.instrument_to_task_classes.items():
            for local_idx, global_task_id in enumerate(valid_task_classes):
                task_mappings[(instr_id, global_task_id)] = local_idx
        return task_mappings

    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        ann_name = self.ann_for_second_stage_names[idx]
        ann_path = os.path.join(self.ann_for_second_stage_dir, ann_name)

        # Extract components from filename
        ann_base = ann_name.split('.')[0]
        img_name, instrument_name, instance_id, verb_name, target_name = ann_base.split(',')
        img_path = os.path.join(self.img_dir, f'{img_name}.png')

        # Convert instrument name to instrument ID
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name]) - 1

        # Get global task ID
        if self.task_name == 'verb':
            global_task_id = int(VERB_CLASS_TO_ID_DICT[verb_name]) - 1
        elif self.task_name == 'target': 
            global_task_id = int(TARGET_CLASS_TO_ID_DICT[target_name]) - 1
        elif self.task_name == 'verbtarget': 
            global_task_id = int(VERBTARGET_CLASS_TO_ID_DICT[f'{verb_name},{target_name}']) - 1   

        # Map to local task ID
        local_task_id = self.task_class_mappings.get((instrument_id, global_task_id), None)

        # Handle missing mappings (optional: log instead of raising error)
        if local_task_id is None:
            raise ValueError(f"{self.task_name} {global_task_id} is not valid for instrument {instrument_id}") 

        # Load images & masks
        img = Image.open(img_path).convert("RGB")  
        mask = Image.open(ann_path).convert("L")  

        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  

        return img, mask, instrument_id, instance_id, local_task_id, ann_base



class SurgicalMultitaskDataset(Dataset):
    def __init__(self,                  
                 config,
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True ):
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        self.class_name = 'SurgicalMultitaskDataset'
        
        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir) 
        self.img_paths = os.listdir(self.img_dir) 
        
        

    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        ann_for_second_stage_name = self.ann_for_second_stage_names[idx]
        ann_for_second_stage_path = join(self.ann_for_second_stage_dir, 
                                         ann_for_second_stage_name)
 
        ann_for_second_stage_name_base = ann_for_second_stage_name.split('.')[0]
        img_name, instrument_name, instance_id, verb_name, target_name = ann_for_second_stage_name_base.split(',')
        img_path = join(self.img_dir, f'{img_name}.png')
        verb_id =  int(VERB_CLASS_TO_ID_DICT[verb_name])-1
        target_id = int(TARGET_CLASS_TO_ID_DICT[target_name])-1
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name])-1
        
        img = Image.open(img_path).convert("RGB")  # Use PIL for transformations
        mask = Image.open(ann_for_second_stage_path).convert("L")  # Load mask as grayscale
        
        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  # Apply custom transformation


        return img, mask, instrument_id, instance_id, verb_id, target_id, ann_for_second_stage_name_base




class SurgicalThreetaskDatasetForParallelLayers(Dataset):
    def __init__(self, 
                 config,
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True):
        
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        self.class_name = 'SurgicalThreetaskDatasetForParallelLayers'

        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir)  
        self.img_paths = os.listdir(self.img_dir)  

        # Instrument-specific task mappings
        self.instrument_id_to_verb_classes = config.instrument_to_verb_classes
        self.instrument_id_to_target_classes = config.instrument_to_target_classes
        self.instrument_id_to_verbtarget_classes = config.instrument_to_verbtarget_classes
        
        self.verb_class_mappings = self._generate_task_class_mappings(self.instrument_id_to_verb_classes)
        self.target_class_mappings = self._generate_task_class_mappings(self.instrument_id_to_target_classes )
        self.verbtarget_class_mappings = self._generate_task_class_mappings(self.instrument_id_to_verbtarget_classes )
        
        print('self.verb_class_mappings', self.verb_class_mappings)

    def _generate_task_class_mappings(self, instrument_id_to_task_classes):
        """
        Creates a mapping from (instrument, global task class) -> local task class index.
        """
        task_mappings = {}
        for instr_id, valid_task_classes in instrument_id_to_task_classes.items():
            for local_idx, global_task_id in enumerate(valid_task_classes):
                task_mappings[(instr_id, global_task_id)] = local_idx
        return task_mappings

    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        ann_name = self.ann_for_second_stage_names[idx]
        ann_path = os.path.join(self.ann_for_second_stage_dir, ann_name)

        # Extract components from filename
        ann_base = ann_name.split('.')[0]
        img_name, instrument_name, instance_id, verb_name, target_name = ann_base.split(',')
        img_path = os.path.join(self.img_dir, f'{img_name}.png')

        # Convert instrument name to instrument ID
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name]) - 1

        # Get global task IDs for verb, target, and verbtarget
        verb_id = int(VERB_CLASS_TO_ID_DICT[verb_name]) - 1
        target_id = int(TARGET_CLASS_TO_ID_DICT[target_name]) - 1
        verbtarget_id = int(VERBTARGET_CLASS_TO_ID_DICT[f'{verb_name},{target_name}']) - 1

        # Map global task IDs to local task IDs (if applicable)
        verb_local_id = self.verb_class_mappings.get((instrument_id, verb_id), None)
        target_local_id = self.target_class_mappings.get((instrument_id, target_id), None)
        verbtarget_local_id = self.verbtarget_class_mappings.get((instrument_id, verbtarget_id), None)

        # Handle missing mappings (optional: log instead of raising error)
        if verb_local_id is None:
            raise ValueError(f"Verb ID {verb_id} is not valid for instrument {instrument_id}")
        if target_local_id is None:
            raise ValueError(f"Target ID {target_id} is not valid for instrument {instrument_id}")
        if verbtarget_local_id is None:
            raise ValueError(f"VerbTarget ID {verbtarget_id} is not valid for instrument {instrument_id}")

        # Load images & masks
        img = Image.open(img_path).convert("RGB")  
        mask = Image.open(ann_path).convert("L")  

        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  

        return img, mask, instrument_id, instance_id, verb_local_id, target_local_id, verbtarget_local_id, ann_base













########################################################Prediction Datasets. There is only one. ##############################    

class PredictionDataset(Dataset):
    def __init__(self, 
                 config,
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True ):
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        self.class_name = 'PredictionDataset'
        self.task_name = config.task_name
        
        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir) 
        self.img_paths = os.listdir(self.img_dir) 
        
    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        # Get mask file name
        ann_for_second_stage_name = self.ann_for_second_stage_names[idx]
        ann_for_second_stage_path = join(self.ann_for_second_stage_dir, 
                                         ann_for_second_stage_name)
 
        ann_for_second_stage_name_base = ann_for_second_stage_name.split('.')[0]
        img_name, instrument_name, instance_id, verb_name, target_name = ann_for_second_stage_name_base.split(',')
        
        if self.task_name == 'verb':
            ground_truth_name = verb_name
        elif self.task_name == 'target':
            ground_truth_name = target_name 
        elif self.task_name == 'verbtarget':
            if verb_name and target_name: 
                ground_truth_name = f'{verb_name},{target_name}'
            else:
                ground_truth_name = None    
        elif self.task_name == 'threetask':
            if verb_name and target_name: 
                ground_truth_name = f'{verb_name},{target_name}'
            else:
                ground_truth_name = None            
        
        img_path = join(self.img_dir, f'{img_name}.png')
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name])-1
                
        img = Image.open(img_path).convert("RGB")  # Use PIL for transformations
        mask = Image.open(ann_for_second_stage_path).convert("L")  # Load mask as grayscale
        
        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  # Apply custom transformation


        return img, mask, instrument_id, instance_id, ann_for_second_stage_name_base, ground_truth_name            
    
    

    