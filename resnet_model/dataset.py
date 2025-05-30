import sys
sys.path.append('../')

from torch.utils.data import  Dataset
import os
from os.path import join
from PIL import Image
from utils.general.dataset_variables import TripletSegmentationVariables
from torchvision.transforms import functional as TF
from utils.general.read_files import read_from_json
import os
import torch
from torch.utils.data import Dataset
import numpy as np

INSTRUMENT_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['instrument']
VERB_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['verb']
TARGET_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['target']
VERBTARGET_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['verbtarget']
TRIPLET_ID_TO_CLASS_DICT = TripletSegmentationVariables.categories['triplet']

INSTRUMENT_CLASS_TO_ID_DICT = {instrument_class: instrument_id for instrument_id, instrument_class in INSTRUMENT_ID_TO_CLASS_DICT.items()}
VERB_CLASS_TO_ID_DICT = {verb_class: verb_id for verb_id, verb_class in VERB_ID_TO_CLASS_DICT.items()}
TARGET_CLASS_TO_ID_DICT = {target_class: target_id for target_id, target_class in TARGET_ID_TO_CLASS_DICT.items()}
VERBTARGET_CLASS_TO_ID_DICT = {verbtarget_class: verbtarget_id for verbtarget_id, verbtarget_class in VERBTARGET_ID_TO_CLASS_DICT.items()}
TRIPLET_CLASS_TO_ID_DICT = {ivt_class: ivt_id for ivt_id, ivt_class in TRIPLET_ID_TO_CLASS_DICT.items()}

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



class SurgicalFourTaskDatasetWithDetectorLogits(Dataset):
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
        self.class_name = 'SurgicalFourTaskDatasetWithDetectorLogits'

        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir)  
        self.img_paths = os.listdir(self.img_dir)  
     

    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        ann_name = self.ann_for_second_stage_names[idx]
        ann_path = os.path.join(self.ann_for_second_stage_dir, ann_name)

        # Extract groundtruth components from filename
        ann_base = ann_name.split('.')[0]
        img_name, instrument_name, gt_instance_id, verb_name, target_name = ann_base.split(',')
        img_path = os.path.join(self.img_dir, f'{img_name}.png')
        
        # Convert instrument name to instrument ID
        gt_instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name]) - 1
        instrument_softmax = torch.zeros(6, dtype=torch.float32)
        instrument_softmax[gt_instrument_id] = 1.0

        # Get global task IDs for verb, target, and verbtarget
        gt_verb_id = int(VERB_CLASS_TO_ID_DICT[verb_name]) - 1
        gt_target_id = int(TARGET_CLASS_TO_ID_DICT[target_name]) - 1
        gt_verbtarget_id = int(VERBTARGET_CLASS_TO_ID_DICT[f'{verb_name},{target_name}']) - 1
        gt_triplet_id = int(TRIPLET_CLASS_TO_ID_DICT[f'{instrument_name},{verb_name},{target_name}']) - 1

        
        # Load images & masks
        img = Image.open(img_path).convert("RGB")  
        mask = Image.open(ann_path).convert("L")  

        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  

        return img, mask, ann_base, gt_instance_id, gt_instrument_id, gt_verb_id, gt_target_id, gt_verbtarget_id, gt_triplet_id, instrument_softmax


#### With target logits  

class SurgicalSingletaskDatasetWithTargetLogits(Dataset):
    def __init__(self, 
                 config, 
                 transform=None, 
                 split='train'):
        
        self.config = config
        self.transform = transform
        self.split = split
        self.task_name = config.task_name
        
        
        # Dynamically set directories
        if split == 'train':
            self.img_dir = config.train_image_dir
            self.ann_for_second_stage_dir = config.train_ann_dir
            self.target_logits_dir = config.target_logits_dir_train
        elif split == 'val':
            self.img_dir = config.val_image_dir
            self.ann_for_second_stage_dir = config.val_ann_dir
            self.target_logits_dir = config.target_logits_dir_val
        elif split == 'test':
            self.img_dir = config.test_image_dir
            self.ann_for_second_stage_dir = config.test_ann_dir
            self.target_logits_dir = config.target_logits_dir_test
        else:
            raise ValueError(f"Unknown split: {split}")

        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir) 
        self.img_paths = os.listdir(self.img_dir) 

    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        
        ann_for_second_stage_name = self.ann_for_second_stage_names[idx]
        ann_for_second_stage_path = join(self.ann_for_second_stage_dir, ann_for_second_stage_name)
        
        ann_for_second_stage_name_base = ann_for_second_stage_name.split('.')[0]
        img_name, instrument_name, instance_id, verb_name, target_name = ann_for_second_stage_name_base.split(',')
        img_path = join(self.img_dir, f'{img_name}.png')
        
        # Load image and mask
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_for_second_stage_path).convert("L")
        
        # Load target logits (.npz)
        logits_file_name = f"{img_name}_logits.npz"
        logits_file_path = join(self.target_logits_dir, logits_file_name)
        logits_data = np.load(logits_file_path)
        target_logits = logits_data['logits']  # Updated key to 'logits'
        target_logits_tensor = torch.tensor(target_logits, dtype=torch.float16)  # Convert to torch tensor
        
        # Get labels
        if self.task_name == 'verb':
            task_id = int(VERB_CLASS_TO_ID_DICT[verb_name]) - 1
        elif self.task_name == 'target': 
            task_id = int(TARGET_CLASS_TO_ID_DICT[target_name]) - 1
        elif self.task_name == 'verbtarget': 
            task_id = int(VERBTARGET_CLASS_TO_ID_DICT[f'{verb_name},{target_name}']) - 1
        
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name]) - 1
        
        # Apply transform (if any) - make sure to handle target logits later
        if self.transform:
            img, mask, target_logits_tensor = self.transform(img, mask, target_logits_tensor, self.split)
        
        return img, mask, target_logits_tensor, instrument_id, instance_id, task_id, ann_for_second_stage_name_base







#######################################################Prediction Datasets. ##############################    

class PredictionDataset(Dataset):
    def __init__(self, 
                 config, 
                 transform=None, 
                 split='train'):
        
        self.config = config
        self.transform = transform
        self.split = split
        
        self.class_name = 'PredictionDataset'
        self.task_name = config.task_name
        self.architecture = config.architecture
        
        
        if split == 'train':
            self.img_dir = config.train_image_dir
            self.ann_for_second_stage_dir = config.train_ann_dir
            self.target_logits_dir = config.target_logits_dir_train
        elif split == 'val':
            self.img_dir = config.val_image_dir
            self.ann_for_second_stage_dir = config.val_ann_dir
            self.target_logits_dir = config.target_logits_dir_val
        elif split == 'test':
            self.img_dir = config.test_image_dir
            self.ann_for_second_stage_dir = config.test_ann_dir
            self.target_logits_dir = config.target_logits_dir_test
        else:
            raise ValueError(f"Unknown split: {split}")

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
        elif self.task_name == 'standard_multitask_verb_and_target':
            if verb_name and target_name: 
                ground_truth_name = f'{verb_name},{target_name}'
            else:
                ground_truth_name = None                  
        
        img_path = join(self.img_dir, f'{img_name}.png')
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name])-1
                
        img = Image.open(img_path).convert("RGB")  # Use PIL for transformations
        mask = Image.open(ann_for_second_stage_path).convert("L")  # Load mask as grayscale
        
        # Load target logits (.npz)
        if self.architecture == 'singletaskwithtargetlogits':
            logits_file_name = f"{img_name}_logits.npz"
            logits_file_path = join(self.target_logits_dir, logits_file_name)
            logits_data = np.load(logits_file_path)
            target_logits = logits_data['logits']  # Updated key to 'logits'
            target_logits_tensor = torch.tensor(target_logits, dtype=torch.float16)  # Convert to torch tensor
        
        
            if self.transform:
                img, mask, target_logits_tensor = self.transform(img, mask, target_logits_tensor, self.split)

            return img, mask, target_logits_tensor, instrument_id, instance_id, ann_for_second_stage_name_base, ground_truth_name   
        else: 
            if self.transform:
                img, mask = self.transform(img, mask, self.split)  # Apply custom transformation


            return img, mask, instrument_id, instance_id, ann_for_second_stage_name_base, ground_truth_name 
                     
    
    

class PredictionDatasetSofmaxInputs(Dataset):
    def __init__(self, 
                 config,
                 img_dir, 
                 ann_for_second_stage_dir,
                 detector_softmax_scores_json,
                 transform=None,
                 train_mode=True ):
        # Supposed to handle prediction mainly. 
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        self.class_name = 'PredictionDatasetSofmaxInputs'
        self.task_name = config.task_name
        
        self.ann_for_second_stage_names = os.listdir(self.ann_for_second_stage_dir)       
        self.detector_softmax_scores = read_from_json(detector_softmax_scores_json) #for training
        
    def __len__(self):
        return len(self.ann_for_second_stage_names)

    def __getitem__(self, idx):
        # Get mask file name
        ann_name = self.ann_for_second_stage_names[idx]
        ann_path = join(self.ann_for_second_stage_dir, ann_name)
 
        base_name = ann_name.split('.')[0]
        img_name, _, instance_id = base_name.rpartition('_instance_')
                
        img_path = join(self.img_dir, f"{img_name}.png")
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(ann_path).convert("L")
        
                                
        
        # Get softmax vector
        if img_name in self.detector_softmax_scores and self.detector_softmax_scores[img_name] is not None:
            instance_softmax_entries = self.detector_softmax_scores[img_name]
            softmax_score = None
            for entry in instance_softmax_entries:
                if str(entry["instance_id"]) == str(instance_id):
                    softmax_score = torch.tensor(entry["softmax"], dtype=torch.float32)
                    break
        else:
            raise ValueError('We could not find softmax score.')

        
        if softmax_score is None:
            raise ValueError(f"Softmax vector not found for instance_id={instance_id} in img={img_name}")    
        
        
        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)

        return img, mask, softmax_score, instance_id, base_name  
    
    # ground_truth_name currently not needed as there is no ground_truth for the test. 
    