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

INSTRUMENT_CLASS_TO_ID_DICT = {instrument_class: instrument_id for instrument_id, instrument_class in INSTRUMENT_ID_TO_CLASS_DICT.items()}
VERB_CLASS_TO_ID_DICT = {verb_class: verb_id for verb_id, verb_class in VERB_ID_TO_CLASS_DICT.items()}
TARGET_CLASS_TO_ID_DICT = {target_class: target_id for target_id, target_class in TARGET_ID_TO_CLASS_DICT.items()}

# Custom Dataset
class SurgicalDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True ):
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        
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
    


class PredictionDataset(Dataset):
    def __init__(self, 
                 img_dir, 
                 ann_for_second_stage_dir,
                 transform=None,
                 train_mode=True ):
        self.img_dir = img_dir
        self.ann_for_second_stage_dir = ann_for_second_stage_dir
        self.transform = transform
        self.train_mode = train_mode
        
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
        img_name, instrument_name, instance_id, _, _ = ann_for_second_stage_name_base.split(',')
        img_path = join(self.img_dir, f'{img_name}.png')
        instrument_id = int(INSTRUMENT_CLASS_TO_ID_DICT[instrument_name])-1
        
        img = Image.open(img_path).convert("RGB")  # Use PIL for transformations
        mask = Image.open(ann_for_second_stage_path).convert("L")  # Load mask as grayscale
        
        if self.transform:
            img, mask = self.transform(img, mask, self.train_mode)  # Apply custom transformatio


        return img, mask, instrument_id, instance_id, ann_for_second_stage_name_base
    