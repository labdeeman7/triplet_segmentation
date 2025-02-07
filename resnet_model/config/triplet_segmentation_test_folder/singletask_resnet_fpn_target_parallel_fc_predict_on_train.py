from os.path import join
import os


# Hyperparameters
batch_size = 64
num_epochs = 20
learning_rate = 0.0005

# Dynamically set the experiment name from the filename
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]

# Model name
model_name = 'SingleTaskResNetFPNForParallelFCLayers'
task_name = 'target'

description =  'predicting target prediction parrallel FC on train'


if model_name in ['SingleTaskResNetFPN', 'SingleTaskResNetFPNWithDecoder']:
    architecture = 'singletask'
elif  model_name in ['MultiTaskResNet', 'MultiTaskResNetFPN', 'MultiTaskResNetFPNTransformerDecoder']:  
    architecture = 'multitask'
elif model_name in ['SingleTaskResNetFPNForParallelFCLayers']:
    architecture = 'singletask_parrallel_fc'
    
     
  

# Dataset Directories
if os.name == 'posix':  # Unix-like systems (Linux, macOS)
    dataset_path = '/nfs/home/talabi/data/triplet_segmentation_dataset_v2_second_stage'
elif os.name == 'nt':  # Windows systems
    dataset_path = 'C:/Users/tal22/Documents/repositories/triplet_segmentation/data/triplet_segmentation_dataset_v2_second_stage'    
else:
    raise EnvironmentError("Unsupported operating system. Unable to set dataset_path.")

verb_and_target_gt_present_for_test = False

train_image_dir = join(dataset_path, 'train/img_dir')
train_ann_dir = join(dataset_path, 'train/ann_second_stage')

val_image_dir = join(dataset_path, 'val/img_dir')
val_ann_dir = join(dataset_path, 'val/ann_second_stage')

test_image_dir = join(dataset_path, 'train/img_dir')
test_ann_dir = join(dataset_path, 'train/ann_second_stage')

# Working Directory
work_dir = f'../resnet_model/work_dirs/{experiment_name}'
save_results_path = join(work_dir, 'results.json')
save_logits_path = join(work_dir, 'results_logits.json')

# Checkpoint and Prediction Settings
# allow_resume = True # allows resumption from latest checkpoint
# load_from_checkpoint = None
# predict_only_mode = False

allow_resume = False # allows resumption from latest checkpoint
load_from_checkpoint =  join(f'../resnet_model/work_dirs/singletask_resnet_fpn_target_parallel_fc', 'epoch_20.pth')
predict_only_mode = True

# Other Constants
image_size = (224, 224)
model_input_size = (224, 224)

use_wce = False
if use_wce:
    # class frequencies weights from train dataset. 
    task_class_frequencies = {
        "gallbladder": 10457,
        "liver": 2895,
        "cystic_duct": 2682,
        "null_target": 1198,
        "omentum": 1093,
        "cystic_artery": 835,
        "cystic_plate": 664,
        "fluid": 528,
        "specimen_bag": 368,
        "abdominal_wall_cavity": 163,
        "blood_vessel": 107,
        "adhesion": 77,
        "gut": 77,
        "cystic_pedicle": 58,
        "peritoneum": 50
    }

    dataset_weight_scaling_factor = 0.25

# instrument_to_task_classes
if architecture ==  'singletask_parrallel_fc':  
    import sys
    sys.path.append('../')
    from utils.general.dataset_variables import TripletSegmentationVariables
    if task_name == 'target':
        instrument_to_task_classes = TripletSegmentationVariables.instrument_to_target_classes
    elif task_name == 'verb':   
        instrument_to_task_classes = TripletSegmentationVariables.instrument_to_verb_classes
    elif task_name == 'verbtarget':   
        instrument_to_task_classes = TripletSegmentationVariables.instrument_to_verbtarget_classes  
         