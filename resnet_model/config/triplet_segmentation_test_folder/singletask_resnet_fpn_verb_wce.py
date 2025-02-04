from os.path import join
import os

# Hyperparameters
batch_size = 16
num_epochs = 20
learning_rate = 0.0005

# Dynamically set the experiment name from the filename
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]

# Model name
model_name = 'SingleTaskResNetFPN'
task_name = 'verb'
description =  'resnet_fpn predicting verb wce img size'

if model_name in ['SingleTaskResNetFPN', 'SingleTaskResNetFPNWithDecoder']:
    architecture = 'singletask'
elif  model_name in ['MultiTaskResNet', 'MultiTaskResNetFPN', 'MultiTaskResNetFPNMaskedEmbeddingsSharedTransformerDecoder',
                     'MultiTaskResNetFPNMaskedEmbeddings']:  
    architecture = 'multitask'
    


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

test_image_dir = join(dataset_path, 'mask2former_instrument_prediction/img_dir')
test_ann_dir = join(dataset_path, 'mask2former_instrument_prediction/ann_second_stage')

# Working Directory
work_dir = f'../resnet_model/work_dirs/{experiment_name}'
save_results_path = join(work_dir, 'results.json')
save_logits_path = join(work_dir, 'results_logits.json')

# Checkpoint and Prediction Settings
allow_resume = True # allows resumption from latest checkpoint
load_from_checkpoint = None
predict_only_mode = False

# allow_resume = False # allows resumption from latest checkpoint
# load_from_checkpoint =  join(work_dir, 'best_model.pth')
# predict_only_mode = True


# Other Constants
image_size = (480, 854)
model_input_size = (448, 800)

# class frequencies weights from train dataset. 
task_class_frequencies = {
    "dissect": 8100,
    "retract": 7825,
    "grasp": 1537,
    "null_verb": 1198,
    "coagulate": 1052,
    "clip": 528,
    "aspirate": 528,
    "cut": 332,
    "irrigate": 152,
    "pack": 0
}


dataset_weight_scaling_factor = 0.25
