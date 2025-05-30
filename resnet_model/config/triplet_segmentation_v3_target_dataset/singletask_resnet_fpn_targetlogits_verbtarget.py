from os.path import join
import os

# Hyperparameters
batch_size = 124
num_epochs = 30
learning_rate = 0.00001

# Dynamically set the experiment name from the filename
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]

# Model name
model_name = 'SingleTaskResNetFPNTargetLogits'  # Updated model name
task_name = 'verbtarget'
description = f'resnet_fpn predicting {task_name} only with target logits'
architecture = 'singletaskwithtargetlogits'

# Dataset Directories
if os.name == 'posix':
    dataset_path = '/nfs/home/talabi/data/triplet_segmentation_dataset_v3'
elif os.name == 'nt':
    dataset_path = 'C:/Users/tal22/Documents/repositories/datasets/my_triplet_seg_datasets/triplet_segmentation_dataset_v3'
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
work_dir = f'../resnet_model/work_dirs_v3/{experiment_name}'
save_results_path = join(work_dir, 'results.json')
save_logits_path = join(work_dir, 'results_logits.json')
vis_dir = join(work_dir, 'vis_dir/')

# Checkpoint and Prediction Settings
allow_resume = True
load_from_checkpoint = None
predict_only_mode = False

use_wce = False

# Other Constants
image_size = (256, 448)
model_input_size = (256, 448)

# Target Logits Info
num_target_logits = 9  
target_logits_dir_train = join(dataset_path, 'target_prediction/train/logits')  
target_logits_dir_val = join(dataset_path, 'target_prediction/val/logits')      
target_logits_dir_test = join(dataset_path, 'target_prediction/test/logits')    
