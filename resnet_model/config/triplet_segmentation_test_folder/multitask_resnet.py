from os.path import join

# Hyperparameters
batch_size = 64
num_epochs = 50
learning_rate = 0.001

# Dynamically set the experiment name from the filename
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]

# Model name
model_name = 'MultiTaskResNet'

# Dataset Directories
dataset_path = '/nfs/home/talabi/data/triplet_segmentation_dataset_v2_second_stage'
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

# Checkpoint and Prediction Settings
load_from_checkpoint = None
predict_only_mode = False

# Other Constants
image_size = (224, 224)