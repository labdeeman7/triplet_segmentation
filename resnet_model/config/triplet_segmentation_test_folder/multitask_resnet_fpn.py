from os.path import join

# Hyperparameters
batch_size = 32
num_epochs = 2
learning_rate = 0.001

# Dynamically set the experiment name from the filename
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]

# Model name
model_name = 'MultiTaskResNetFPN'

# Dataset Directories
dataset_name = 'triplet_segmentation_dataset_v2_second_stage'
verb_and_target_gt_present_for_test = False

train_image_dir = f'../data/{dataset_name}/train/img_dir'
train_ann_dir = f'../data/{dataset_name}/train/ann_second_stage'

val_image_dir = f'../data/{dataset_name}/val/img_dir'
val_ann_dir = f'../data/{dataset_name}/val/ann_second_stage'

test_image_dir = f'../data/{dataset_name}/mask2former_instrument_prediction/img_dir'
test_ann_dir = f'../data/{dataset_name}/mask2former_instrument_prediction/ann_second_stage'


# Working Directory
work_dir = f'../resnet_model/work_dirs/{experiment_name}'
save_results_path = join(work_dir, 'results.json')

# Checkpoint and Prediction Settings
load_from_checkpoint = None
predict_only_mode = False

# Other Constants
image_size = (224, 224)