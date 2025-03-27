from os.path import join
import os
import sys
sys.path.append('../')
from utils.general.dataset_variables import TripletSegmentationVariables


# Hyperparameters
# ---------------------- Hyperparameters ----------------------
batch_size = 64
num_epochs = 1
# batch_size = 1
# num_epochs = 1
learning_rate = 0.0005

# ---------------------- Experiment Info ----------------------
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]
model_name = 'FourTaskResNetFPNWithMoEDecodersAndSoftmaxInputs'
task_name = 'fourtask_moe'
architecture = 'fourtask_moe'
description = 'MoE 4-task prediction (verb, target, verbtarget, ivt) using detector logits as softmax input'


# ---------------------- Dataset Directories ----------------------
if os.name == 'posix':
    dataset_path = '/nfs/home/talabi/data/triplet_segmentation_dataset_v2_second_stage'
elif os.name == 'nt':
    dataset_path = 'C:/Users/tal22/Documents/repositories/datasets/my_triplet_seg_datasets/triplet_segmentation_dataset_v2_second_stage'
else:
    raise EnvironmentError("Unsupported OS")

verb_and_target_gt_present_for_test = False

train_image_dir = join(dataset_path, 'train/img_dir')
train_ann_dir = join(dataset_path, 'train/ann_second_stage')

val_image_dir = join(dataset_path, 'val/img_dir')
val_ann_dir = join(dataset_path, 'val/ann_second_stage')

test_image_dir = join(dataset_path, 'test_soft_labels/img_dir')
test_ann_dir = join(dataset_path, 'test_soft_labels/predicted_instance_masks')

# Path to detector logits (softmax) JSON file
detector_softmax_scores_json = join(dataset_path, 'test_soft_labels', 'logits_from_first_stage.json')

# ---------------------- Working Directory ----------------------
work_dir = f'../resnet_model/work_dirs/{experiment_name}'
save_results_path = join(work_dir, 'results.json')
save_logits_path = join(work_dir, 'results_logits.json')
vis_dir = join(work_dir, 'vis_dir/')

# ---------------------- Checkpoint and Prediction Settings ----------------------
allow_resume = True
load_from_checkpoint = None
predict_only_mode = False

# allow_resume = False
# load_from_checkpoint = join(work_dir, 'best_model.pth')
# predict_only_mode = True

# ---------------------- Input Shape ----------------------
image_size = (224, 224)
model_input_size = (224, 224)

# ---------------------- Loss Settings ----------------------
use_wce = False
if use_wce:
    task_class_frequencies = None
    dataset_weight_scaling_factor = None

# ---------------------- Class Subset Mappings ----------------------
instrument_to_verb_classes = TripletSegmentationVariables.instrument_to_verb_classes
instrument_to_target_classes = TripletSegmentationVariables.instrument_to_target_classes
instrument_to_verbtarget_classes = TripletSegmentationVariables.instrument_to_verbtarget_classes
instrument_to_triplet_classes = TripletSegmentationVariables.instrument_to_triplet_classes
