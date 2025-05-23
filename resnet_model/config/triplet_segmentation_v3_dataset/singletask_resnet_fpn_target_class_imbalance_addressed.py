from os.path import join
import os

# Hyperparameters
batch_size = 16
num_epochs = 25
learning_rate = 0.001

# Dynamically set the experiment name from the filename
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]

# Model name
model_name = 'SingleTaskResNetFPN'
task_name = 'target'
description =  'resnet_fpn predicting target only class imbalance addressed'


if model_name in ['SingleTaskResNetFPN', 'SingleTaskResNetFPNWithDecoder']:
    architecture = 'singletask'
elif  model_name in ['MultiTaskResNet', 'MultiTaskResNetFPN', 'MultiTaskResNetFPNMaskedEmbeddingsSharedTransformerDecoder',
                     'MultiTaskResNetFPNMaskedEmbeddings']:  
    architecture = 'multitask'
    


# Dataset Directories
if os.name == 'posix':  # Unix-like systems (Linux, macOS)
    dataset_path = '/nfs/home/talabi/data/triplet_segmentation_v3_final'
elif os.name == 'nt':  # Windows systems
    dataset_path = 'C:/Users/tal22/Documents/repositories/triplet_segmentation/data/triplet_segmentation_v3_final'    
else:
    raise EnvironmentError("Unsupported operating system. Unable to set dataset_path.")

if task_name == 'target':
    label_id_json_file_path = join(dataset_path, 'label_ids/label_ids_target_train_v3.json')
elif task_name == 'verb':
    label_id_json_file_path = join(dataset_path, 'label_ids/label_ids_verb_train_v3.json')
elif task_name == 'verbtarget':
    label_id_json_file_path = join(dataset_path, 'label_ids/label_ids_verbtarget_train_v3.json')            

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
# allow_resume = False # allows resumption from latest checkpoint
# load_from_checkpoint =  join(work_dir, 'best_model.pth')
# predict_only_mode = True

allow_resume = True # allows resumption from latest checkpoint
load_from_checkpoint = None
predict_only_mode = False

# Other Constants
image_size = (480, 854)
model_input_size = (448, 800)

# class frequencies weights from train dataset. 
task_class_frequencies = {
    "gallbladder": 37468,
    "cystic_duct": 6758,
    "liver": 6272, 
    "omentum": 3641,   
    "null_target": 3462,    
    "cystic_artery": 2887,
    "cystic_plate": 1857,
    "fluid": 1215,
    "specimen_bag": 1088,
    "peritoneum": 554,
    "abdominal_wall_cavity": 307,
    "blood_vessel": 234,
    "gut": 211,
    "adhesion": 162,
    "cystic_pedicle": 145,
}

dataset_weight_scaling_factor = 0.5
