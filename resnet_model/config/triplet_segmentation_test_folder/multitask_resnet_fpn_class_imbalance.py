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
model_name = 'MultiTaskResNetFPN'
task_name = 'standard_multitask_verb_and_target'


if model_name in ['SingleTaskResNetFPN', 'SingleTaskResNetFPNWithDecoder']:
    architecture = 'singletask'
elif  model_name in ['MultiTaskResNet', 'MultiTaskResNetFPN', 'MultiTaskResNetFPNMaskedEmbeddingsSharedTransformerDecoder',
                     'MultiTaskResNetFPNMaskedEmbeddings']:  
    architecture = 'multitask'
    
description =  'resnet_fpn predicting verb target standard multitask way'

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
elif task_name == 'standard_multitask_verb_and_target':
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
allow_resume = True # allows resumption from latest checkpoint
load_from_checkpoint = None
predict_only_mode = False

# Other Constants
image_size = (480, 854)
model_input_size = (448, 800)

# class frequencies weights from train dataset. 
task_class_frequencies =  {
    "retract,gallbladder": 17791,
    "dissect,gallbladder": 16358,
    "dissect,cystic_duct": 4846,
    "retract,liver": 4531,
    "null_verb,null_target": 3462,
    "grasp,gallbladder": 3066,
    "dissect,omentum": 1944,
    "dissect,cystic_artery": 1917,
    "retract,omentum": 1508,
    "coagulate,liver": 1501,
    "dissect,cystic_plate": 1303,
    "aspirate,fluid": 1215,
    "clip,cystic_duct": 1115,
    "grasp,specimen_bag": 1088,
    "clip,cystic_artery": 556,
    "cut,cystic_duct": 385,
    "cut,cystic_artery": 338,
    "retract,cystic_duct": 334,
    "retract,cystic_plate": 296,
    "grasp,peritoneum": 243,
    "coagulate,gallbladder": 242,
    "coagulate,blood_vessel": 211,
    "retract,gut": 183,
    "irrigate,abdominal_wall_cavity": 163,
    "coagulate,cystic_plate": 149,
    "coagulate,omentum": 147,
    "coagulate,abdominal_wall_cavity": 144,
    "cut,peritoneum": 126,
    "grasp,liver": 106,
    "grasp,cystic_plate": 95,
    "dissect,peritoneum": 95,
    "retract,peritoneum": 76,
    "cut,adhesion": 75,
    "dissect,adhesion": 70,
    "cut,liver": 68,
    "irrigate,liver": 66,
    "coagulate,cystic_pedicle": 61,
    "coagulate,cystic_artery": 57,
    "dissect,cystic_pedicle": 47,
    "grasp,cystic_duct": 44,
    "grasp,omentum": 36,
    "coagulate,cystic_duct": 34,
    "grasp,gut": 28,
    "grasp,cystic_artery": 19,
    "cut,blood_vessel": 18,
    "irrigate,cystic_pedicle": 18,
    "coagulate,peritoneum": 14,
    "clip,cystic_plate": 14,
    "clip,cystic_pedicle": 13,
    "grasp,cystic_pedicle": 12,
    "pack,gallbladder": 11,
    "retract,cystic_pedicle": 11,
    "cut,omentum": 6,
    "clip,blood_vessel": 5,
    "dissect,blood_vessel": 0,
    "cut,cystic_plate": 0
}

dataset_weight_scaling_factor = 0.5
