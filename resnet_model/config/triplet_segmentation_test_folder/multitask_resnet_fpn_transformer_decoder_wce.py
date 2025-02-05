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
model_name = 'MultiTaskResNetFPNTransformerDecoder'
task_name = 'standard_multitask_verb_and_target'

description =  'multitask ResnetFPNWithTransformerDecoder predicting verb and targets with wce'


if model_name in ['SingleTaskResNetFPN', 'SingleTaskResNetFPNWithDecoder']:
    architecture = 'singletask'
elif  model_name in ['MultiTaskResNet', 'MultiTaskResNetFPN', 'MultiTaskResNetFPNTransformerDecoder']:  
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
image_size = (224, 224)
model_input_size = (224, 224)

# class frequencies weights from train dataset. 
task_class_frequencies = {
    "retract,gallbladder": 4986,
    "dissect,gallbladder": 4429,
    "retract,liver": 1944,
    "dissect,cystic_duct": 1807,
    "null_verb,null_target": 1198,
    "grasp,gallbladder": 1011,
    "coagulate,liver": 745,
    "dissect,omentum": 622,
    "dissect,cystic_artery": 583,
    "dissect,cystic_plate": 578,
    "aspirate,fluid": 528,
    "retract,omentum": 453,
    "clip,cystic_duct": 384,
    "grasp,specimen_bag": 368,
    "retract,cystic_duct": 333,
    "cut,cystic_duct": 155,
    "clip,cystic_artery": 130,
    "coagulate,blood_vessel": 105,
    "irrigate,abdominal_wall_cavity": 100,
    "cut,cystic_artery": 96,
    "grasp,liver": 90,
    "retract,gut": 75,
    "dissect,adhesion": 70,
    "cut,liver": 68,
    "coagulate,abdominal_wall_cavity": 63,
    "irrigate,liver": 48,
    "coagulate,cystic_plate": 42,
    "coagulate,gallbladder": 31,
    "grasp,peritoneum": 27,
    "coagulate,cystic_artery": 24,
    "coagulate,cystic_pedicle": 24,
    "grasp,cystic_plate": 22,
    "retract,cystic_plate": 22,
    "coagulate,omentum": 16,
    "clip,cystic_pedicle": 13,
    "grasp,cystic_pedicle": 12,
    "retract,peritoneum": 11,
    "dissect,peritoneum": 7,
    "cut,adhesion": 7,
    "cut,peritoneum": 5,
    "dissect,cystic_pedicle": 4,
    "irrigate,cystic_pedicle": 4,
    "grasp,cystic_artery": 2,
    "grasp,gut": 2,
    "grasp,omentum": 2,
    "coagulate,cystic_duct": 2,
    "grasp,cystic_duct": 1,
    "retract,cystic_pedicle": 1,
    "cut,blood_vessel": 1,
    "clip,blood_vessel": 1,
    "pack,gallbladder": 0,
    "coagulate,peritoneum": 0,
    "dissect,blood_vessel": 0,
    "cut,cystic_plate": 0,
    "cut,omentum": 0,
    "clip,cystic_plate": 0
}
use_wce = True

dataset_weight_scaling_factor = 0.25
