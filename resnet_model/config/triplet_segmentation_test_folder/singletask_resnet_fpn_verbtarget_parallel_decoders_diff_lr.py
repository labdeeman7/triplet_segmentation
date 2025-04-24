from os.path import join
import os
import sys
sys.path.append('../')
from utils.general.dataset_variables import TripletSegmentationVariables


# Hyperparameters
batch_size = 128
num_epochs = 30
learning_rate = 0.00001

# Dynamically set the experiment name from the filename
import os
experiment_name = os.path.splitext(os.path.basename(__file__))[0]

# Model name 
model_name = 'SingleTaskResNetFPNWithParallelTransformersSpatialFeatures' 
task_name = 'verbtarget'

description =  'predict verbtargets with parrallel transformers and fcs'
architecture = 'singletask_parrallel_fc'
     
  

# Dataset Directories
if os.name == 'posix':  # Unix-like systems (Linux, macOS)
    dataset_path = '/nfs/home/talabi/data/triplet_segmentation_dataset_v2_second_stage'
elif os.name == 'nt':  # Windows systems
    dataset_path = 'C:/Users/tal22/Documents/repositories/datasets/my_triplet_seg_datasets/triplet_segmentation_dataset_v2_second_stage'    
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
vis_dir =  join(work_dir, 'vis_dir/')

# Checkpoint and Prediction Settings
# allow_resume = True # allows resumption from latest checkpoint
# load_from_checkpoint = None
# predict_only_mode = False

# allow_resume = True # allows resumption from latest checkpoint
# load_from_checkpoint = join(work_dir, 'epoch_10.pth')
# predict_only_mode = False

allow_resume = False 
load_from_checkpoint =  '/nfs/home/talabi/repositories/triplet_segmentation/checkpoints/singletask_resnet_fpn_verbtarget_parallel_decoders_spatial_tokens/epoch_9_best_mean_so_far.pth'
predict_only_mode = True

# Other Constants
image_size = (256, 448)
model_input_size = (256, 448)

use_wce = False
if use_wce:
    # class frequencies weights from train dataset. 
    task_class_frequencies = None
    dataset_weight_scaling_factor = None

# instrument_to_task_classes
if architecture ==  'singletask_parrallel_fc':  
    if task_name == 'target':
        instrument_to_task_classes = TripletSegmentationVariables.instrument_to_target_classes
    elif task_name == 'verb':   
        instrument_to_task_classes = TripletSegmentationVariables.instrument_to_verb_classes
    elif task_name == 'verbtarget':   
        instrument_to_task_classes = TripletSegmentationVariables.instrument_to_verbtarget_classes  
        
elif architecture ==  'threetask_parallel_fc':  
    from utils.general.dataset_variables import TripletSegmentationVariables
    instrument_to_verb_classes = TripletSegmentationVariables.instrument_to_verb_classes
    instrument_to_target_classes = TripletSegmentationVariables.instrument_to_target_classes
    instrument_to_verbtarget_classes = TripletSegmentationVariables.instrument_to_verbtarget_classes  
   