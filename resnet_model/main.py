import sys
sys.path.append('../')

from dataset import SurgicalDataset, PredictionDataset
from torch.utils.data import DataLoader
from model import MultiTaskResNet
import torch.optim as optim
from loss import MultiTaskLoss
from custom_transform import CustomTransform
from utils.general.dataset_variables import TripletSegmentationVariables

from resnet_model.train_and_test_predict_loop import train_model, test_model, predict_with_model

# Hyperparameters
batch_size = 32
num_epochs = 20
learning_rate = 0.001

# Data Transforms

# Define the transformation
transform = CustomTransform(image_size=(224, 224))


# Dataset Directories
dataset_name = 'triplet_segmentation_dataset_v2_second_stage'
# dataset_name = 'triplet_segmentation_dataset_tiny_second_stage'

train_image_dir = f'../data/{dataset_name}/train/img_dir' 
train_ann_dir = f'../data/{dataset_name}/train/ann_second_stage' 

test_image_dir = f'../data/{dataset_name}/test/img_dir'
test_ann_dir = f'../data/{dataset_name}/test/ann_second_stage'
test_output_json_path = f'../resnet_model/results/verb_target_prediction/test_{dataset_name}_results.json'  
save_model_path = f'../resnet_model/results/model/{dataset_name}_results.pth'

predict_image_dir = f'../data/triplet_segmentation_dataset_v2_second_stage/test/img_dir'
predict_ann_dir = f'../results/mas2former_on_cholecinstanceseg_pretrained/mask2former_test_triplet_segmentation_v2_results/ann_second_stage'
predict_output_json_path = f'../resnet_model/results/verb_target_prediction/predict_mask2former_test_triplet_segmentation_v2_results_full.json'  



# Datasets and DataLoaders
train_dataset = SurgicalDataset(train_image_dir, train_ann_dir, transform, train_mode=True)
test_dataset = SurgicalDataset(test_image_dir, test_ann_dir, transform, train_mode=False)
predict_dataset = PredictionDataset(predict_image_dir, predict_ann_dir, transform, train_mode=False)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
predict_loader = DataLoader(predict_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model, Loss, Optimizer
num_instruments = TripletSegmentationVariables.num_instuments
num_verbs  = TripletSegmentationVariables.num_verbs
num_targets  = TripletSegmentationVariables.num_targets

model = MultiTaskResNet(num_instruments, num_verbs, num_targets)
loss_fn = MultiTaskLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and Test
train_model(model, train_loader, optimizer, loss_fn, num_epochs)
test_model(model, test_loader, test_output_json_path, save_model_path)
predict_with_model(model, predict_loader, predict_output_json_path, device='cuda')
