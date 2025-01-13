import sys
sys.path.append('../')

from dataset import PredictionDataset
from torch.utils.data import DataLoader
from resnet_model.models.multitask_resnet import MultiTaskResNet
import torch

from custom_transform import CustomTransform
from utils.general.dataset_variables import TripletSegmentationVariables
from resnet_model.train_and_test_predict_loop import predict_with_model

# Define paths
img_dir = "path_to_images"
mask_dir = "path_to_masks"
output_json_path = "predictions.json"

# Define transforms
transform = CustomTransform(image_size=(224, 224))

# Create prediction dataset and dataloader
prediction_dataset = PredictionDataset(img_dir, mask_dir, transform=transform)
prediction_dataloader = DataLoader(prediction_dataset, batch_size=16, shuffle=False)

# Initialize Model, Loss, Optimizer
num_instruments = TripletSegmentationVariables.num_instuments
num_verbs  = TripletSegmentationVariables.num_verbs
num_targets  = TripletSegmentationVariables.num_targets

# Load model
model = MultiTaskResNet(num_instruments, num_verbs, num_targets)
model.load_state_dict(torch.load("model_weights.pth"))
model.to('cuda')

# Run predictions
predict_with_model(model, prediction_dataloader, output_json_path)
