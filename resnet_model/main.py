from dataset import SurgicalDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from model import MultiTaskResNet
import torch.optim as optim
from loss import MultiTaskLoss

from train_and_test_loop import train_model, test_model

# Hyperparameters
batch_size = 16
num_epochs = 20
learning_rate = 0.001

# Data Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset Directories
train_image_dir = "dataset/train/images"
test_image_dir = "dataset/test/images"
train_annotations_file = "dataset/train/instruments.csv"
test_annotations_file = "dataset/test/instruments.csv"
train_actions_file = "dataset/train/actions.csv"
test_actions_file = "dataset/test/actions.csv"
train_targets_file = "dataset/train/targets.csv"
test_targets_file = "dataset/test/targets.csv"

# Datasets and DataLoaders
train_dataset = SurgicalDataset(train_image_dir, train_annotations_file, train_actions_file, train_targets_file, transform)
test_dataset = SurgicalDataset(test_image_dir, test_annotations_file, test_actions_file, test_targets_file, transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Initialize Model, Loss, Optimizer
num_actions = len(pd.read_csv(train_actions_file).iloc[:, 1].unique())
num_targets = len(pd.read_csv(train_targets_file).iloc[:, 1].unique())

model = MultiTaskResNet(num_actions, num_targets)
loss_fn = MultiTaskLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Train and Test
train_model(model, train_loader, optimizer, loss_fn, num_epochs)
test_model(model, test_loader)