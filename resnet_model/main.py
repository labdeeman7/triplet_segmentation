import sys
sys.path.append('../')

from os.path import join
import argparse
import importlib
import torch
from torch.utils.data import DataLoader
from dataset import SurgicalDataset, PredictionDataset
from loss import MultiTaskLoss
from custom_transform import CustomTransform
from utils.general.dataset_variables import TripletSegmentationVariables
from train_and_test_predict_loop import train_model, test_model, load_checkpoint, predict_with_model


def main():
    # Argument parser
    parser = argparse.ArgumentParser(description="Train or predict on a model.")
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help="The Python module (without .py extension) containing the configuration settings."
    )
    parser.add_argument(
        '--predict',
        action='store_true',
        help="Run the model in prediction mode instead of training."
    )
    args = parser.parse_args()

    # Dynamic configuration import
    config = importlib.import_module(args.config)
    # print(config)

    # Dynamic model import
    model_class = getattr(importlib.import_module("models"), config.model_name)
    # print(model_class)

    # Define the transformation
    transform = CustomTransform(image_size=config.image_size)

    # Datasets and DataLoaders
    train_dataset = SurgicalDataset(config.train_image_dir, config.train_ann_dir, transform, train_mode=True)
    val_dataset = SurgicalDataset(config.val_image_dir, config.val_ann_dir, transform, train_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    if config.verb_and_target_gt_present_for_test:
        test_dataset = SurgicalDataset(config.test_image_dir, config.test_ann_dir, transform, train_mode=False)
    else:
        test_dataset = PredictionDataset(config.test_image_dir, config.test_ann_dir, transform, train_mode=False)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize Model, Loss, Optimizer
    num_instruments = TripletSegmentationVariables.num_instuments
    num_verbs = TripletSegmentationVariables.num_verbs
    num_targets = TripletSegmentationVariables.num_targets

    model = model_class(num_instruments, num_verbs, num_targets)
    loss_fn = MultiTaskLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Resume training from checkpoint if provided
    start_epoch = 0
    best_val_accuracy = 0.0
    if config.load_from_checkpoint:
        start_epoch, best_val_accuracy = load_checkpoint(config.load_from_checkpoint, model, optimizer)

    # Run in prediction mode
    if config.predict_only_mode:
        assert config.load_from_checkpoint, "A checkpoint to load the model is required for prediction mode."
        predict_with_model(
            model=model,
            dataloader=test_loader,
            work_dir=config.work_dir,
            device='cuda',
            save_results_path=config.save_results_path
        )
    else:
        # Train and test the model
        train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            work_dir=config.work_dir,
            num_epochs=config.num_epochs,
            device='cuda',
            start_epoch=start_epoch,
            best_val_accuracy=best_val_accuracy
        )
        
        #load best model checkpoint
        best_model_path = join(config.work_dir, 'best_model.pth')
        start_epoch, best_val_accuracy = load_checkpoint(best_model_path, model, optimizer)

        test_model(
            model=model,
            dataloader=test_loader,
            device='cuda',
            save_results_path=config.save_results_path,
            verbose=True,
            verb_and_target_gt_present_for_test=config.verb_and_target_gt_present_for_test
        )


if __name__ == '__main__':
    main()
