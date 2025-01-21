import sys
sys.path.append('../')
import torch.nn as nn

from os.path import join
import os
import argparse
import importlib
import torch
from torch.utils.data import DataLoader
from resnet_model.dataset import SurgicalSingletaskDataset, PredictionDataset, SurgicalMultitaskDataset
from loss import MultiTaskLoss
from custom_transform import CustomTransform
from utils.general.dataset_variables import TripletSegmentationVariables
from resnet_model.train_test_predict_loop_singletask import train_model_singletask, test_model_singletask, predict_with_model_singletask
from resnet_model.train_test_predict_loop_multitask import train_model_multitask, test_model_multitask, predict_with_model_multitask
from resnet_model.checkpoint_utils import load_checkpoint, load_checkpoint_from_latest

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
    
    # Dynamic model import
    model_class = getattr(importlib.import_module("models"), config.model_name)
    # print(model_class)

    # Define the transformation
    transform = CustomTransform(image_size=config.image_size)
    
    if config.architecture == 'singletask':
        _SurgicalDataset = SurgicalSingletaskDataset
        _loss_fn = nn.CrossEntropyLoss()
        _train_model = train_model_singletask
        _test_model = test_model_singletask
        _predict_with_model = predict_with_model_singletask
        
    elif  config.architecture == 'multitask':   
        _SurgicalDataset = SurgicalMultitaskDataset
        _loss_fn = MultiTaskLoss()
        _train_model = train_model_multitask
        _test_model = test_model_multitask
        _predict_with_model = predict_with_model_multitask
    else:
        raise ValueError("we currently only accept 'singletask', 'multitask'")      

    # Datasets and DataLoaders
    train_dataset = _SurgicalDataset(config, config.train_image_dir, config.train_ann_dir, transform, train_mode=True)
    val_dataset = _SurgicalDataset(config, config.val_image_dir, config.val_ann_dir, transform, train_mode=True)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    if config.verb_and_target_gt_present_for_test:
        test_dataset = _SurgicalDataset(config, config.test_image_dir, config.test_ann_dir, transform, train_mode=False)
    else:
        test_dataset = PredictionDataset(config.test_image_dir, config.test_ann_dir, transform, train_mode=False)

    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize Model, Loss, Optimizer
    num_instruments = TripletSegmentationVariables.num_instuments
    num_verbs = TripletSegmentationVariables.num_verbs
    num_targets = TripletSegmentationVariables.num_targets
    num_verbtargets = TripletSegmentationVariables.num_verbtargets
    task_name = getattr(config, 'task_name', None)

    if task_name == 'verb':
        num_task_class = num_verbs
    elif task_name == 'target':
        num_task_class = num_targets
    elif task_name == 'verbtarget':
        num_task_class = num_verbtargets
    elif not task_name:
        pass  # Do nothing if task_name is defined but not recognized
    else:
        raise ValueError("We currently only accept 'verb', 'target', or 'verbtarget'.")
    
    if config.architecture == 'singletask':
        model = model_class(num_instruments, num_task_class)
    elif config.architecture == 'multitask':   
        model = model_class(num_instruments, num_verbs, num_targets) 
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Resume training from checkpoint if provided
    start_epoch = 0
    best_val_accuracy = 0.0
    
    #allow for resumption
    if config.allow_resume:
        if os.path.isdir(config.work_dir) and len(os.listdir(config.work_dir) ):            
            start_epoch, best_val_accuracy = load_checkpoint_from_latest(config, model, optimizer)
            print(f'resuming from epoch, {start_epoch}, best_val_accuracy {best_val_accuracy}' )
    
    if config.load_from_checkpoint:
        start_epoch, best_val_accuracy = load_checkpoint(config, model, optimizer)
        print(f'load from epoch, {start_epoch}, best_val_accuracy {best_val_accuracy}' )

    # Run in prediction mode
    if config.predict_only_mode:
        assert config.load_from_checkpoint, "A checkpoint to load the model is required for prediction mode."
        _predict_with_model(
            config=config,
            model=model,
            dataloader=test_loader,
            device='cuda',
            save_results_path=config.save_results_path
        )
    else:              
        # Train and test the model
        _train_model(
            config=config,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=_loss_fn,
            num_epochs=config.num_epochs,
            device='cuda',
            start_epoch=start_epoch,
            best_val_accuracy=best_val_accuracy
        )
        
        # save from latest
        _test_model(
            config=config,
            model=model,
            dataloader=test_loader,
            device='cuda',
            save_results_path=config.save_latest_results_path,
            verbose=True,
        )
        
        #load best model checkpoint
        best_model_path = join(config.work_dir, 'best_model.pth')
        start_epoch, best_val_accuracy = load_checkpoint(best_model_path, model, optimizer)
        
        _test_model(
            config=config,
            model=model,
            dataloader=test_loader,
            device='cuda',
            save_results_path=config.save_results_path,
            verbose=True,
        )


if __name__ == '__main__':
    main()
