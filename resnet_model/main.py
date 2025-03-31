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
from resnet_model.dataset import SurgicalSingletaskDatasetForParallelFCLayers, SurgicalThreetaskDatasetForParallelLayers
from resnet_model.dataset import SurgicalFourTaskDatasetWithDetectorLogits, PredictionDatasetSofmaxInputs

from custom_transform import CustomTransform

from loss import MultiTaskLoss, MultiTaskLossThreeTasks, MultiTaskLossFourTasks


from utils.general.dataset_variables import TripletSegmentationVariables
from resnet_model.model_utils import get_dataset_label_ids
from resnet_model.checkpoint_utils import load_checkpoint, load_checkpoint_from_latest


from resnet_model.train_test_predict_loop_singletask import train_model_singletask,  predict_with_model_singletask, predict_with_model_parallel_fc_layers
from resnet_model.train_test_predict_loop_threetask import train_model_threetask, predict_with_model_threetask
from resnet_model.train_test_predict_loop_two_task import train_model_multitask,  predict_with_model_multitask
from resnet_model.train_test_predict_loop_four_task import train_model_fourtask,  predict_with_model_fourtask





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
    
    # get some metadata
    task_name = getattr(config, 'task_name', None)
     
    num_instruments = TripletSegmentationVariables.num_instuments
    num_verbs = TripletSegmentationVariables.num_verbs
    num_targets = TripletSegmentationVariables.num_targets
    num_verbtargets = TripletSegmentationVariables.num_verbtargets
    num_triplets = TripletSegmentationVariables.num_triplets    
   
    
    instrument_dict = TripletSegmentationVariables.categories['instrument']
    verb_dict =  TripletSegmentationVariables.categories['verb']
    target_dict = TripletSegmentationVariables.categories['target']
    verbtarget_dict = TripletSegmentationVariables.categories['verbtarget']
    triplet_dict = TripletSegmentationVariables.categories['triplet']
    
    # Dynamic model import
    model_class = getattr(importlib.import_module("models"), config.model_name)
    
    # Get some constants that are dependent on information
    if task_name == 'verb':
        num_task_class = num_verbs
        print('class names', list(verb_dict.values())) 
    elif task_name == 'target':
        num_task_class = num_targets
        print('class names', list(target_dict.values())) 
    elif task_name == 'verbtarget':
        num_task_class = num_verbtargets
        print('class names', list(verbtarget_dict.values()))      
    elif task_name == 'standard_multitask_verb_and_target':
        num_task_class = num_verbtargets
        print('class names', list(verbtarget_dict.values()))  
    elif task_name == 'threetask':
        print('we are predicting verbs, targets and verbtargets. Threetasks!! ')   
    elif task_name == 'fourtask_moe':
        print('we are predicting verbs, targets verbtargets and ivt. fourtasks!! ')                       
    else:
        raise ValueError("We currently only accept 'verb', 'target', 'verbtarget', 'standard_multitask_verb_and_target', 'threetask'")

    # Define the transformation
    transform = CustomTransform(image_size=config.image_size,
                                model_input_size=config.model_input_size)
    
    if config.architecture == 'singletask':
        _SurgicalDataset = SurgicalSingletaskDataset
        _train_model = train_model_singletask
        _predict_with_model = predict_with_model_singletask
    elif  config.architecture == 'singletask_parrallel_fc':
        _SurgicalDataset = SurgicalSingletaskDatasetForParallelFCLayers
        _train_model = train_model_singletask
        _predict_with_model = predict_with_model_parallel_fc_layers 
    elif  config.architecture == 'multitask':   
        _SurgicalDataset = SurgicalMultitaskDataset
        _train_model = train_model_multitask
        _predict_with_model = predict_with_model_multitask
    elif  config.architecture == 'threetask_parallel_fc':
        _SurgicalDataset = SurgicalThreetaskDatasetForParallelLayers
        _train_model = train_model_threetask
        _predict_with_model = predict_with_model_threetask    
    elif config.architecture == 'fourtask_moe':
        _SurgicalDataset = SurgicalFourTaskDatasetWithDetectorLogits
        _train_model = train_model_fourtask
        _predict_with_model = predict_with_model_fourtask        
    else:
        raise ValueError("we currently only accept 'singletask', 'multitask', 'singletask_parrallel_fc', 'threetask_parallel_fc'")      

    # Datasets and DataLoaders
    train_dataset = _SurgicalDataset(config, config.train_image_dir, config.train_ann_dir, transform, train_mode=True)    
    val_dataset = _SurgicalDataset(config, config.val_image_dir, config.val_ann_dir, transform, train_mode=False)
    if config.architecture == 'fourtask_moe':
        test_dataset = PredictionDatasetSofmaxInputs(
            config,
            config.test_image_dir,
            config.test_ann_dir,
            config.detector_softmax_scores_json,
            transform,
            train_mode=False
        )
    else:
        test_dataset = PredictionDataset(config, config.test_image_dir, config.test_ann_dir, transform, train_mode=False)


    # Get class_name to id.  for wce. 
    if task_name == 'verb':
        class_to_idx_zero_index = {value: int(key)-1 for key, value in verb_dict.items()}  
    elif task_name == 'target':
        class_to_idx_zero_index = {value: int(key)-1 for key, value in target_dict.items()} 
    elif task_name == 'verbtarget':
        class_to_idx_zero_index = {value: int(key)-1 for key, value in verbtarget_dict.items()}   
    elif task_name == 'standard_multitask_verb_and_target':   
        class_to_idx_zero_index = {value: int(key)-1 for key, value in verbtarget_dict.items()}   
    elif task_name == 'threetask':
        pass    
    elif task_name == 'fourtask_moe':
        pass        
    else:
        raise ValueError("We currently only accept 'verb', 'target', 'verbtarget', 'verbtarget_multitask' or 'triplet'")
    
    # weighted cross entropy or normal crossentropy 
    if config.architecture == 'singletask': 
        if config.use_wce:
            assert hasattr(config, "task_class_frequencies"), 'task frequencies are required'
            print('using weighted cross entropy')
            loss_class_weights = {cls: (1 / (freq ** config.dataset_weight_scaling_factor)) if freq > 0 else 0  
                        for cls, freq in config.task_class_frequencies.items()}   
            #Normalize and ensure it sums to 1. 
            total_weight = sum(loss_class_weights.values())
            loss_normalized_weights = {cls: weight / total_weight for cls, weight in loss_class_weights.items()}  
            # arange and convert to tensor.   
            loss_weights_tensor = torch.tensor([loss_normalized_weights[cls] for cls in 
                                                sorted(class_to_idx_zero_index, 
                                                    key=class_to_idx_zero_index.get)],
                                            dtype=torch.float, 
                                            device='cuda') 
            print(f'loss_weights_tensor {loss_weights_tensor}')             
            _loss_fn = nn.CrossEntropyLoss(weight=loss_weights_tensor)        
        else: 
            print('using standard cross entropy')
            _loss_fn = nn.CrossEntropyLoss()   
    elif config.architecture == 'singletask_parrallel_fc':
        _loss_fn = nn.CrossEntropyLoss()           
    elif config.architecture == 'multitask':               
        _loss_fn = MultiTaskLoss(config)
    elif  config.architecture == 'threetask_parallel_fc':               
        _loss_fn = MultiTaskLossThreeTasks(config)    
    elif  config.architecture == 'fourtask_moe':               
        _loss_fn = MultiTaskLossFourTasks(config)   
    else:
        raise ValueError("we currently only accept 'singletask', 'multitask', 'singletask_parrallel_fc', 'threetask_parallel_fc'")      

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # Initialize Model, Loss, Optimizer 
    if config.architecture == 'singletask':
        model = model_class(num_instruments, num_task_class)
    elif config.architecture == 'singletask_parrallel_fc':
        model = model_class(config, num_instruments, config.instrument_to_task_classes)    
    elif config.architecture == 'multitask':   
        model = model_class(num_instruments, num_verbs, num_targets) 
    elif config.architecture == 'threetask_parallel_fc':   
        model = model_class(config,
                            config.instrument_to_verb_classes, 
                            config.instrument_to_target_classes, 
                            config.instrument_to_verbtarget_classes,)  
    elif config.architecture == 'fourtask_moe':
        model = model_class(config,
                            config.instrument_to_verb_classes,
                            config.instrument_to_target_classes,
                            config.instrument_to_verbtarget_classes,
                            config.instrument_to_triplet_classes,
                            num_instruments=num_instruments,
                            num_verbs=num_verbs,
                            num_targets=num_targets,
                            num_verbtargets=num_verbtargets,
                            num_triplets=num_triplets)       
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # Resume training from checkpoint if provided
    start_epoch = 0
    best_val_accuracy = -1.0
    
    #allow for resumption
    if config.allow_resume:
        if os.path.isdir(config.work_dir) and  len([f for f in os.listdir(config.work_dir) if os.path.isfile(os.path.join(config.work_dir, f))]):            
            start_epoch, best_val_accuracy = load_checkpoint_from_latest(config, model, optimizer)
            print(f'resuming from epoch, {start_epoch+1}, best_val_accuracy {best_val_accuracy}' )
    
    if config.load_from_checkpoint:
        start_epoch, best_val_accuracy = load_checkpoint(config, model, optimizer)
        print(f'load from epoch, {start_epoch+1}, best_val_accuracy {best_val_accuracy}' )

    # Run in prediction mode
    if config.predict_only_mode:
        print(f'predicting only with checkpoint {config.load_from_checkpoint}')
        assert config.load_from_checkpoint, "A checkpoint to load the model is required for prediction mode."
        _predict_with_model(
            config=config,
            model=model,
            dataloader=test_loader,
            device='cuda',
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
        
        #load best model checkpoint and test.
        config.load_from_checkpoint = join(config.work_dir, 'best_model.pth')
        start_epoch, best_val_accuracy = load_checkpoint(config, model, optimizer)
        
        _predict_with_model(
            config=config,
            model=model,
            dataloader=test_loader,
            device='cpu',
            store_results=True,
            verbose=True,
        )


if __name__ == '__main__':
    main()
