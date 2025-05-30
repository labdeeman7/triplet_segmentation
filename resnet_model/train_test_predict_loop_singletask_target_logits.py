import torch
import json
import os
import wandb
import types
from resnet_model.checkpoint_utils import save_checkpoint
import numpy as np
from collections import defaultdict
from resnet_model.model_utils import save_visualization
from utils.general.dataset_variables import TripletSegmentationVariables

def train_model_singletask_with_target_logits(config,
                model, 
                train_loader, 
                val_loader, 
                optimizer, 
                loss_fn, 
                num_epochs=10, 
                device='cuda', 
                start_epoch=0,
                best_val_accuracy=0.0, 
                verbose=True):
    
    # Initialize WandB
    wandb_config = {
        key: value
        for key, value in vars(config).items()
        if not key.startswith("__") and not callable(value) and not isinstance(value, types.ModuleType)
    }
    
    print('config', wandb_config)
    wandb.init(project="triplet_segmentation", config=wandb_config)
    
    task_name = config.task_name  
    
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)    

    for epoch in range(start_epoch, num_epochs):
        print(f'started epoch {epoch+1}, best_val_accuracy, {best_val_accuracy}', flush=True)
        model.train()
        # model.eval() #debug change
        running_loss = 0.0
        total_task_correct = 0
        total_samples = 0
        
        # For mean accuracy
        class_correct = defaultdict(int)
        class_counts = defaultdict(int)

        
        for imgs, masks, target_logits, instrument_ids, instance_ids, task_gt_ids, _ in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            target_logits = target_logits.to(device)
            instrument_ids = instrument_ids.to(device)
            task_gt_ids = task_gt_ids.to(device)            
            
            optimizer.zero_grad()

            # Forward pass
            task_logits = model(imgs, masks, target_logits, instrument_ids)
            loss = loss_fn(task_logits, task_gt_ids)   
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            task_preds = torch.argmax(task_logits, dim=1)
            total_task_correct += (task_preds == task_gt_ids).sum().item()
            total_samples += imgs.size(0)
            
            
            # Compute per-class accuracy
            for cls in torch.unique(task_gt_ids):
                cls = cls.item()
                class_correct[cls] += ((task_preds == cls) & (task_gt_ids == cls)).sum().item()
                class_counts[cls] += (task_gt_ids == cls).sum().item()
            
            
            # break
            
            
            
        # Compute per-class accuracy & mean accuracy
        class_accuracies = {
            cls: class_correct[cls] / class_counts[cls] if class_counts[cls] > 0 else 0
            for cls in class_counts
        }
        mean_accuracy = np.mean(list(class_accuracies.values()))           
        
        # Calculate accuracy
        train_loss = running_loss / len(train_loader)
        task_accuracy = total_task_correct / total_samples 
                        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}", flush=True)
            print(f"Train {task_name} Accuracy: {task_accuracy:.2f}", flush=True)
            print(f"Train {task_name} mean Accuracy: {mean_accuracy:.2f}", flush=True)
        
        
        # Log training metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            f"train_{task_name}_accuracy": task_accuracy,
            f"train_{task_name}_mean_accuracy": mean_accuracy,
        })    
        
            

        # Validate the model
        print('Validation')
        val_task_accuracy, val_mean_accuracy = validation_singletask_with_target_logits(config,
                                                model, 
                                                val_loader, 
                                                device=device,
                                                store_results=False, 
                                                verbose=True)
        
        if verbose:
            print(f"Val {task_name} Accuracy: {val_task_accuracy:.2f}", flush=True)
            print(f"Val {task_name} mean Accuracy: {val_mean_accuracy:.2f}", flush=True)

        wandb.log({
            f"val_{task_name}_accuracy": val_task_accuracy,
            f"val_{task_name}_mean_accuracy": val_mean_accuracy,
        })
        
        
        # Save the best model
        if val_task_accuracy > best_val_accuracy:
            best_val_accuracy = val_task_accuracy
            best_model_path = os.path.join(config.work_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, best_model_path)
            print(f"New best model saved to {best_model_path} with accuracy {best_val_accuracy:.2f}", flush=True)

        # Save the latest model
        latest_model_path = os.path.join(config.work_dir, f"epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, best_val_accuracy, latest_model_path)
        
        # Delete previous if it exist
        previous_saved_model_path = os.path.join(config.work_dir, f"epoch_{epoch}.pth")

        if os.path.exists(previous_saved_model_path):
            os.remove(previous_saved_model_path)

         
            

def validation_singletask_with_target_logits(config,
                            model, 
                            dataloader, 
                            device='cuda', 
                            store_results = True,
                            verbose=True ):
    
    task_name = config.task_name
    model.eval()
    total_task_correct = 0
    total_samples = 0
    class_correct = defaultdict(int)
    class_counts = defaultdict(int)
    results = {}
    logits_results = {}  # Dictionary to store logits

    with torch.no_grad():
        for imgs, masks, target_logits, instrument_ids, instance_ids, task_gt_ids, mask_names in dataloader:
            imgs = imgs.to(device)
            masks = masks.to(device)
            target_logits = target_logits.to(device)
            instrument_ids = instrument_ids.to(device)
            task_gt_ids = task_gt_ids.to(device)
            
           

            task_logits = model(imgs, masks, target_logits, instrument_ids)
            task_preds = torch.argmax(task_logits, dim=1)
            # Compute accuracy
            total_task_correct += (task_preds == task_gt_ids).sum().item()
            total_samples += imgs.size(0)
           
            
            # Compute per-class accuracy
            for cls in torch.unique(task_gt_ids):
                cls = cls.item()
                class_correct[cls] += ((task_preds == cls) & (task_gt_ids == cls)).sum().item()
                class_counts[cls] += (task_gt_ids == cls).sum().item()

            
            for i in range(len(mask_names)):
                # Store results in dictionary
                results[mask_names[i]] = {
                    f"{task_name}": task_preds[i].item(),
                    "instance_id": instance_ids[i]
                }
                
                # Store full logits
                logits_results[mask_names[i]] = {
                    "logits": task_logits[i].tolist(),  # Convert tensor to list
                    "instance_id": instance_ids[i]
                }
            
            # break    
              
    # Compute per-class accuracy & mean accuracy
    class_accuracies = {
        cls: class_correct[cls] / class_counts[cls] if class_counts[cls] > 0 else 0
        for cls in class_counts
    }
    mean_accuracy = np.mean(list(class_accuracies.values())) if class_accuracies else 0
    
    # Calculate accuracy
    task_accuracy = total_task_correct / total_samples

    if verbose:
        print(f"  {task_name} Accuracy: {task_accuracy:.2f}", flush=True)

    if store_results:
        if hasattr(config, "save_results_path"):
            # Save predictions to JSON
            with open(config.save_results_path, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Predictions saved to {config.save_results_path}", flush=True)

        # Save logits to separate JSON file
        if hasattr(config, "save_logits_path") :
            with open(config.save_logits_path, 'w') as f:
                json.dump(logits_results, f, indent=4)
                
            print(f"logits saved to {config.save_logits_path}", flush=True)

    return task_accuracy, mean_accuracy  #Return accuracy


# Predict loop
def predict_with_model_singletask_with_target_logits(config,
                       model, 
                       dataloader,
                       device='cuda',
                       store_results = True,
                       verbose=True):
        
    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)   
    task_name = config.task_name
    
    model = model.to(device)
    model.eval()
    results = {}  # Dictionary to store top-class predictions
    logits_results = {}  # Dictionary to store logits
    
    if verbose:
        print('began prediction...', flush=True)

    with torch.no_grad():
        for img, mask, target_logit, instrument_id, instance_id, mask_name, ground_truth_name in dataloader:
            img = img.to(device)
            mask = mask.to(device)
            target_logit = target_logit.to(device)
            instrument_id = instrument_id.to(device)
            
            # Forward pass
            task_logits = model(img, mask, target_logit, instrument_id)

            # Get predicted classes
            task_preds = torch.argmax(task_logits, dim=1)

            
            for i in range(len(mask_name)):
                # Store results top prediction
                results[mask_name[i]] = {
                    f"{task_name}": task_preds[i].item(),
                    "instance_id": instance_id[i]
                }
           
                # Store full logits
                logits_results[mask_name[i]] = {
                    f"logits_{task_name}": task_logits[i].tolist(),  # Convert tensor to list
                    "instance_id": instance_id[i]
                }   
                
            # break

                   
            
    if store_results:           
        if hasattr(config, "save_results_path"):
            # Save predictions to JSON
            with open(config.save_results_path, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"Predictions saved to {config.save_results_path}", flush=True)

        # Save logits to separate JSON file
        if hasattr(config, "save_logits_path") :
            with open(config.save_logits_path, 'w') as f:
                json.dump(logits_results, f, indent=4)
                
            print(f"logits saved to {config.save_logits_path}", flush=True)
        
        