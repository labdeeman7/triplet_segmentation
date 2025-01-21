import torch
import json
import os
import wandb
import types
from resnet_model.checkpoint_utils import save_checkpoint


def train_model_singletask(config,
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

    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)    

    for epoch in range(start_epoch, num_epochs):
        print(f'started epoch {epoch}, best_val_accuracy, {best_val_accuracy}', flush=True)
        model.train()
        running_loss = 0.0
        total_task_correct = 0
        total_samples = 0

        for img, mask, instrument_id, instance_id, task_gt_id, mask_name in train_loader:
            img = img.to(device)
            mask = mask.to(device)
            instrument_id = instrument_id.to(device)
            task_gt_id = task_gt_id.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            task_preds = model(img, mask, instrument_id)

            # Compute loss
            loss = loss_fn(task_preds, task_gt_id)   

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Get predicted classes
            task_preds = torch.argmax(task_preds, dim=1)
            
            # Compute accuracy
            total_task_correct += (task_preds == task_gt_id).sum().item()
            total_samples += img.size(0)
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        task_accuracy = total_task_correct / total_samples 
                        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}", flush=True)
            print(f"Train {task_name} Accuracy: {task_accuracy:.2f}", flush=True)
        
        
        # Log training metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            f"train_{task_name}_accuracy": task_accuracy,
        })    
        
            

        # Validate the model
        print('Validation')
        val_task_accuracy = test_model_with_evaluation_singletask(config,
                                                model, 
                                                val_loader, 
                                                device=device, 
                                                verbose=True)

        wandb.log({
            f"val_{task_name}_accuracy": val_task_accuracy,
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


def test_model_singletask(config,
               model, 
               dataloader, 
               device='cuda', 
               save_results_path='',
               verbose=True):
    
    if config.verb_and_target_gt_present_for_test:
        test_model_with_evaluation_singletask(config=config, 
                                model = model, 
                                dataloader = dataloader, 
                                device=device, 
                                save_results_path=save_results_path,
                                verbose=verbose )
    else:
        predict_with_model_singletask(config=config, 
                           model = model, 
                            dataloader = dataloader,   
                            device=device, 
                            save_results_path=save_results_path,
                            verbose=verbose )
        
            

def test_model_with_evaluation_singletask(config,
                            model, 
                            dataloader, 
                            device='cuda', 
                            save_results_path='',
                            verbose=True ):
    
    task_name = config.task_name
    model.eval()
    total_task_correct = 0
    total_samples = 0
    results = {}

    with torch.no_grad():
        for img, mask, instrument_id, instance_id, task_gt_id, mask_name in dataloader:
            img = img.to(device)
            mask = mask.to(device)
            instrument_id = instrument_id.to(device)
            task_gt_id = task_gt_id.to(device)

            task_preds = model(img, mask, instrument_id)

            # Get predicted classes
            task_preds = torch.argmax(task_preds, dim=1)

            # Compute accuracy
            total_task_correct += (task_preds == task_gt_id).sum().item()
            total_samples += img.size(0)

            # Store results in dictionary
            for i in range(len(mask_name)):
                results[mask_name[i]] = {
                    f"{task_name}": task_preds[i].item(),
                    "instance_id": instance_id[i]
                }

    # Calculate accuracy
    task_accuracy = total_task_correct / total_samples

    if verbose:
        print(f"  {task_name} Accuracy: {task_accuracy:.2f}", flush=True)

    if save_results_path:  # Save predictions to JSON
        with open(save_results_path, 'w') as f:
            json.dump(results, f, indent=4)

    return task_accuracy  #Return accuracy


# Predict loop
def predict_with_model_singletask(config,
                       model, 
                       dataloader,
                       save_results_path='',
                       device='cuda',
                       verbose=True):
    
    
    
    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)   
    task_name = config.task_name
    
    model = model.to(device)
    model.eval()
    results = {}  # Dictionary to store predictions
    
    if verbose:
        print('began prediction...', flush=True)

    with torch.no_grad():
        for img, mask, instrument_id, instance_id, mask_name in dataloader:
            img = img.to(device)
            mask = mask.to(device)
            instrument_id = instrument_id.to(device)

            # Perform predictions
            task_preds = model(img, mask, instrument_id)

            # Get predicted classes
            task_preds = torch.argmax(task_preds, dim=1)

            # Store results
            for i in range(len(mask_name)):
                results[mask_name[i]] = {
                    f"{task_name}": task_preds[i].item(),
                    "instance_id": instance_id[i]
                }
    
    # Save predictions to JSON
    with open(save_results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Predictions saved to {save_results_path}", flush=True)
