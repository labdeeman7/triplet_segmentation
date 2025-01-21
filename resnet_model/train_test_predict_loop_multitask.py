import torch
import json
import os
import wandb
import types
from resnet_model.checkpoint_utils import save_checkpoint

def train_model_multitask(config,
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
    
    model = model.to(device)

    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)    

    for epoch in range(start_epoch, num_epochs):
        print(f'started epoch {epoch}', flush=True)
        model.train()
        running_loss = 0.0
        total_verb_correct = 0
        total_target_correct = 0
        total_samples = 0

        for img, mask, instrument_id, instance_id, verb_id, target_id, mask_name in train_loader:
            img = img.to(device)
            mask = mask.to(device)
            instrument_id = instrument_id.to(device)
            verb_id = verb_id.to(device)
            target_id = target_id.to(device)

            optimizer.zero_grad()

            # Forward pass
            verb_preds, target_preds = model(img, mask, instrument_id)

            # Compute loss
            loss = loss_fn(verb_preds, target_preds, verb_id, target_id)   

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            # Get predicted classes
            verb_preds = torch.argmax(verb_preds, dim=1)
            target_preds = torch.argmax(target_preds, dim=1)
            
            # Compute accuracy
            total_verb_correct += (verb_preds == verb_id).sum().item()
            total_target_correct += (target_preds == target_id).sum().item()
            total_samples += img.size(0)
        
        # Calculate metrics
        train_loss = running_loss / len(train_loader)
        verb_accuracy = total_verb_correct / total_samples
        target_accuracy = total_target_correct / total_samples    
                        
        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}", flush=True)
            print(f"Train Verb Accuracy: {verb_accuracy:.2f}", flush=True)
            print(f"Train Target Accuracy: {target_accuracy:.2f}", flush=True)
        
        
        # Log training metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_verb_accuracy": verb_accuracy,
            "train_target_accuracy": target_accuracy,
        })    
        
            

        # Validate the model
        print('Validation')
        val_verb_accuracy, val_target_accuracy = test_model_with_evaluation_multitask(config,
                                                model, 
                                                val_loader, 
                                                device=device, 
                                                verbose=True)

        average_val_accuracy = (val_verb_accuracy + val_target_accuracy)
        wandb.log({
            "val_verb_accuracy": val_verb_accuracy,
            "val_target_accuracy": val_target_accuracy,
            "average_val_accuracy": average_val_accuracy,
        })
        
        
        # Save the best model
        if average_val_accuracy > best_val_accuracy:
            best_val_accuracy = average_val_accuracy
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

def test_model_multitask(config,
               model, 
               dataloader, 
               device='cuda', 
               save_results_path='',
               verbose=True):
    
    if config.verb_and_target_gt_present_for_test:
        test_model_with_evaluation_multitask(config=config, 
                                model = model, 
                                dataloader = dataloader, 
                                device=device, 
                                save_results_path=save_results_path,
                                verbose=verbose )
    else:
        predict_with_model_multitask(config=config, 
                           model = model, 
                            dataloader = dataloader,   
                            device=device, 
                            save_results_path=save_results_path,
                            verbose=verbose )
        
def test_model_with_evaluation_multitask(config,
                            model, 
                            dataloader, 
                            device='cuda', 
                            save_results_path='',
                            verbose=True ):
    
    
    model.eval()
    total_verb_correct = 0
    total_target_correct = 0
    total_samples = 0
    results = {}

    with torch.no_grad():
        for img, mask, instrument_id, instance_id, verb_id, target_id, mask_name in dataloader:
            img = img.to(device)
            mask = mask.to(device)
            instrument_id = instrument_id.to(device)
            verb_id = verb_id.to(device)
            target_id = target_id.to(device)

            verb_preds, target_preds = model(img, mask, instrument_id)

            # Get predicted classes
            verb_preds = torch.argmax(verb_preds, dim=1)
            target_preds = torch.argmax(target_preds, dim=1)

            # Compute accuracy
            total_verb_correct += (verb_preds == verb_id).sum().item()
            total_target_correct += (target_preds == target_id).sum().item()
            total_samples += img.size(0)

            # Store results in dictionary
            for i in range(len(mask_name)):
                results[mask_name[i]] = {
                    "verb": verb_preds[i].item(),
                    "target": target_preds[i].item(),
                    "instance_id": instance_id[i]
                }

    # Calculate accuracy
    verb_accuracy = total_verb_correct / total_samples
    target_accuracy = total_target_correct / total_samples

    if verbose:
        print(f"  Verb Accuracy: {verb_accuracy:.2f}", flush=True)
        print(f"  Target Accuracy: {target_accuracy:.2f}", flush=True) 

    if save_results_path:  # Save predictions to JSON
        with open(save_results_path, 'w') as f:
            json.dump(results, f, indent=4)

    return verb_accuracy, target_accuracy  #Return both accuracy

# Predict loop
def predict_with_model_multitask(config,
                       model, 
                       dataloader,
                       save_results_path='',
                       device='cuda',
                       verbose=True):
    
    
    
    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)   
    
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
            verb_preds, target_preds = model(img, mask, instrument_id)

            # Get predicted classes
            verb_preds = torch.argmax(verb_preds, dim=1)
            target_preds = torch.argmax(target_preds, dim=1)

            # Store results
            for i in range(len(mask_name)):
                results[mask_name[i]] = {
                    "verb": verb_preds[i].item(),
                    "target": target_preds[i].item(),
                    "instance_id": instance_id[i]
                }
    
    # Save predictions to JSON
    with open(save_results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Predictions saved to {save_results_path}", flush=True)
