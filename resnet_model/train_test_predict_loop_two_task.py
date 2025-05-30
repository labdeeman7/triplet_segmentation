import torch
import json
import os
import wandb
import types
from resnet_model.checkpoint_utils import save_checkpoint
from collections import defaultdict
import numpy as np

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
        print(f'started epoch {epoch+1}, best_val_accuracy, {best_val_accuracy}', flush=True)
        model.train()
        running_loss = 0.0
        total_verb_correct = 0
        total_target_correct = 0
        total_samples = 0
        
        # For mean accuracy
        class_verb_correct = defaultdict(int)
        class_verb_counts = defaultdict(int)
        class_target_correct = defaultdict(int)
        class_target_counts = defaultdict(int)

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
            
            # Compute per-class accuracy
            # verb
            for cls in torch.unique(verb_id):
                cls = cls.item()
                class_verb_correct[cls] += ((verb_preds == cls) & (verb_id == cls)).sum().item()
                class_verb_counts[cls] += (verb_id == cls).sum().item()
            
            
            for cls in torch.unique(target_id):
                cls = cls.item()
                class_target_correct[cls] += ((target_preds == cls) & (target_id == cls)).sum().item()
                class_target_counts[cls] += (target_id == cls).sum().item()  
            
                  
            # Remove
        
        # Calculate metrics
        #Compute per-class accuracy & mean accuracy
        verb_class_accuracies = {
            cls: class_verb_correct[cls] / class_verb_counts[cls] if class_verb_counts[cls] > 0 else 0
            for cls in class_verb_counts
        }
        mean_verb_accuracy = np.mean(list(verb_class_accuracies.values()))
        
        target_class_accuracies = {
            cls: class_target_correct[cls] / class_target_counts[cls] if class_target_counts[cls] > 0 else 0
            for cls in class_target_counts
        }
        mean_target_accuracy = np.mean(list(target_class_accuracies.values()))
        
        #Calculate accuracy
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
            "train_verb_mean_accuracy": mean_verb_accuracy,
            "train_target_mean_accuracy": mean_target_accuracy,
        })    
        
            

        # Validate the model
        print('Validation')
        val_verb_accuracy, val_target_accuracy, val_mean_verb_accuracy, val_mean_target_accuracy   = test_model_with_evaluation_multitask(config,
                                                                                                                                        model, 
                                                                                                                                        val_loader, 
                                                                                                                                        device=device, 
                                                                                                                                        store_results=False,
                                                                                                                                        verbose=True)

        average_val_accuracy = (val_verb_accuracy + val_target_accuracy)
        average_val_mean_accuracy = (val_verb_accuracy + val_target_accuracy)
        
        if verbose:
            print(f"Val Verb Accuracy: {val_verb_accuracy:.2f}", flush=True)
            print(f"Val Verb mean Accuracy: {val_mean_verb_accuracy:.2f}", flush=True)
            print(f"Val Target Accuracy: {val_target_accuracy:.2f}", flush=True)
            print(f"Val Target mean Accuracy: {val_mean_target_accuracy:.2f}", flush=True)
        
        
        wandb.log({
            "val_verb_accuracy": val_verb_accuracy,
            "val_verb_mean_accuracy": val_mean_verb_accuracy,
            "val_target_accuracy": val_target_accuracy,
            "val_target_mean_accuracy": val_mean_target_accuracy,
            "average_val_accuracy": average_val_accuracy,
            "average_val_mean_accuracy": average_val_mean_accuracy
        })
        
        
        # Save the best model
        # going back to saving with val_accuracy. 
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
               store_results = True,
               verbose=True):
    
    if config.verb_and_target_gt_present_for_test:
        test_model_with_evaluation_multitask(config=config, 
                                model = model, 
                                dataloader = dataloader, 
                                device=device, 
                                store_results=store_results,
                                verbose=verbose )
    else:
        predict_with_model_multitask(config=config, 
                           model = model, 
                            dataloader = dataloader,   
                            device=device, 
                            store_results=store_results,
                            verbose=verbose )
        
def test_model_with_evaluation_multitask(config,
                            model, 
                            dataloader, 
                            device='cuda',
                            store_results = True,
                            verbose=True ):
    
    
    model.eval()
    total_verb_correct = 0
    total_target_correct = 0
    total_samples = 0
    # For mean accuracy
    class_verb_correct = defaultdict(int)
    class_verb_counts = defaultdict(int)
    class_target_correct = defaultdict(int)
    class_target_counts = defaultdict(int)
    results = {}
    logits_results = {}  # Dictionary to store logits

    with torch.no_grad():
        for img, mask, instrument_id, instance_id, verb_id, target_id, mask_name in dataloader:
            img = img.to(device)
            mask = mask.to(device)
            instrument_id = instrument_id.to(device)
            verb_id = verb_id.to(device)
            target_id = target_id.to(device)

            verb_logits, target_logits = model(img, mask, instrument_id)

            # Get predicted classes
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)

            # Compute accuracy
            total_verb_correct += (verb_preds == verb_id).sum().item()
            total_target_correct += (target_preds == target_id).sum().item()
            total_samples += img.size(0)
            
            # Compute per-class accuracy
            # verb
            for cls in torch.unique(verb_id):
                cls = cls.item()
                class_verb_correct[cls] += ((verb_preds == cls) & (verb_id == cls)).sum().item()
                class_verb_counts[cls] += (verb_id == cls).sum().item()
            
            
            for cls in torch.unique(target_id):
                cls = cls.item()
                class_target_correct[cls] += ((target_preds == cls) & (target_id == cls)).sum().item()
                class_target_counts[cls] += (target_id == cls).sum().item()   

            
            for i in range(len(mask_name)):
                # Store results in dictionary
                results[mask_name[i]] = {
                    "verb": verb_preds[i].item(),
                    "target": target_preds[i].item(),
                    "instance_id": instance_id[i]
                }
                
                # Store full logits
                logits_results[mask_name[i]] = {
                    "logits_verb": verb_logits[i].tolist(),  # Convert tensor to list
                    "logits_target": target_logits[i].tolist(),  # Convert tensor to list
                    "instance_id": instance_id[i]
                }
            
            # Remove

    # Calculate metrics
    #Compute per-class accuracy & mean accuracy
    verb_class_accuracies = {
        cls: class_verb_correct[cls] / class_verb_counts[cls] if class_verb_counts[cls] > 0 else 0
        for cls in class_verb_counts
    }
    mean_verb_accuracy = np.mean(list(verb_class_accuracies.values()))
    
    target_class_accuracies = {
        cls: class_target_correct[cls] / class_target_counts[cls] if class_target_counts[cls] > 0 else 0
        for cls in class_target_counts
    }
    mean_target_accuracy = np.mean(list(target_class_accuracies.values()))
    
    # Calculate accuracy
    verb_accuracy = total_verb_correct / total_samples
    target_accuracy = total_target_correct / total_samples

    if verbose:
        print(f"  Verb Accuracy: {verb_accuracy:.2f}", flush=True)
        print(f"  Target Accuracy: {target_accuracy:.2f}", flush=True) 

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

    return verb_accuracy, target_accuracy, mean_verb_accuracy, mean_target_accuracy  #Return both accuracy

# Predict loop
def predict_with_model_multitask(config,
                       model, 
                       dataloader,
                       device='cuda',
                       store_results = True,
                       verbose=True):
    
    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)   
    
    model = model.to(device)
    model.eval()
    results = {}  # Dictionary to store top-class predictions
    logits_results = {}  # Dictionary to store logits
    
    if verbose:
        print('began prediction...', flush=True)

    with torch.no_grad():
        for img, mask, instrument_id, instance_id, mask_name, ground_truth_name in dataloader:
            img = img.to(device)
            mask = mask.to(device)
            instrument_id = instrument_id.to(device)

            # Perform predictions
            verb_logits, target_logits = model(img, mask, instrument_id)

            # Get predicted classes
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)

            
            for i in range(len(mask_name)):
                
                # Store results top prediction
                results[mask_name[i]] = {
                    "verb": verb_preds[i].item(),
                    "target": target_preds[i].item(),
                    "instance_id": instance_id[i]
                }
                
                # Store full logits
                logits_results[mask_name[i]] = {
                    "logits_verb": verb_logits[i].tolist(),  # Convert tensor to list
                    "logits_target": target_logits[i].tolist(),  # Convert tensor to list
                    "instance_id": instance_id[i]
                }
            
            # Remove 
    
    # Save predictions to JSON
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
