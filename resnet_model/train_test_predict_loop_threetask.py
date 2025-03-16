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

import torch
import torch.nn as nn
import os
import wandb
import json
import numpy as np
from collections import defaultdict

def train_model_threetask(config,
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
    loss_fn = loss_fn.to(device)

    # Create the save directory if it doesn't exist
    os.makedirs(config.work_dir, exist_ok=True)    

    for epoch in range(start_epoch, num_epochs):
        print(f'Started epoch {epoch+1}, best_val_accuracy: {best_val_accuracy}', flush=True)
        model.train()

        running_loss = 0.0
        total_correct = {'verb': 0, 'target': 0, 'verbtarget': 0}
        total_samples = 0

        # Track per-class accuracy
        class_correct = {'verb': defaultdict(int), 'target': defaultdict(int), 'verbtarget': defaultdict(int)}
        class_counts = {'verb': defaultdict(int), 'target': defaultdict(int), 'verbtarget': defaultdict(int)}

        for imgs, masks, instrument_ids, instance_ids, verb_gt_ids, target_gt_ids, verbtarg_gt_ids, mask_names in train_loader:
            
            imgs, masks = imgs.to(device), masks.to(device)
            instrument_ids = instrument_ids.to(device)
            verb_gt_ids = verb_gt_ids.to(device)
            target_gt_ids = target_gt_ids.to(device)
            verbtarg_gt_ids = verbtarg_gt_ids.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            verb_logits, target_logits, verbtarg_logits = model(imgs, masks, instrument_ids)
            
            # print('verb_logits', verb_logits)
            # print('target_logits', target_logits)
            # print('verbtarg_logits', verbtarg_logits)

            # Get loss
            loss = loss_fn(verb_logits, target_logits, verbtarg_logits, 
                                verb_gt_ids, target_gt_ids, verbtarg_gt_ids)
            
            
            
            
            # print("Loss before backward:", loss.item())
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Get predictions
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)
            verbtarg_preds = torch.argmax(verbtarg_logits, dim=1)

            # Compute accuracy
            total_correct['verb'] += (verb_preds == verb_gt_ids).sum().item()
            total_correct['target'] += (target_preds == target_gt_ids).sum().item()
            total_correct['verbtarget'] += (verbtarg_preds == verbtarg_gt_ids).sum().item()
            total_samples += imgs.size(0)

            # Compute per-class accuracy
            for task, preds, gt_ids in zip(['verb', 'target', 'verbtarget'], 
                                           [verb_preds, target_preds, verbtarg_preds], 
                                           [verb_gt_ids, target_gt_ids, verbtarg_gt_ids]):
                for cls in torch.unique(gt_ids):
                    cls = cls.item()
                    class_correct[task][cls] += ((preds == cls) & (gt_ids == cls)).sum().item()
                    class_counts[task][cls] += (gt_ids == cls).sum().item()
            
            #remove
                     


        # Compute per-class accuracy & mean accuracy
        mean_accuracies = {
            task: np.mean([
                class_correct[task][cls] / class_counts[task][cls] if class_counts[task][cls] > 0 else 0
                for cls in class_counts[task]
            ]) for task in ['verb', 'target', 'verbtarget']
        }
        
        # Calculate total accuracy
        train_loss = running_loss / len(train_loader)
        task_accuracies = {task: total_correct[task] / total_samples for task in total_correct}

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}", flush=True)
            for task in task_accuracies:
                print(f"Train {task} Accuracy: {task_accuracies[task]:.2f}", flush=True)

        # Log training metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            **{f"train_{task}_accuracy": acc for task, acc in task_accuracies.items()},
            **{f"train_{task}_mean_accuracy": acc for task, acc in mean_accuracies.items()}
        })

        # Validate the model
        print('Validation')
        val_accuracies = validation_threetask(config, model, val_loader, device=device, verbose=True)

        wandb.log({
            **{f"val_{task}_accuracy": val_accuracies[task] for task in val_accuracies}
        })
        
        # Save the best model
        if val_accuracies["verbtarget"] > best_val_accuracy:
            best_val_accuracy = val_accuracies["verbtarget"]
            best_model_path = os.path.join(config.work_dir, "best_model.pth")
            save_checkpoint(model, optimizer, epoch, best_val_accuracy, best_model_path)
            print(f"New best model saved to {best_model_path} with accuracy {best_val_accuracy:.2f}", flush=True)

        # Save the latest model
        latest_model_path = os.path.join(config.work_dir, f"epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch, best_val_accuracy, latest_model_path)

        # Delete previous checkpoint
        previous_saved_model_path = os.path.join(config.work_dir, f"epoch_{epoch}.pth")
        if os.path.exists(previous_saved_model_path):
            os.remove(previous_saved_model_path)




def validation_threetask(config,
                         model, 
                         dataloader, 
                         device='cuda', 
                         verbose=True):
    
    model.eval()
    total_correct = {'verb': 0, 'target': 0, 'verbtarget': 0}
    total_samples = 0

    with torch.no_grad():
        for imgs, masks, instrument_ids, instance_ids, verb_gt_ids, target_gt_ids, verbtarg_gt_ids, mask_names in dataloader:
            imgs, masks = imgs.to(device), masks.to(device)
            instrument_ids = instrument_ids.to(device)
            verb_gt_ids = verb_gt_ids.to(device)
            target_gt_ids = target_gt_ids.to(device)
            verbtarg_gt_ids = verbtarg_gt_ids.to(device)

            verb_logits, target_logits, verbtarg_logits = model(imgs, masks, instrument_ids)

            # Get predictions
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)
            verbtarg_preds = torch.argmax(verbtarg_logits, dim=1)

            # Compute accuracy
            total_correct['verb'] += (verb_preds == verb_gt_ids).sum().item()
            total_correct['target'] += (target_preds == target_gt_ids).sum().item()
            total_correct['verbtarget'] += (verbtarg_preds == verbtarg_gt_ids).sum().item()
            total_samples += imgs.size(0)
            
            #remove
             

    # Calculate accuracies
    task_accuracies = {task: total_correct[task] / total_samples for task in total_correct}

    if verbose:
        print(f"Validation Accuracies: {task_accuracies}", flush=True)

    return task_accuracies



def predict_with_model_threetask(config,
                                 model, 
                                 dataloader,
                                 device='cuda',
                                 store_results=True,
                                 verbose=True):
    instrument_to_verb_classes = config.instrument_to_verb_classes
    instrument_to_target_classes = config.instrument_to_target_classes
    instrument_to_verbtarget_classes = config.instrument_to_verbtarget_classes

    
    os.makedirs(config.work_dir, exist_ok=True)   
    os.makedirs(config.vis_dir, exist_ok=True)  

    # Create directories for separate task visualizations
    vis_dirs = {
        "verb": os.path.join(config.vis_dir, "verb_vis"),
        "target": os.path.join(config.vis_dir, "target_vis"),
        "verbtarget": os.path.join(config.vis_dir, "verbtarget_vis"),
    }
    for vis_dir in vis_dirs.values():
        os.makedirs(vis_dir, exist_ok=True)

    # Get mappings
    verb_id_to_name = TripletSegmentationVariables.categories['verb']
    target_id_to_name = TripletSegmentationVariables.categories['target']
    verbtarg_id_to_name = TripletSegmentationVariables.categories['verbtarget']

    verb_name_to_id = {v: k for k, v in verb_id_to_name.items()}
    target_name_to_id = {v: k for k, v in target_id_to_name.items()}
    verbtarg_name_to_id = {v: k for k, v in verbtarg_id_to_name.items()}

    # Initialize accuracy tracking
    total_correct = {"verb": 0, "target": 0, "verbtarget": 0}
    total_samples = 0
    class_correct = {"verb": defaultdict(int), "target": defaultdict(int), "verbtarget": defaultdict(int)}
    class_counts = {"verb": defaultdict(int), "target": defaultdict(int), "verbtarget": defaultdict(int)}

    model = model.to(device)
    model.eval()

    results = {}
    logits_results = {}

    if verbose:
        print("Began prediction...", flush=True)

    with torch.no_grad():
        for img, mask, instrument_id, instance_id, mask_name, ground_truth_name in dataloader:
            img, mask = img.to(device), mask.to(device)
            instrument_id = instrument_id.to(device)

            # Perform predictions
            verb_logits, target_logits, verbtarg_logits = model(img, mask, instrument_id)

            # Get predicted local class IDs
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)
            verbtarg_preds = torch.argmax(verbtarg_logits, dim=1)

            for i in range(len(mask_name)):
                instr_id = instrument_id[i].item()
                
                gt_name_single = ground_truth_name[i]
                
                if gt_name_single: 
                    verb_gt_name, target_gt_name = gt_name_single.split(',')
                else:
                    verb_gt_name = None
                    target_gt_name = None   

                
                # Handle missing ground truth
                verb_gt = verb_gt_name if verb_gt_name != "None" else None
                target_gt = target_gt_name if target_gt_name != "None" else None
                verbtarg_gt = f"{verb_gt},{target_gt}" if verb_gt and target_gt else None


                # Convert ground truth names to global IDs
                verb_gt_id = int(verb_name_to_id[verb_gt]) - 1 if verb_gt else None
                target_gt_id = int(target_name_to_id[target_gt]) - 1 if target_gt else None
                verbtarg_gt_id = int(verbtarg_name_to_id[verbtarg_gt]) - 1 if verbtarg_gt else None

                # Convert local predictions to global task IDs
                verb_global_id = instrument_to_verb_classes[instr_id][verb_preds[i].item()]
                target_global_id = instrument_to_target_classes[instr_id][target_preds[i].item()]
                verbtarg_global_id = instrument_to_verbtarget_classes[instr_id][verbtarg_preds[i].item()]

                # Visualization & storing results
                for task, pred_id, gt_id, gt_name, logits, vis_folder, id_to_name in [
                    ("verb", verb_global_id, verb_gt_id, verb_gt, verb_logits[i], vis_dirs["verb"], verb_id_to_name),
                    ("target", target_global_id, target_gt_id, target_gt, target_logits[i], vis_dirs["target"], target_id_to_name),
                    ("verbtarget", verbtarg_global_id, verbtarg_gt_id, verbtarg_gt, verbtarg_logits[i], vis_dirs["verbtarget"], verbtarg_id_to_name),
                ]:
                    
                    # prediction_name = f"Prediction: {pred_id} {id_to_name[str(pred_id + 1)]}"
                    # ground_truth_text = f"GT: {gt_id} {gt_name}" if gt_name else None
                    # save_path = os.path.join(vis_folder, f"{mask_name[i]}.png")

                    # save_visualization(img[i].cpu(), mask[i].cpu(), prediction_name, ground_truth_text, save_path)

                    # Accuracy Calculation
                    if gt_name:
                        is_correct = (pred_id == gt_id)
                        total_correct[task] += is_correct
                        class_correct[task][gt_id] += is_correct
                        class_counts[task][gt_id] += 1

                    # Store results
                    results.setdefault(mask_name[i], {})[task] = pred_id
                    logits_results.setdefault(mask_name[i], {})[f"logits_{task}"] = logits.tolist()
                
            #remove
                

    # Compute per-class accuracy and mean accuracy
    if total_samples > 0:
        class_accuracies = {
            task: {
                cls: class_correct[task][cls] / class_counts[task][cls] if class_counts[task][cls] > 0 else 0
                for cls in class_counts[task]
            } for task in ["verb", "target", "verbtarget"]
        }
        mean_accuracies = {
            task: np.mean(list(class_accuracies[task].values())) if class_accuracies[task] else 0
            for task in ["verb", "target", "verbtarget"]
        }
        task_accuracies = {task: total_correct[task] / total_samples for task in total_correct}

        if verbose:
            for task in task_accuracies:
                print(f"\n{task} Accuracy: {task_accuracies[task]:.2f}")
            print(f"Mean Class Accuracy: {mean_accuracies}")
            print(f"Class-wise Accuracy: {class_accuracies}")
    else:
        if verbose:
            print("\nNo ground truth provided for accuracy calculation.")

    # Save results if required
    if store_results:
        if hasattr(config, "save_results_path"):
            with open(config.save_results_path, "w") as f:
                json.dump(results, f, indent=4)
            print(f"Predictions saved to {config.save_results_path}")

        if hasattr(config, "save_logits_path"):
            with open(config.save_logits_path, "w") as f:
                json.dump(logits_results, f, indent=4)
            print(f"Logits saved to {config.save_logits_path}")
