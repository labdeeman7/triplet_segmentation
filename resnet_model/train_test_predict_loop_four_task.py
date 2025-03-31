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

def train_model_fourtask(config,
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

        running_total_loss = 0.0
        running_loss_verb = 0.0
        running_loss_target = 0.0
        running_loss_verbtarget = 0.0
        running_loss_ivt = 0.0
        
        
        total_correct = {'verb': 0, 'target': 0, 'verbtarget': 0, 'ivt': 0}
        total_samples = 0

        # Track per-class accuracy
        class_correct = {'verb': defaultdict(int), 'target': defaultdict(int), 'verbtarget': defaultdict(int), 'ivt': defaultdict(int)}
        class_counts = {'verb': defaultdict(int), 'target': defaultdict(int), 'verbtarget': defaultdict(int), 'ivt': defaultdict(int)}

        for (imgs, masks, mask_names, instance_ids, instrument_ids, verb_gt_ids, 
             target_gt_ids, verbtarg_gt_ids, gt_ivt_ids, instrument_softmax ) in train_loader:
            
            imgs, masks = imgs.to(device), masks.to(device)
            instrument_ids = instrument_ids.to(device)
            verb_gt_ids = verb_gt_ids.to(device)
            target_gt_ids = target_gt_ids.to(device)
            verbtarg_gt_ids = verbtarg_gt_ids.to(device)
            gt_ivt_ids = gt_ivt_ids.to(device)
            instrument_softmax = instrument_softmax.to(device)
            
            optimizer.zero_grad()

            # Forward pass
            verb_logits, target_logits, verbtarg_logits, ivt_logits = model(imgs, masks, instrument_softmax)
                          
            # Get loss
            loss, loss_breakdown_dict = loss_fn(verb_logits, target_logits, verbtarg_logits, ivt_logits, 
                                verb_gt_ids, target_gt_ids, verbtarg_gt_ids, gt_ivt_ids)
            
            
            # print("Loss before backward:", loss.item())
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            running_total_loss += loss.item()
            running_loss_verb += loss_breakdown_dict["loss_verb"]
            running_loss_target += loss_breakdown_dict["loss_target"]
            running_loss_verbtarget += loss_breakdown_dict["loss_verbtarget"]
            running_loss_ivt += loss_breakdown_dict["loss_ivt"]


            # Get predictions
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)
            verbtarg_preds = torch.argmax(verbtarg_logits, dim=1)
            ivt_preds = torch.argmax(ivt_logits, dim=1)

            # Compute accuracy
            total_correct['verb'] += (verb_preds == verb_gt_ids).sum().item()
            total_correct['target'] += (target_preds == target_gt_ids).sum().item()
            total_correct['verbtarget'] += (verbtarg_preds == verbtarg_gt_ids).sum().item()
            total_correct['ivt'] += (ivt_preds == gt_ivt_ids).sum().item()
            total_samples += imgs.size(0)

            # Compute per-class accuracy
            for task, preds, gt_ids in zip(['verb', 'target', 'verbtarget', 'ivt'], 
                                           [verb_preds, target_preds, verbtarg_preds, ivt_preds], 
                                           [verb_gt_ids, target_gt_ids, verbtarg_gt_ids, gt_ivt_ids]):
                for cls in torch.unique(gt_ids):
                    cls = cls.item()
                    class_correct[task][cls] += ((preds == cls) & (gt_ids == cls)).sum().item()
                    class_counts[task][cls] += (gt_ids == cls).sum().item()
            
                     


        # Compute per-class accuracy & mean accuracy
        mean_accuracies = {
            task: np.mean([
                class_correct[task][cls] / class_counts[task][cls] if class_counts[task][cls] > 0 else 0
                for cls in class_counts[task]
            ]) for task in ['verb', 'target', 'verbtarget', 'ivt']
        }
        
        # Calculate total accuracy
        train_loss = running_total_loss / len(train_loader)
        task_accuracies = {task: total_correct[task] / total_samples for task in total_correct}

        if verbose:
            print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}", flush=True)
            for task in task_accuracies:
                print(f"Train {task} Accuracy: {task_accuracies[task]:.2f}", flush=True)

        # Log training metrics to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_loss_verb": running_loss_verb / len(train_loader),
            "train_loss_target": running_loss_target / len(train_loader),
            "train_loss_verbtarget": running_loss_verbtarget / len(train_loader),
            "train_loss_ivt": running_loss_ivt / len(train_loader),
            **{f"train_{task}_accuracy": acc for task, acc in task_accuracies.items()},
            **{f"train_{task}_mean_accuracy": acc for task, acc in mean_accuracies.items()}
        })

        # Validate the model
        print('Validation')
        val_accuracies = validation_fourtask(config, model, val_loader, device=device, verbose=True)

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
        if os.path.exists(previous_saved_model_path) and (epoch % 5 != 0):
            os.remove(previous_saved_model_path)




def validation_fourtask(config,
                         model, 
                         dataloader, 
                         device='cuda', 
                         verbose=True):
    
    model.eval()
    total_correct = {'verb': 0, 'target': 0, 'verbtarget': 0, 'ivt': 0}
    total_samples = 0

    with torch.no_grad():
        for (imgs, masks, mask_names, instance_ids, instrument_ids, verb_gt_ids, 
             target_gt_ids, verbtarg_gt_ids, ivt_gt_ids, instrument_softmax) in dataloader:
            
            imgs = imgs.to(device)
            masks = masks.to(device)
            instrument_softmax = instrument_softmax.to(device)
            verb_gt_ids = verb_gt_ids.to(device)
            target_gt_ids = target_gt_ids.to(device)
            verbtarg_gt_ids = verbtarg_gt_ids.to(device)
            ivt_gt_ids = ivt_gt_ids.to(device)

            # Forward pass
            verb_logits, target_logits, verbtarg_logits, ivt_logits = model(
                imgs, masks, instrument_softmax
            )

            # Predictions
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)
            verbtarg_preds = torch.argmax(verbtarg_logits, dim=1)
            ivt_preds = torch.argmax(ivt_logits, dim=1)

            # Accuracy tracking
            total_correct['verb'] += (verb_preds == verb_gt_ids).sum().item()
            total_correct['target'] += (target_preds == target_gt_ids).sum().item()
            total_correct['verbtarget'] += (verbtarg_preds == verbtarg_gt_ids).sum().item()
            total_correct['ivt'] += (ivt_preds == ivt_gt_ids).sum().item()
            total_samples += imgs.size(0)
            
            #remove
             

    # Calculate accuracies
    task_accuracies = {task: total_correct[task] / total_samples for task in total_correct}

    if verbose:
        print(f"Validation Accuracies: {task_accuracies}", flush=True)

    return task_accuracies


# Come back here. 
def predict_with_model_fourtask(config,
                                 model, 
                                 dataloader,
                                 device='cuda',
                                 store_results=True,
                                 verbose=True):

    
    os.makedirs(config.work_dir, exist_ok=True)   
    os.makedirs(config.vis_dir, exist_ok=True)  
    
    model = model.to(device)
    model.eval()

    results = {}
    logits_results = {}
    
    if verbose:
        print("Began prediction...", flush=True)


    # Get mappings
    verb_id_to_name = TripletSegmentationVariables.categories['verb']
    target_id_to_name = TripletSegmentationVariables.categories['target']
    verbtarg_id_to_name = TripletSegmentationVariables.categories['verbtarget']
    ivt_id_to_name = TripletSegmentationVariables.categories['triplet']

    # Initialize accuracy tracking
    total_correct = {"verb": 0, "target": 0, "verbtarget": 0, "ivt": 0}
    total_samples = 0
     
    

    with torch.no_grad():
        for img, mask, instrument_softmax, instance_id, ann_base in dataloader:
            img, mask = img.to(device), mask.to(device)
            instrument_softmax = instrument_softmax.to(device)

            # Model prediction
            verb_logits, target_logits, verbtarg_logits, ivt_logits = model(img, mask, instrument_softmax)

            # Get predicted local class IDs
            verb_preds = torch.argmax(verb_logits, dim=1)
            target_preds = torch.argmax(target_logits, dim=1)
            verbtarg_preds = torch.argmax(verbtarg_logits, dim=1)
            ivt_preds = torch.argmax(ivt_logits, dim=1)


            for i in range(len(ann_base)):
                mask_base_name = ann_base[i]
                if verbose:
                    print(f"Predicting for {mask_base_name}")


                # Visualization & storing results
                for task, pred, logits, id_to_name in [
                    ("verb", verb_preds[i].item(), verb_logits[i], verb_id_to_name),
                    ("target", target_preds[i].item(), target_logits[i], target_id_to_name),
                    ("verbtarget", verbtarg_preds[i].item(), verbtarg_logits[i], verbtarg_id_to_name),
                    ("ivt", ivt_preds[i].item(), ivt_logits[i], ivt_id_to_name),
                ]:
                    # No need for a visualization?
                    
                    # Save prediction
                    results.setdefault(mask_base_name, {})[task] = pred
                    logits_results.setdefault(mask_base_name, {})[f"logits_{task}"] = logits.cpu().tolist()
                
            #remove
            total_samples += img.size(0)
                

    # Save results
    if store_results:
        if hasattr(config, "save_results_path"):
            with open(config.save_results_path, "w") as f:
                json.dump(results, f, indent=2)
            print(f"Predictions saved to {config.save_results_path}")

        if hasattr(config, "save_logits_path"):
            with open(config.save_logits_path, "w") as f:
                json.dump(logits_results, f, indent=2)
            print(f"Logits saved to {config.save_logits_path}")

    return
    

