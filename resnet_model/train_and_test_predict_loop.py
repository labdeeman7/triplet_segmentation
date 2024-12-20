import torch
import json

def train_model(model, dataloader, optimizer, loss_fn, num_epochs=10, device='cuda'):
    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for img, mask, instrument_id, instance_id, verb_id, target_id, mask_name in dataloader:
            img = img.to(device)
            instrument_id = instrument_id.to(device)
            verb_id = verb_id.to(device)
            target_id = target_id.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # Forward pass
            verb_preds, target_preds = model(img, mask, instrument_id)

            # Compute loss
            loss = loss_fn(verb_preds, target_preds, verb_id, target_id)

            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(dataloader)}")



# Test Loop
def test_model(model, dataloader, output_json_path, save_model_path=None, device='cuda'):
    model.eval()
    total_verb_correct = 0
    total_target_correct = 0
    total_samples = 0
    results = {} 

    with torch.no_grad():
        for img, mask, instrument_id, instance_id, verb_id, target_id, mask_name in dataloader:
            img = img.to(device)
            instrument_id = instrument_id.to(device)
            verb_id = verb_id.to(device)
            target_id = target_id.to(device)
            mask = mask.to(device)

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

    # Save results to JSON file
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)
    
    # Save the model (optional)
    if save_model_path:
        torch.save(model.state_dict(), save_model_path)
        print(f"Model saved to {save_model_path}")    
        
    print(f"verb Accuracy: {total_verb_correct / total_samples:.2f}")
    print(f"Target Accuracy: {total_target_correct / total_samples:.2f}")



# Predict loop
def predict_with_model(model, dataloader, output_json_path, device='cuda'):
    model.eval()
    results = {}  # Dictionary to store predictions
    
    print('began prediction...')

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
    with open(output_json_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Predictions saved to {output_json_path}")
