def test_model(model, dataloader, device='cuda'):
    model = model.to(device)
    model.eval()
    total_correct_action, total_correct_target = 0, 0
    total_samples = 0

    with torch.no_grad():
        for images, instrument_annotations, actions, targets in dataloader:
            images, instrument_annotations, actions, targets = (
                images.to(device),
                instrument_annotations.to(device),
                actions.to(device),
                targets.to(device),
            )

            action_preds, target_preds = model(images, instrument_annotations)
            _, action_preds_classes = torch.max(action_preds, dim=1)
            _, target_preds_classes = torch.max(target_preds, dim=1)

            total_correct_action += (action_preds_classes == actions).sum().item()
            total_correct_target += (target_preds_classes == targets).sum().item()
            total_samples += actions.size(0)

    print(f"Action Accuracy: {total_correct_action / total_samples:.2f}")
    print(f"Target Accuracy: {total_correct_target / total_samples:.2f}")