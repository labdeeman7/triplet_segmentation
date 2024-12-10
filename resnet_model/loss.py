import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.criterion_action = nn.CrossEntropyLoss()
        self.criterion_target = nn.CrossEntropyLoss()

    def forward(self, action_preds, target_preds, action_labels, target_labels):
        loss_action = self.criterion_action(action_preds, action_labels)
        loss_target = self.criterion_target(target_preds, target_labels)
        return loss_action + loss_target