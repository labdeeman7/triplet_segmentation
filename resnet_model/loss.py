import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self):
        super(MultiTaskLoss, self).__init__()
        self.criterion_verb = nn.CrossEntropyLoss()
        self.criterion_target = nn.CrossEntropyLoss()

    def forward(self, verb_preds, target_preds, verb_labels, target_labels):
        loss_verb = self.criterion_verb(verb_preds, verb_labels)
        loss_target = self.criterion_target(target_preds, target_labels)
        return loss_verb + loss_target