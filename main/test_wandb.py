import wandb
import random

print('started')

# start a new wandb run to track this script
wandb.init(
    # set the wandb project where this run will be logged
    project="triplet_segmentation",

    # track hyperparameters and run metadata
    config={
    "description": 'testing wandb',    
    "learning_rate": 0.02,
    "architecture": "CNN",
    "dataset": "CIFAR-100",
    "epochs": 10,
    }
)

# simulate training
epochs = 10
offset = random.random() / 5
for epoch in range(2, epochs):
    acc = 1 - 2 ** -epoch - random.random() / epoch - offset
    loss = 2 ** -epoch + random.random() / epoch + offset

    print(epoch)
    print(acc)
    # log metrics to wandb
    wandb.log({"acc": acc, "loss": loss})

# [optional] finish the wandb run, necessary in notebooks
wandb.finish()