from torch.utils.data import  Dataset

# Custom Dataset
class SurgicalDataset(Dataset):
    def __init__(self, image_paths, instrument_annotations, actions, targets, transform=None):
        self.image_paths = image_paths
        self.instrument_annotations = instrument_annotations
        self.actions = actions
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        instrument_annotation = self.instrument_annotations[idx]  # Assume it's a tensor
        action = self.actions[idx]
        target = self.targets[idx]

        return image, instrument_annotation, action, target