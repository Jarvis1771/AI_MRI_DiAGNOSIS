from torch.utils.data import DataLoader
from mri_dataset import MRIDataset

train_dataset = MRIDataset("data/mri/train")
val_dataset   = MRIDataset("data/mri/val")
test_dataset  = MRIDataset("data/mri/test")

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader  = DataLoader(test_dataset, batch_size=8, shuffle=False)

print("Train samples:", len(train_dataset))
print("Val samples:", len(val_dataset))
print("Test samples:", len(test_dataset))

images, labels = next(iter(train_loader))
print("Batch image shape:", images.shape)
print("Batch labels shape:", labels.shape)
