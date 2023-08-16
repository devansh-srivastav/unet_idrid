import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.utils as vutils
from dataloader import IDRiDDataset 
from loss import DiceLoss, DiceBCELoss
from unet import Unet
from glob import glob
import os
from utils import create_dir, seeding, epoch_time
import time
from augment import RandomRotate90, ApplyCLAHE, ImageEnhancer, RandomCrop, ToTensor

# Hyperparameters
batch_size = 8
learning_rate = 0.0001
num_epochs = 50
seeding(42)

create_dir("visualization")

# Data_transform for images and masks
data_transform = transforms.Compose([
    # transforms.Resize((512, 512)),
    # RandomCrop(512),
    RandomRotate90(),
    ImageEnhancer(color_jitter=True, green=False),
    ApplyCLAHE(),
    ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]),
])

# Dataset instance for training
path = './Segmentation.nosync'
train_image_paths = sorted(glob(os.path.join(path, "train_images", "*.jpg")))
train_mask_root_dirs = [os.path.join(path, "train_masks", "EX"), os.path.join(path, "train_masks", "HE"), os.path.join(path, "train_masks", "MA"), os.path.join(path, "train_masks", "SE")]
train_mask_extensions = ["tif", "tif", "tif", "tif"]
train_dataset = IDRiDDataset(train_image_paths, train_mask_root_dirs, train_mask_extensions, transform=data_transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Instantiate U-Net model
model = Unet()
model = model.double()

# Loss function
criterion = DiceBCELoss()

# Optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    start_time = time.time()
    for batch_images, batch_masks in train_loader:
        batch_images = batch_images.to(device).double()
        
        batch_masks = torch.stack(batch_masks, dim=1).to(device).double()

        optimizer.zero_grad()
        outputs = model(batch_images)

        loss = criterion(outputs, batch_masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    end_time = time.time()

    print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {total_loss / len(train_loader)}, Time: {epoch_time(start_time, end_time)}")

    
# Save visualization images
    if epoch == 49:
        with torch.no_grad():
            model.eval()
            for i, (batch_images, batch_masks) in enumerate(train_loader):
                batch_images = batch_images.to(device).double()
                batch_masks = torch.stack(batch_masks, dim=1).to(device).double()

                outputs = model(batch_images)

                num_images = batch_images.size(0)

                for j in range(4):
                    # Save original batch images
                    vutils.save_image(
                        batch_images,
                        f"visualization/epoch_{epoch+1}_batch_{i+1}_original.png",
                        nrow=num_images,
                        normalize=True,
                        scale_each=True
                    )

                    # # Save ground truth masks
                    # vutils.save_image(
                    #     batch_masks[:, j+1],
                    #     f"visualization/epoch_{epoch+1}_batch_{i+1}_gt_mask_{j+1}.png",
                    #     nrow=num_images,
                    #     normalize=True,
                    #     scale_each=True
                    # )

                    # Save predicted masks
                    vutils.save_image(
                        outputs[:, j:j+1],
                        f"visualization/epoch_{epoch+1}_batch_{i+1}_predicted_mask_{j+1}.png",
                        nrow=num_images,
                        normalize=True,
                        scale_each=True
                    )



# Save the trained model
torch.save(model.state_dict(), 'unet_model.pth')
