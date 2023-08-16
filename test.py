from unet import Unet
import torch
from dataloader import IDRiDDataset 
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from glob import glob
import os
from utils import create_dir

batch_size = 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the path to the saved model
model_path = 'unet_model.pth'

# Instantiate the U-Net model
model = Unet()
model = model.double()  # Make sure to set the same data type as during testing

# Load the tested weights
model.load_state_dict(torch.load(model_path))
model.to(device)  # Move model to GPU if available
model.eval()  # Set the model to evaluation mode



create_dir("visualization_test")

# Load new data for inference
# Dataset instance for testing
path = './Segmentation.nosync'
test_image_paths = sorted(glob(os.path.join(path, "test_images", "*.jpg")))
test_mask_root_dirs = [os.path.join(path, "test_masks", "EX"), os.path.join(path, "test_masks", "HE"), os.path.join(path, "test_masks", "MA"), os.path.join(path, "test_masks", "SE")]
test_mask_extensions = ["tif", "tif", "tif", "tif"]
test_dataset = IDRiDDataset(test_image_paths, test_mask_root_dirs, test_mask_extensions, transform=None)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Perform inference
with torch.no_grad():
    for i, (batch_images, batch_masks) in enumerate(test_loader):
        batch_images = batch_images.to(device).double()
        batch_masks = torch.stack(batch_masks, dim=1).to(device).double()

        outputs = model(batch_images)

        num_images = batch_images.size(0)

        for j in range(4):
            # Save original batch images
            vutils.save_image(
                batch_images,
                f"visualization_test/batch_{i+1}_original.png",
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
                f"visualization_test/batch_{i+1}_predicted_mask_{j+1}.png",
                nrow=num_images,
                normalize=True,
                scale_each=True
            )

