from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import numpy as np

class IDRiDDataset(Dataset):
    def __init__(self, image_paths, mask_root_dirs, mask_extensions, transform=None):
        self.image_paths = image_paths
        self.mask_root_dirs = mask_root_dirs
        self.mask_extensions = mask_extensions
        self.transform = transform
        # self.augment = augment

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image = Image.open(self.image_paths[index])
        image_num = self.image_paths[index].split('/')[-1].split('.')[0].split('_')[-1]
        image = image.resize((512, 512))  # Resize the image
        image = np.array(image)

        # if self.transform:
        #     image = self.transform(image)  # Convert image to tensor

        # if self.augment:
        #     image = self.augment(image)  # Augment image
        
        masks_per_image = []
        for mask_dir, mask_type, mask_ext in zip(self.mask_root_dirs, ['EX', 'HE', 'SE', 'MA'], self.mask_extensions):
            mask_path = os.path.join(mask_dir, f"IDRiD_{image_num}_{mask_dir.split('/')[-1]}.{mask_ext}")  # Note the +1 to match your dataset indices
            mask = Image.open(mask_path).resize((512, 512))
            # mask = self.transform(mask.convert('L'))  # Convert mask to tensor
            if mask.mode == 'RGB':
                mask = mask.convert('L')
            mask = np.array(mask)
            masks_per_image.append(mask)
        
        sample = {'image': image, 'masks': masks_per_image}

        if self.transform:
            sample = self.transform(sample)  # Convert image to tensor
        
        return sample['image'], sample['masks']
