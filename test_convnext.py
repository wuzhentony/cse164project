import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from ssl_convnext import *
from torchvision.transforms import v2
from torchvision.utils import save_image
from torchvision.utils import save_image



def unpatchify(x, p):
    B, C, H, W = x.shape
    x = x.reshape(B, C, H * W)
    x = x.permute(0, 2, 1)
    h = w = int(x.shape[1]**.5)    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    return imgs

if __name__ == "__main__":
    batch_size = 8
    mask_ratio = 0.5
    checkpoint_path = "models/ssl/model.pt"
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    val_path = data_path / 'val/images'

    # Transforms
    transform=v2.Compose([
        v2.Resize((224,224)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Create datasets
    dataset = ImageFolderDataset(folder_path=val_path,transform=transform)
    total_samples = len(dataset)
    print(f"Total samples: {total_samples}")
    
    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    model = ConvNeXtSSL(mask_ratio=mask_ratio).to(device)
    checkpoint = torch.load(checkpoint_path,  map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model'])

    model.eval()
    for images in loader:
        images = images.to(device)
        # Forward pass
        res, mask = model(images)
        res = unpatchify(res, 16)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

        #randomly reconstruct 5 images
        for i in range(5):
            tensor = images[i] * std + mean
            tensor = tensor.clamp(0,1)
            save_image(tensor, f"test_generation/original_{i}.png")
            tensor = res[i] * std + mean
            tensor = tensor.clamp(0,1)
            save_image(tensor, f"test_generation/reconstructed_{i}.png")
        break

    