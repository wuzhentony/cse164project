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
import timm
from convnextv2_util import Block
import argparse
import matplotlib.pyplot as plt

class ImageFolderDataset(Dataset):
    """Dataset that loads all images from a folder"""
    VALID_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}
    
    def __init__(self, folder_path, transform=None):
        """
        Args:
            folder_path: Path to folder containing images
            transform: Optional image transformations
        """
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [
            f for f in os.listdir(folder_path) 
            if os.path.splitext(f)[1].lower() in self.VALID_EXTENSIONS
        ]
        
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image


class ConvNeXtSSL(nn.Module):
    def __init__(self, mask_ratio=0.75, patch_size=16, decoder_dim=256, decoder_depth=1, drop_path_rate=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.encoder = timm.create_model(
            "convnextv2_tiny.fcmae",
            pretrained=False,features_only=True,
            drop_path_rate=drop_path_rate
        )
        encoder_dim = self.encoder.feature_info.channels()[-2] + self.encoder.feature_info.channels()[-1] 
        self.proj = nn.Conv2d(encoder_dim,decoder_dim,kernel_size=1)
        self.decoder = nn.Sequential(
            Block(decoder_dim) for i in range(decoder_depth)
        )        
        self.pred = nn.Conv2d(decoder_dim, patch_size*patch_size*3, kernel_size=1)


    def random_mask(self, images):
        B = images.shape[0]
        H = images.shape[2] // self.patch_size
        W = images.shape[3] // self.patch_size
        L = H * W
        len_keep = int(L * (1 - self.mask_ratio))
        noise = torch.rand(B, L, device=images.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        patch_mask = torch.ones(B, L, device=images.device)
        patch_mask[:, :len_keep] = 0
        patch_mask = torch.gather(patch_mask, dim=1, index=ids_restore)
        #change back to original image mask
        image_mask = (patch_mask.reshape(B, H, W).repeat_interleave(self.patch_size, dim=1).repeat_interleave(self.patch_size, dim=2))
        masked_images = images * (1 - image_mask.unsqueeze(1))

        return masked_images, patch_mask

    def forward(self, images):
        #encoder portion
        x, patch_mask = self.random_mask(images)
        _, _, f3, f4 = self.encoder(x)
        f4 = torch.nn.functional.interpolate(f4, size=f3.shape[-2:], mode="bilinear")
        x = torch.cat([f3, f4], dim=1)
        #decoder portion
        x = self.proj(x)
        x = self.decoder(x)
        x = self.pred(x)
        return x, patch_mask

def patchify(imgs, p=16):
    h = w = imgs.shape[2] // p
    x = imgs.reshape(imgs.shape[0],3,h,p,w,p)
    x = torch.einsum("nchpwq->nhwpqc",x)
    x = x.reshape(imgs.shape[0], h*w, p*p*3)
    return x

# def reconstruction_loss(prediction, target, mask):
#     mask = mask.unsqueeze(1)
#     loss = (prediction - target).pow(2)
#     loss = loss * mask
#     return loss.sum() / ( mask.sum() * prediction.shape[1])

def reconstruction_loss(pred, images, patch_mask, patch_size=16):
    B, C, H, W = pred.shape
    pred = pred.reshape(B, C, -1)
    pred = pred.permute(0, 2, 1)
    target = patchify(images, patch_size)
    mean = target.mean(dim=-1, keepdim=True)
    var = target.var(dim=-1, keepdim=True)
    target = (target - mean) / torch.sqrt(var + 1e-6)
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)
    loss = (loss*patch_mask).sum() / patch_mask.sum()
    return loss

def train(model, patch_size, train_loader, optimizer, scaler, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc='SSL Training')
    for images in pbar:
        images = images.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"):
            reconstruction, mask = model(images)
            loss = reconstruction_loss(reconstruction,images,mask,patch_size)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

if __name__ == "__main__":
    batch_size = 128
    epochs = 30
    checkpoint_path = "models/ssl"
    encoder_name = "encoder"
    model_name = "model"
    mask_ratio = 0.75
    patch_size = 16
    drop_path_rate = 0.1
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    train_labeled_path = data_path / 'train_labeled/images'
    train_unlabeled_path = data_path / 'train_unlabeled/images'
    val_path = data_path / 'val/images'

    # Transforms
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5,1.0)),
        transforms.RandomHorizontalFlip(0.5),
        #transforms.RandomVerticalFlip(0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    # Create datasets
    datasets = []
    datasets.append(ImageFolderDataset(folder_path=train_labeled_path,transform=transform))
    datasets.append(ImageFolderDataset(folder_path=train_unlabeled_path,transform=transform))
    datasets.append(ImageFolderDataset(folder_path=val_path,transform=transform))
    combined_dataset = ConcatDataset(datasets)
    
    #test_dataset = ImageFolderDataset(folder_path=test_path,transform=test_transform)

    total_samples = len(combined_dataset)
    print(f"Total samples: {total_samples}")
    
    # Create dataloader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    print(f"Training ConvNeXt encoder")

    model = ConvNeXtSSL(mask_ratio=mask_ratio, patch_size=patch_size, drop_path_rate=drop_path_rate).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda")

    losses = []
    for epoch in range(epochs):
        avg_loss = train(model, patch_size, train_loader, optimizer, scaler, device)
        losses.append(avg_loss)
        #val_loss = validate(model, test_loader, device)
        scheduler.step()
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.4f} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        # Save checkpoint
        torch.save({
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        }, f"{checkpoint_path}/{model_name}.pt")

        torch.save(model.encoder.state_dict(), f"{checkpoint_path}/{encoder_name}_{epoch+1}.pt")
        print(f"Saved checkpoint")

    epochs = list(range(1, len(losses) + 1))
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, losses, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss")
    plt.title("SSL Pretraining Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/ssl_loss_curve.png")
    plt.show()
