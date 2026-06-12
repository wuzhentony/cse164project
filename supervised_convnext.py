import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from torchvision.transforms import v2
from torchvision.models import convnext_tiny
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import timm
from timm.utils import ModelEmaV3

class LabeledImageDataset(Dataset):
    def __init__(self, json_file, data_path, transform=None):
        with open(json_file, 'r') as f:
            self.samples = json.load(f)
        self.data_path = data_path
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = os.path.join(self.data_path, sample['image'])
        label = sample['class_id']
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        return image, label

class ConvNeXtClassifier(nn.Module):
    def __init__(self, num_classes=300, encoder_weights=None, drop_path_rate=0):
        super().__init__()

        self.encoder = timm.create_model("convnextv2_tiny.fcmae",pretrained=False,features_only=True)
        if encoder_weights:
            checkpoint = torch.load(encoder_weights, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(checkpoint)
            
        self.head = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, num_classes)
        )

    def forward(self, images):
        x = self.encoder(images)[-1]
        x = x.mean(dim=(-2, -1)) # global average pooling
        x = self.head(x)
        return x

def train(model, train_loader, optimizer, scaler, ema, device):
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc="Training")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            outputs = model(images)
            loss = torch.nn.functional.cross_entropy(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #ema.update(model)
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        if labels.ndim == 2:  # CutMix labels
            target = labels.argmax(dim=1)
        else:
            target = labels
        correct += (predicted == target).sum().item()
        total += images.size(0)
        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            acc=f"{100.0 * correct / total:.2f}%"
        )
    avg_loss = total_loss / len(train_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

def validate(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")

        for images, labels in pbar:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda",dtype=torch.float16):
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs,labels)

            total_loss += loss.item()

            predicted = outputs.argmax(dim=1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                acc=f"{100.0 * correct / total:.2f}%"
            )

    avg_loss = total_loss / len(val_loader)
    accuracy = 100.0 * correct / total

    return avg_loss, accuracy

if __name__ == "__main__":
    batch_size = 64
    epochs = 20
    checkpoint_path = "models/supervised"
    encoder_name = "encoder_v4"
    model_name = "model_v4"
    encoder_weights = None # "models/ssl/encoder_40.pt"
    prev_weights = "models/supervised/model_v3.pt"
    unfreeze_epoch = 1 
    drop_path_rate = 0.2
    
    # Ensure output directory exists
    #os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make datasets with randaugment
    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    train_path = data_path 
    train_json = data_path / 'metadata/train_labeled.json'
    segment_path = data_path 
    segment_json = data_path / 'metadata/train_seg.json'
    val_path = data_path / 'val/images'
    val_json = data_path / 'val/classification.json'
    

    train_transform = v2.Compose([
        #v2.Resize((224, 224)),
        v2.RandomResizedCrop(224, scale=(0.8,1.0)),
        v2.ToImage(),
        v2.RandomHorizontalFlip(),
        v2.RandAugment(num_ops=2, magnitude=7),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = LabeledImageDataset(
        data_path=train_path,
        json_file=train_json,
        transform = train_transform
    )

    # segment_dataset = LabeledImageDataset(
    #     data_path=segment_path,
    #     json_file=segment_json,
    #     transform = train_transform
    # )

    val_dataset = LabeledImageDataset(
        data_path=val_path,
        json_file=val_json,
        transform=v2.Compose([
            v2.Resize((224,224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )

    # train_dataset = ConcatDataset([train_dataset, segment_dataset])

    # make dataloaders with cutmix 
    cutmix = v2.CutMix(num_classes=300)
    def collate_fn(batch):
        images, labels = torch.utils.data.default_collate(batch)
        return cutmix(images, labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    print(f"Image classifcation with ConvNeXt backbone")

    freeze_encoder = unfreeze_epoch > 1

    model = ConvNeXtClassifier(encoder_weights=encoder_weights, drop_path_rate=drop_path_rate).to(device)
    if prev_weights:
        checkpoint = torch.load(prev_weights,  map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device)
    ema = ModelEmaV3(model, decay=0.9999)

    for param in model.encoder.parameters():
        param.requires_grad = False
    best_val_loss = float('inf')
    for epoch in range(epochs):
        if (epoch+1) == unfreeze_epoch:
            print("Unfreezing encoder")
            for param in model.encoder.parameters():
                param.requires_grad = True

            optimizer = torch.optim.AdamW(
                [{"params": model.encoder.parameters(), "lr": 1e-4,}, 
                {"params": model.head.parameters(), "lr": 3e-4,}],
                weight_decay=0.05
            )

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch)
        
        train_loss, train_acc = train(model, train_loader, optimizer, scaler, ema, device)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        # Save checkpoint
        if best_val_loss > val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, f"{checkpoint_path}/{model_name}.pt")

            torch.save(model.encoder.state_dict(), f"{checkpoint_path}/{encoder_name}.pt")
            print(f"Saved checkpoint")