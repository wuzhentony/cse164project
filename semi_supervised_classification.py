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
import torch.nn.functional as F
from supervised_convnext.py import LabeledImageDataset, ConvNeXtClassifier


def train(model, train_loader, optimizer, scaler, device, ema):
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
            if labels.ndim == 2:
                loss = F.cross_entropy(outputs, labels)
            else:
                loss = F.cross_entropy(outputs,labels,label_smoothing=0.1)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)
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
    epochs = 50
    checkpoint_path = "models/supervised"
    encoder_name = "encoder_test"
    model_name = "model_test"
    encoder_weights = "models/ssl/encoder_40.pt"
    prev_weights = None #"models/supervised/encoder.pt"
    drop_path_rate = 0.2
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # make datasets with randaugment
    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    train_path = data_path 
    train_json = data_path / 'metadata/train_labeled.json'
    val_path = data_path / 'val/images'
    val_json = data_path / 'val/classification.json'

    train_dataset = LabeledImageDataset(
        data_path=train_path,
        json_file=train_json,
        transform = v2.Compose([
            v2.ToImage(),
            v2.Resize((224,224)),
            # v2.RandomResizedCrop(224, scale=(0.6, 1.0)),
            # v2.RandomHorizontalFlip(),
            v2.RandAugment(num_ops=2, magnitude=7),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])
    )


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

    # make dataloaders with cutmix 
    # mixup = v2.MixUp(num_classes=300)
    # cutmix = v2.CutMix(num_classes=300)
    # mixupcutmix = v2.RandomChoice([mixup, cutmix])
    # def collate_fn(batch):
    #     images, labels = torch.utils.data.default_collate(batch)
    #     return mixupcutmix(images, labels)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        #collate_fn=collate_fn,
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

    model = ConvNeXtClassifier(encoder_weights=encoder_weights).to(device)
    if prev_weights:
        checkpoint = torch.load(prev_weights,  map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        checkpoint = torch.load(encoder_weights,  map_location="cpu", weights_only=True)
        model.encoder.load_state_dict(checkpoint)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4,weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = ModelEmaV3(model, decay=0.9999)
    scaler = torch.amp.GradScaler(device)

    best_val_acc = 0.0
    for epoch in range(epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, scaler, device, ema)
        val_loss, val_acc = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        # Save checkpoint
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, f"{checkpoint_path}/{model_name}.pt")

            torch.save(model.encoder.state_dict(), f"{checkpoint_path}/{encoder_name}.pt")
            print(f"Saved checkpoint")