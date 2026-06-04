import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import timm
from supervised_convnext import *

def train(model, train_loader, optimizer, scaler, device):
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
        total_loss += loss.item()
        predicted = outputs.argmax(dim=1)
        target = labels
        #target = labels.argmax(dim=1)
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
    epochs = 10
    encoder_folder = "models/ssl/tst"

    # make datasets
    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    train_path = data_path 
    train_json = data_path / 'metadata/train_labeled.json'
    val_path = data_path / 'val/images'
    val_json = data_path / 'val/classification.json'

    train_dataset = LabeledImageDataset(
        data_path=train_path,
        json_file=train_json,
        transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.RandAugment(num_ops=2, magnitude=7),
            v2.ToImage(),
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
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create models
    print(f"Test multiple models with different ssl_encoders")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    best_val_acc = 0.0
    best_encoder = None
    encoder_vals = {}
    for f in os.listdir(encoder_folder): 
        print(f"Evaluating {f}")
        encoder_file_path = f"{encoder_folder}/{f}"
        epoch_best_val = 0.0
        model = ConvNeXtClassifier(encoder_weights=encoder_file_path, freeze_encoder=True).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=0.05)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        scaler = torch.amp.GradScaler(device)
        for epoch in range(epochs):
            train_loss, train_acc = train(model, train_loader, optimizer, scaler, device)
            val_loss, val_acc = validate(model, val_loader, device)
            scheduler.step()
            print(f"Epoch {epoch+1}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if val_acc > epoch_best_val:
                epoch_best_val = val_acc
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_encoder = f
        encoder_vals[f] = epoch_best_val

    print(f"Best encoder: {best_encoder}, val_acc: {best_val_acc}")
    print(encoder_vals)