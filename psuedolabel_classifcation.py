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
from supervised_convnext import *


class PseudoLabeledDataset(Dataset):
    def __init__(self, data_path):
        self.image_paths = [
            p for p in Path(data_path).glob("*")
        ]
        self.transform = v2.Compose([
            v2.Resize((224,224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        return image, str(path)

if __name__ == "__main__":
    batch_size = 64
    epochs = 30
    pseudo_json_name = "confidence_bins/lt_0.8.json"
    checkpoint_path = "models/semi_supervised"
    encoder_name = "pseudo_encoder_v5"
    model_name = "pseudo_model_v5"
    encoder_weights = None # "models/ssl/encoder_40.pt"
    prev_weights = "models/semi_supervised/pseudo_model_v4.pt"
    unfreeze_epoch = 1 
    drop_path_rate = 0.2
    threshold = 0.8
    create_pseudo_labels = False
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # load the model
    model = ConvNeXtClassifier(encoder_weights=encoder_weights, drop_path_rate=drop_path_rate).to(device)
    if prev_weights:
        checkpoint = torch.load(prev_weights,  map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])

    # create pseudo labels
    if create_pseudo_labels:
        data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
        unlabeled_path = data_path / "train_unlabeled/images"
        
        results = []

        unlabeled_dataset = PseudoLabeledDataset(data_path=unlabeled_path)
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
        )
        

        model.eval()

        results = []

        with torch.no_grad():
            pbar = tqdm(unlabeled_loader, desc="Psuedo Labeling")
            for images, paths in pbar:

                images = images.to(device)

                logits = model(images)
                probs = torch.softmax(logits, dim=1)

                confidences, predictions = probs.max(dim=1)

                for path, pred, conf in zip(paths, predictions.cpu(), confidences.cpu()):
                   #if conf.item() >= threshold:
                    results.append({
                        "class_id": int(pred.item()),
                        "image": path,
                        "confidence": float(conf.item())
                    })
        print(len(results))
        with open(pseudo_json_name, "w") as f:
            json.dump(results, f, indent=2)

    
    # make datasets with randaugment
    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    train_path = data_path 
    train_json = data_path / 'metadata/train_labeled.json'
    segment_path = data_path 
    segment_json = data_path / 'metadata/train_seg.json'
    pseudo_path = "/"
    pseudo_json = pseudo_json_name
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

    segment_dataset = LabeledImageDataset(
        data_path=segment_path,
        json_file=segment_json,
        transform = train_transform
    )

    pseudo_dataset = LabeledImageDataset(
        data_path=pseudo_path,
        json_file=pseudo_json,
        transform = train_transform
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

    train_dataset = ConcatDataset([train_dataset, segment_dataset, pseudo_dataset])

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
        
        train_loss, train_acc = 0,0# train(model, train_loader, optimizer, scaler, ema, device)
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
            print("Saved Checkpoint")