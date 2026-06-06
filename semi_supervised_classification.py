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
from supervised_convnext import LabeledImageDataset, ConvNeXtClassifier

class UnlabeledDataset(Dataset):
    VALID_EXTENSIONS = {'.jpg', '.jpeg'}

    def __init__(self, data_path, weak_transform, strong_transform):
        self.data_path = Path(data_path)

        self.image_files = [
            p for p in self.data_path.iterdir()
            if p.suffix.lower() in self.VALID_EXTENSIONS
        ]

        self.weak_transform = weak_transform
        self.strong_transform = strong_transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image = Image.open(self.image_files[idx]).convert("RGB")
        weak = self.weak_transform(image)
        strong = self.strong_transform(image)
        return weak, strong



def fixmatch_loss(model, labeled_images, labels, unlabeled_weak, unlabeled_strong, threshold=0.95, lambda_u=1.0):
    logits_x = model(labeled_images)
    loss_sup = F.cross_entropy(logits_x, labels)
    with torch.no_grad():
        logits_u_w = model(unlabeled_weak)
        probs = torch.softmax(logits_u_w, dim=1)
        max_probs, pseudo_labels = probs.max(dim=1)
        mask = max_probs.ge(threshold).float()

    logits_u_s = model(unlabeled_strong)
    loss_unsup = F.cross_entropy(logits_u_s,pseudo_labels,reduction='none')
    loss_unsup = (loss_unsup * mask).sum() / mask.sum().clamp(min=1)
    total_loss = loss_sup + lambda_u * loss_unsup

    return total_loss, loss_sup, loss_unsup, mask.mean(), logits_x

def train(model,labeled_loader,unlabeled_loader,optimizer,scaler,device,ema,threshold=0.95,lambda_u=0.5):
    model.train()

    total_loss = 0
    total_sup_loss = 0
    total_unsup_loss = 0
    total_pseudo_rate = 0

    correct = 0
    total = 0

    pbar = tqdm(labeled_loader, total=len(labeled_loader),desc="Training")

    unlabeled_iter = iter(unlabeled_loader)
    for images, labels in pbar:
        try:
            u_w, u_s = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(unlabeled_loader)
            u_w, u_s = next(unlabeled_iter)

        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        u_w = u_w.to(device, non_blocking=True)
        u_s = u_s.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss, sup_loss, unsup_loss, pseudo_rate, logits_x = fixmatch_loss(
                model,images,labels,u_w,u_s,threshold=threshold,lambda_u=lambda_u
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        ema.update(model)

        total_loss += loss.item()
        total_sup_loss += sup_loss.item()
        total_unsup_loss += unsup_loss.item()
        total_pseudo_rate += pseudo_rate.item()

        predicted = logits_x.argmax(dim=1)

        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            sup=f"{sup_loss.item():.4f}",
            unsup=f"{unsup_loss.item():.4f}",
            pseudo=f"{pseudo_rate.item():.3f}",
            acc=f"{100*correct/total:.2f}%"
        )

    avg_loss = total_loss / len(labeled_loader)
    avg_accuracy = 100 * correct / total
    avg_pseudo_rate = total_pseudo_rate / len(labeled_loader)
    return avg_loss, avg_accuracy, avg_pseudo_rate

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
    batch_size = 16
    epochs = 50
    checkpoint_path = "models/semi_supervised"
    encoder_name = "encoder_fixmatch"
    model_name = "model_fixmatch"
    encoder_weights = None #"models/supervised/encoder_1.pt"
    prev_weights = "models/supervised/model_v3.pt"
    drop_path_rate = 0.2 #drop_path_rate for model
    threshold = .8 #threshold for considering unlabeled data
    lambda_u = 1 # weight for unlabeled loss
    mu = 7 #how many times greater the batch size is for unlabeled data
    
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # make datasets with randaugment
    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    train_path = data_path 
    train_json = data_path / 'metadata/train_labeled.json'
    unlabeled_path = data_path / 'train_unlabeled/images'
    val_path = data_path / 'val/images'
    val_json = data_path / 'val/classification.json'

    weak_transform = v2.Compose([
        v2.ToImage(),
        #v2.Resize((224,224)),
        v2.RandomResizedCrop(224, scale=(0.8,1.0)),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    strong_transform = v2.Compose([
        v2.ToImage(),
        #v2.Resize((224,224)),
        v2.RandomResizedCrop(224, scale=(0.8,1.0)),
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
        transform = weak_transform
    )

    unlabeled_dataset = UnlabeledDataset(
        data_path=unlabeled_path,
        weak_transform = weak_transform,
        strong_transform = strong_transform
    )

    val_dataset = LabeledImageDataset(
        data_path=val_path,
        json_file=val_json,
        transform=v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ]),
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

    unlabeled_loader = DataLoader(
        unlabeled_dataset,
        batch_size=batch_size*mu,
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

    # Create model
    print(f"Image classifcation with ConvNeXt backbone")

    model = ConvNeXtClassifier(encoder_weights=encoder_weights).to(device)
    if prev_weights:
        checkpoint = torch.load(prev_weights,  map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    

    optimizer = optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4,weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    ema = ModelEmaV3(model, decay=0.9999)
    scaler = torch.amp.GradScaler(device)


    best_val_acc = 0.0
    for epoch in range(epochs):

        train_loss, train_acc, pseudo_rate = train(model, train_loader, unlabeled_loader, optimizer, scaler, device, ema, threshold, lambda_u)
        val_loss, val_acc = validate(ema.module, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%" 
        print(f"Pseudo Rate: {pseudo_rate:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        # Save checkpoint
        if best_val_acc < val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model": ema.module.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, f"{checkpoint_path}/{model_name}.pt")

            torch.save(ema.module.encoder.state_dict(), f"{checkpoint_path}/{encoder_name}.pt")
            print(f"Saved checkpoint")