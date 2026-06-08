import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import  torch.nn.functional as F
from torchvision import transforms, models
from torchvision.transforms import v2
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from segmentation_models_pytorch.losses import DiceLoss
import torchvision.transforms.functional as TF
from torchvision import tv_tensors
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
import timm
from timm.utils import ModelEmaV3

import torch
import numpy as np
from PIL import Image

class SemanticImageDataset(Dataset):
    def __init__(self, json_file, data_path, transform=None):
        with open(json_file, 'r') as f:
            self.samples = json.load(f)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # image
        image_path = os.path.join(self.data_path, sample['image'])
        image = Image.open(image_path).convert('RGB')
        # mask
        mask_path = os.path.join(self.data_path, sample['mask'])
        mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.int32)

        mask = mask[:, :, 0] + 256 * mask[:, :, 1]
        mask[mask == 1000] = 0
        mask = (mask != 0).astype(np.int64)   # binary segmentation
        mask = tv_tensors.Mask(mask)

        if self.transform:
            image, mask = self.transform(image, mask)

        mask = mask.to(torch.long)

        return image, mask

class SemanticEvalDataset(Dataset):
    def __init__(self, json_file, data_path, transform=None):
        with open(json_file, 'r') as f:
            self.samples = json.load(f)
        self.data_path = data_path
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # image
        image_path = os.path.join(self.data_path, sample['image'])
        image = Image.open(image_path).convert('RGB')
        # mask
        mask_path = os.path.join(self.data_path, sample['mask'])
        mask = np.asarray(Image.open(mask_path).convert("RGB"), dtype=np.int32)

        mask = mask[:, :, 0] + 256 * mask[:, :, 1]
        mask[mask == 1000] = 0
        mask = (mask != 0).astype(np.int64)   # binary segmentation
        mask = torch.from_numpy(mask).long()

        if self.transform:
            image = self.transform(image)

        return image, mask, mask.shape[-2:]


class ConvNeXtV2UPerNet(nn.Module):
    def __init__(self, num_classes, encoder_weights=None):
        super().__init__()

        self.encoder = timm.create_model(
            "convnextv2_tiny.fcmae", 
            pretrained=False, 
            features_only=True, 
            out_indices=(0,1,2,3)
        )

        if encoder_weights:
            checkpoint = torch.load(encoder_weights, map_location="cpu", weights_only=True)
            self.encoder.load_state_dict(checkpoint)

        encoder_channels = self.encoder.feature_info.channels()
        self.decoder = UPerNetDecoder(
            encoder_channels=encoder_channels,
            encoder_depth=len(encoder_channels),
            decoder_channels=256,
            use_norm="batchnorm"
        )

        self.segmentation_head = SegmentationHead(
            in_channels=256,
            out_channels=num_classes,
            kernel_size=1,
            upsampling=16
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.segmentation_head(x)
        return x


def compute_iou(pred, target, num_classes=2, eps=1e-7):
    B = pred.shape[0]
    image_ious = []
    for b in range(B):
        class_ious = []
        for cls in range(num_classes):
            pred_i = pred[b] == cls
            target_i = target[b] == cls
            intersection = (pred_i & target_i).sum().float()
            union = (pred_i | target_i).sum().float()
            if union == 0:
                class_ious.append(torch.tensor(1.0, device=pred.device))
            else:
                class_ious.append(intersection / (union + eps))
        image_ious.append(torch.stack(class_ious).mean())
    return torch.stack(image_ious).mean().item()

# def compute_iou(pred, target, num_classes=2):
    # ious = []
    # for cls in range(num_classes):
    #     pred_i = (pred == cls)
    #     target_i = (target == cls)
    #     intersection = (pred_i & target_i).sum().item()
    #     union = (pred_i | target_i).sum().item()
    #     ious.append(intersection / (union + 1e-7))
    # return sum(ious) / len(ious) if ious else 0.0

def get_boundary(mask, width=3):
    x = mask.float().unsqueeze(1)
    eroded = -F.max_pool2d(-x,kernel_size=width,stride=1,padding=1)
    boundary = (x != eroded)
    return boundary.squeeze(1)

def boundary_weight(masks, width=3, alpha=4.0):
    """
    labels: [B,H,W] integer class labels

    returns:
        boundary: [B,H,W] float tensor (0 or 1)
    """
    boundary = get_boundary(masks, width=width)
    weight_map = (1.0 + alpha * boundary) #weight the boundaries by a factor of alpha
    return weight_map

def boundary_f_score(pred,target,boundary_width=3,tolerance=3,eps=1e-7):
    """
    pred:   [B,H,W] predicted labels
    target: [B,H,W] ground truth labels

    returns:
        mean boundary F-score
    """

    pred_boundary = get_boundary(pred == 1, width=boundary_width).bool()

    gt_boundary = get_boundary(target == 1,width=boundary_width).bool()

    pred_boundary_f = pred_boundary.float().unsqueeze(1)
    gt_boundary_f = gt_boundary.float().unsqueeze(1)

    pred_dilated = F.max_pool2d(pred_boundary_f, kernel_size=2 * tolerance + 1, stride=1, padding=tolerance).bool()

    gt_dilated = F.max_pool2d(gt_boundary_f, kernel_size=2 * tolerance + 1, stride=1, padding=tolerance).bool()

    precision_match = pred_boundary & gt_dilated.squeeze(1)
    recall_match = gt_boundary & pred_dilated.squeeze(1)

    precision = (precision_match.sum(dim=(1, 2)).float() / (pred_boundary.sum(dim=(1, 2)).float() + eps))
    recall = (recall_match.sum(dim=(1, 2)).float() / (gt_boundary.sum(dim=(1, 2)).float() + eps))

    fscore = (2 * precision * recall / (precision + recall + eps))

    empty = ((pred_boundary.sum(dim=(1,2)) == 0) & (gt_boundary.sum(dim=(1,2)) == 0))

    fscore[empty] = 1.0
    return fscore.mean().item()


def train(model, train_loader, optimizer, scaler, ema, alpha, device, classes=2):
    model.train()

    total_loss = 0.0
    total_iou = 0.0
    total_f = 0.0
    count = 0
    
    dice_loss_func = DiceLoss(mode="binary", from_logits=True)
    pbar = tqdm(train_loader, desc="Training")

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            logits = model(images)
            boundary_weights = boundary_weight(masks, alpha=alpha)
            boundary_weights = boundary_weights / boundary_weights.mean()
            ce_loss = (F.cross_entropy(logits, masks, reduction="none") * boundary_weights).mean()
            foreground_logits = logits[:, 1:2]
            dice_loss = dice_loss_func(foreground_logits, masks)
            loss = ce_loss + dice_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        preds = logits.argmax(dim=1)

        iou = compute_iou(preds, masks, num_classes=classes)
        boundary_f = boundary_f_score(preds, masks)

        total_loss += loss.item()
        total_iou += iou
        total_f += boundary_f
        count += 1

        pbar.set_postfix(
            loss=f"{loss.item():.4f}",
            miou=f"{iou:.4f}",
            boundary_f=f"{boundary_f:.4f}"
        )

    return total_loss / count, total_iou / count, total_f / count

def validate(model, val_loader, device):
    model.eval()

    total_loss = 0.0
    total_iou = 0.0
    total_f = 0.0
    count = 0

    dice_loss_func = DiceLoss(mode="binary", from_logits=True)
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validating")

        for images, masks in pbar:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                # logits = F.interpolate(
                #     logits,
                #     size=shape,
                #     mode="bilinear",
                #     align_corners=False
                # )
                boundary_weights = boundary_weight(masks, alpha=alpha)
                boundary_weights = boundary_weights / boundary_weights.mean()
                ce_loss = (F.cross_entropy(logits, masks, reduction="none") * boundary_weights).mean()
                foreground_logits = logits[:, 1:2]
                dice_loss = dice_loss_func(foreground_logits, masks)
                loss = ce_loss + dice_loss

            preds = logits.argmax(dim=1)
            iou = compute_iou(preds, masks, num_classes=2)
            boundary_f = boundary_f_score(preds, masks)

            total_loss += loss.item()
            total_iou += iou
            total_f += boundary_f
            count += 1

            pbar.set_postfix(
                loss=f"{loss.item():.4f}",
                miou=f"{iou:.4f}",
                boundary_f=f"{boundary_f:.4f}"
            )

    return total_loss / count, total_iou / count, total_f / count

if __name__ == "__main__":
    batch_size = 64
    epochs = 50
    unfreeze_epoch = 10
    checkpoint_path = "models/segmentation"
    model_name = "model_delete"
    encoder_weights = None # "models/supervised/encoder_v3.pt"
    prev_weights = "models/segmentation/model_1.pt"
    alpha = 4.0 #weight for boundary pixels
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # make datasets
    data_path = Path('/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data')
    train_path = data_path 
    train_json = data_path / 'metadata/train_seg.json'
    val_path = data_path / 'val'
    val_json = 'val_segment.json'

    train_transform = v2.Compose([
        v2.ToImage(),
        #v2.Resize((224,224)),
        v2.RandomResizedCrop(224, scale=(0.8,1.0)),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_dataset = SemanticImageDataset(
        data_path=train_path,
        json_file=train_json,
        transform = train_transform
    )


    val_dataset = SemanticImageDataset(
        data_path=val_path,
        json_file=val_json,
        transform=v2.Compose([
            v2.Resize((224,224)),
            #v2.CenterCrop(224),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    )
    
    # val_dataset = SemanticEvalDataset(
    #     data_path=val_path,
    #     json_file=val_json,
    #     transform=v2.Compose([
    #         v2.Resize((224,224)),
    #         #v2.CenterCrop(224)
    #         v2.ToImage(),
    #         v2.ToDtype(torch.float32, scale=True),
    #         v2.Normalize(
    #             mean=[0.485, 0.456, 0.406],
    #             std=[0.229, 0.224, 0.225]
    #         )
    #     ])
    # )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


    val_loader = DataLoader(
        val_dataset,
        batch_size=32, #due to varying sizes of masks
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create model
    print(f"Image classifcation with ConvNeXt backbone")

    model = ConvNeXtV2UPerNet(num_classes=2, encoder_weights=encoder_weights).to(device)
    if prev_weights:
        checkpoint = torch.load(prev_weights,  map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,weight_decay=0.05)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler(device)
    ema = ModelEmaV3(model, decay=0.9999)

    best_val_loss = float('inf')
    epoch = 0

    #freeze encoder intially to allow for decoder/head to learn
    for param in model.encoder.parameters():
        param.requires_grad = False
    for epoch in range(epochs):
        if (epoch+1) == unfreeze_epoch:
            print("Unfreezing encoder")
            for param in model.encoder.parameters():
                param.requires_grad = True
            optimizer = torch.optim.AdamW(
                [   
                    {"params": model.encoder.parameters(), "lr": 1e-4,}, 
                    {"params": model.decoder.parameters(), "lr": 3e-4,},
                    {"params": model.segmentation_head.parameters(), "lr": 3e-4,}
                ],
                weight_decay=0.05
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs-epoch)
        

        train_loss, train_iou, train_f = 0,0,0 #train(model, train_loader, optimizer, scaler, ema, alpha, device)
        val_loss, val_iou, val_f = validate(model, val_loader, device)
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{epochs} - LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train IOU: {train_iou:.2f}, Train Boundary F: {train_f:.2f}")
        print(f"Val Loss: {val_loss:.4f}, Val IOU: {val_iou:.2f}, Val Boundary F: {val_f:.2f}")
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }, f"{checkpoint_path}/{model_name}.pt")

            print(f"Saved checkpoint")
    




