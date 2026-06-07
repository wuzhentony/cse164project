import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import v2
from PIL import Image
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import timm
from segmentation_models_pytorch.decoders.upernet.decoder import UPerNetDecoder
from segmentation_models_pytorch.base import SegmentationHead
from supervised_convnext import ConvNeXtClassifier
from segmentation_convnext import ConvNeXtV2UPerNet
import pandas as pd


class CombinedInference:
    def __init__(self, seg_model_path, clf_model_path, device='cuda'):
        self.device = device
        self.window_size = 224
        self.overlap = 56

        # Load segmentation model
        print("Loading segmentation model...")
        self.seg_model = ConvNeXtV2UPerNet(num_classes=2, encoder_weights=None).to(device)
        seg_checkpoint = torch.load(seg_model_path, map_location="cpu", weights_only=False)
        self.seg_model.load_state_dict(seg_checkpoint["model"])
        self.seg_model.eval()
        
        # Load classifier model
        print("Loading classifier model...")
        self.clf_model = ConvNeXtClassifier(num_classes=300, encoder_weights=None).to(device)
        clf_checkpoint = torch.load(clf_model_path, map_location="cpu", weights_only=False)
        self.clf_model.load_state_dict(clf_checkpoint["model"])
        self.clf_model.eval()
        
        # Preprocessing
        self.transform = v2.Compose([
            v2.Resize((224, 224)),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    @torch.no_grad()
    def sliding_window_inference(self, image_path):
        """
        Returns:
            mask: HxW array
                0 = background
                predicted_class + 1 = foreground
        """
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        print(f"  Processing image: {orig_width}x{orig_height}")

        #classification
        full_image_tensor = (self.transform(image).unsqueeze(0).to(self.device))
        with torch.no_grad():
            clf_logits = self.clf_model(full_image_tensor)

        predicted_class = clf_logits.argmax(dim=1).item()
        class_value = predicted_class + 1

        print(f"  Predicted class: {predicted_class} -> mask value: {class_value}")

        # ------------------------------------------------------------------
        # Probability accumulation maps
        # ------------------------------------------------------------------
        foreground_prob_sum = np.zeros((orig_height, orig_width),dtype=np.float32)

        overlap_count = np.zeros((orig_height, orig_width),dtype=np.float32)
        step = self.window_size - self.overlap
        windows_processed = 0
        # Generate window positions
        y_positions = list(range(0, orig_height, step))
        x_positions = list(range(0, orig_width, step))
        # Force final window to touch image border
        if y_positions[-1] + self.window_size < orig_height:
            y_positions.append(orig_height - self.window_size)
        if x_positions[-1] + self.window_size < orig_width:
            x_positions.append(orig_width - self.window_size)

        # Remove duplicates
        y_positions = sorted(set(y_positions))
        x_positions = sorted(set(x_positions))

        with torch.no_grad():
            for y in y_positions:
                for x in x_positions:

                    y_start = min(y, max(0, orig_height - self.window_size))
                    x_start = min(x, max(0, orig_width - self.window_size))

                    y_end = min(y_start + self.window_size, orig_height)
                    x_end = min(x_start + self.window_size, orig_width)

                    window = image.crop((x_start, y_start, x_end, y_end))

                    # Pad border windows if necessary
                    if (window.size[0] < self.window_size or window.size[1] < self.window_size):
                        padded = Image.new("RGB",(self.window_size, self.window_size),(0, 0, 0))
                        padded.paste(window, (0, 0))
                        window = padded

                    window_tensor = (self.transform(window).unsqueeze(0).to(self.device))
                    
                    #extract segmentation result
                    seg_logits = self.seg_model(window_tensor)
                    seg_probs = torch.softmax(seg_logits,dim=1)

                    foreground_prob = (seg_probs[:, 1].squeeze(0).cpu().numpy())

                    actual_height = y_end - y_start
                    actual_width = x_end - x_start

                    foreground_prob = foreground_prob[:actual_height,:actual_width]
                    foreground_prob_sum[y_start:y_end,x_start:x_end] += foreground_prob
                    overlap_count[y_start:y_end,x_start:x_end] += 1

                    windows_processed += 1

        # ------------------------------------------------------------------
        # Average overlapping predictions
        # ------------------------------------------------------------------
        avg_foreground_prob = np.divide(
            foreground_prob_sum,
            overlap_count,
            out=np.zeros_like(foreground_prob_sum),
            where=overlap_count > 0
        )

        # Threshold
        foreground_mask = avg_foreground_prob > 0.5
        output_mask = np.zeros((orig_height, orig_width),dtype=np.int32)
        output_mask[foreground_mask] = class_value

        #print(f"  Processed {windows_processed} windows")

        return predicted_class, output_mask
    
    @torch.no_grad()
    def interpolate_inference(self, image_path):
        """
        Returns:
            mask: HxW array
                0 = background
                predicted_class + 1 = foreground
        """
        image = Image.open(image_path).convert("RGB")
        orig_width, orig_height = image.size
        print(f"  Processing image: {orig_width}x{orig_height}")

        #classification
        full_image_tensor = (self.transform(image).unsqueeze(0).to(self.device))
        clf_logits = self.clf_model(full_image_tensor)

        predicted_class = clf_logits.argmax(dim=1).item()
        class_value = predicted_class + 1

        print(f"  Predicted class: {predicted_class} -> mask value: {class_value}")
        logits = self.seg_model(full_image_tensor)
        logits_resized = F.interpolate(
            logits,
            size=((orig_height, orig_width)),
            mode="bilinear",
            align_corners=False
        )
        preds = logits_resized.argmax(dim=1) * class_value
        pred_mask = preds.squeeze(0).cpu().numpy()
        return predicted_class, pred_mask
    
    def save_mask_as_image(self, mask, output_path):
        """Save mask as PNG with color mapping"""
        # Create RGB image from mask
        # 0 = black (background), other values = color based on class
        rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        
        for class_val in np.unique(mask):
            if class_val == 0:
                continue
            class_idx = class_val - 1  # Convert back to 0-299
            # Simple color mapping based on class
            r = (class_idx * 67) % 256 + 100
            g = (class_idx * 131) % 256 + 100
            b = (class_idx * 179) % 256 + 100
            rgb_image[mask == class_val] = [r, g, b]
        
        img = Image.fromarray(rgb_image, 'RGB')
        img.save(output_path)
    

def encode_mask_ids(mask_ids: np.ndarray) -> str:
    """Encode a 2D id mask as row-major 1-indexed RLE triples.

    The encoded string is a space-separated sequence of:

        start length value start length value ...

    Only non-background pixels are stored. `start` is 1-indexed after row-major
    flattening, `length` is the run length, and `value` is the segmentation id.
    """
    flat = np.asarray(mask_ids, dtype=np.int64).reshape(-1)
    nonzero = flat != 0
    if not np.any(nonzero):
        return ""
    idx = np.flatnonzero(nonzero)
    values = flat[idx]

    run_break = np.ones(len(idx), dtype=bool)
    run_break[1:] = (idx[1:] != idx[:-1] + 1) | (values[1:] != values[:-1])
    starts = np.flatnonzero(run_break)
    ends = np.r_[starts[1:], len(idx)]

    parts: list[str] = []
    for start_pos, end_pos in zip(starts, ends):
        start = int(idx[start_pos]) + 1
        length = int(end_pos - start_pos)
        value = int(values[start_pos])
        parts.extend([str(start), str(length), str(value)])
    return " ".join(parts)


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Paths
    seg_model_path = "/home/tony/project/models/segmentation/model_1.pt"
    clf_model_path = "/home/tony/project/models/supervised/model_v3.pt"
    # test_images_dir = Path("/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data/test/images")
    test_images_dir = Path("/home/tony/.cache/kagglehub/competitions/cse-164-final-project-2026/data/val/images")
    output_dir = Path("/home/tony/project/test_predictions")
    submit_name = "val_submission.csv"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize inference engine
    inference = CombinedInference(seg_model_path, clf_model_path, device)
    
    # Get all test images
    test_images = sorted(list(test_images_dir.glob("*.JPEG")))
    print(f"Found {len(test_images)} test images")
    
    # Process each test image
    
    rows = []

    i = 0
    for image_path in tqdm(test_images, desc="Processing test images"):

        try:
            image_name = image_path.name
            # Run inference
            class_id, mask = inference.sliding_window_inference(str(image_path))
            # class_id, mask = inference.interpolate_inference(str(image_path))
            segmentation_rle = encode_mask_ids(mask)
            if len(segmentation_rle) < 1:
                segmentation_rle = f"1 1 {class_id+1}"
            rows.append({
                "image": image_name,
                "class_id": class_id,
                "segmentation_rle": segmentation_rle
            })
            if (200 < i < 205):
                inference.save_mask_as_image(mask, output_dir / image_name)
            i += 1
            
        except Exception as e:
            print(f"  Error processing {image_path}: {e}")
            break
            import traceback
            traceback.print_exc()
    
    submission = pd.DataFrame(rows)
    submission.to_csv(submit_name, index=False)


if __name__ == "__main__":
    main()
