# drive_dataloader_fixed.py
import os
from typing import List, Tuple, Optional
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

def find_image_and_label_folders(split_dir: str) -> Tuple[str, str]:
    subs = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
    image_sub = None
    label_sub = None
    for s in subs:
        sl = s.lower()
        if sl.startswith("image"):
            image_sub = s
        if sl.startswith("label"):
            label_sub = s
    if image_sub is None or label_sub is None:
        raise RuntimeError(f"Could not find image/label subfolders in {split_dir}. Found: {subs}")
    return os.path.join(split_dir, image_sub), os.path.join(split_dir, label_sub)


def build_file_list(root_dir: str, splits: Optional[List[str]] = None) -> dict:
    if splits is None:
        splits = ["train1", "train2", "train3", "val", "test"]
    file_list = {}
    for sp in splits:
        split_dir = os.path.join(root_dir, sp)
        if not os.path.isdir(split_dir):
            print(f"[build_file_list] skip missing split dir: {split_dir}")
            file_list[sp] = []
            continue
        img_dir, lbl_dir = find_image_and_label_folders(split_dir)
        imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        pairs = []
        for img_name in imgs:
            img_path = os.path.join(img_dir, img_name)
            base = os.path.splitext(img_name)[0]
            lbl_name = base + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if not os.path.isfile(lbl_path):
                lbl_path = None
            pairs.append((img_path, lbl_path))
        file_list[sp] = pairs
    return file_list



def parse_yolo_label_file(label_path: Optional[str], img_w: int, img_h: int, zero_based_class: bool = True):
    if label_path is None or not os.path.isfile(label_path):
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    boxes = []
    labels = []
    with open(label_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if line == "":
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            if not zero_based_class:
                cls = cls - 1
            cx = float(parts[1])
            cy = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])

            cx_px = cx * img_w
            cy_px = cy * img_h
            w_px = w * img_w
            h_px = h * img_h

            x1 = cx_px - w_px / 2.0
            y1 = cy_px - h_px / 2.0
            x2 = cx_px + w_px / 2.0
            y2 = cy_px + h_px / 2.0

            # clip
            x1 = max(0.0, min(x1, img_w - 1.0))
            y1 = max(0.0, min(y1, img_h - 1.0))
            x2 = max(0.0, min(x2, img_w - 1.0))
            y2 = max(0.0, min(y2, img_h - 1.0))

            # tiny box filter
            if (x2 - x1) <= 0.5 or (y2 - y1) <= 0.5:
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(cls)
    if len(boxes) == 0:
        return torch.zeros((0, 4), dtype=torch.float32), torch.zeros((0,), dtype=torch.int64)
    return torch.tensor(boxes, dtype=torch.float32), torch.tensor(labels, dtype=torch.int64)



def letterbox_resize(img: Image.Image, boxes: torch.Tensor, target_size: int = 640, color: int = 114):
    
    """ 
    if an image isn't square (say 1280 x 720) if we directly resize it to 640 x 640
    the image becomes streched and disort objects instead we use letterbox

    1. Resize it while keeping the aspect ratio 
    2. Pad(add borders) to make it squares 
    
    Target = 640x640
    Resized = 640x360
    Padding needed = 640 - 360 = 280 pixels vertically
    pad_top = 140, pad_bottom = 140
    """
    orig_w, orig_h = img.size
    if orig_w == 0 or orig_h == 0:
        raise ValueError("Image has zero dimension")
    scale = min(target_size / orig_w, target_size / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    img_resized = img.resize((new_w, new_h), resample=Image.BILINEAR)
    pad_w = target_size - new_w
    pad_h = target_size - new_h
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    new_im = Image.new("RGB", (target_size, target_size), (color, color, color))
    new_im.paste(img_resized, (pad_left, pad_top))
    if boxes is None or boxes.numel() == 0:
        return new_im, torch.zeros((0, 4), dtype=torch.float32), scale, (pad_left, pad_top)
    boxes = boxes.clone().float()
    boxes *= float(scale)
    boxes[:, [0, 2]] += pad_left
    boxes[:, [1, 3]] += pad_top
    return new_im, boxes, scale, (pad_left, pad_top)


def random_horizontal_flip(img: Image.Image, boxes: torch.Tensor, p: float = 0.5):
    if random.random() < p:
        img = TF.hflip(img)
        if boxes is not None and boxes.numel() > 0:
            w = img.width
            x1 = boxes[:, 0].clone()
            x2 = boxes[:, 2].clone()
            boxes[:, 0] = w - x2 - 1
            boxes[:, 2] = w - x1 - 1
    return img, boxes


class DriveIndiaDataset(Dataset):
    def __init__(self,
                 root_dir: str,
                 split: str = "train1",
                 input_size: int = 640,
                 transform=None,
                 augment: bool = True,
                 zero_based_class: bool = True):
        
        """
        root_dir: path to 'Datasets/Drive-india' folder
        split: 'train1','train2','train3','val','test'
        input_size: model input (square), letterboxing used
        transform: optional torchvision transforms applied AFTER letterbox (tensor, normalize, etc.)
        augment: enable/disable simple augmentations (flip)
        zero_based_class: set to False if labels in files are 1..N and you want 0..N-1 internally
        """
        
        self.root_dir = root_dir
        self.split = split
        self.input_size = input_size
        self.transform = transform
        self.augment = augment
        self.zero_based_class = zero_based_class
        all_files = build_file_list(self.root_dir, splits=[self.split])
        self.samples = all_files.get(self.split, [])
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found for split {split} in {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lbl_path = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size
        boxes, labels = parse_yolo_label_file(lbl_path, orig_w, orig_h, zero_based_class=self.zero_based_class)
        if self.augment:
            img, boxes = random_horizontal_flip(img, boxes, p=0.5)
        img, boxes, scale, pad = letterbox_resize(img, boxes, target_size=self.input_size)
        img_t = TF.to_tensor(img)  # float tensor on CPU
        target = {
            "boxes": boxes,
            "labels": labels
        }
        return img_t, target



def collate_fn(batch):
    imgs = [b[0] for b in batch]
    targets = [b[1] for b in batch]
    images = torch.stack(imgs, dim=0)
    return images, targets



def move_batch_to_device(batch, device):
    """
    Moves images and target tensors to `device`.
    batch: (images, targets) where targets is list of dicts with 'boxes' and 'labels'.
    Returns: images_on_device, targets_on_device (list of dicts)
    """
    images, targets = batch
    images = images.to(device, non_blocking=True)
    targets_on_device = []
    for t in targets:
        t_on = {}
        # boxes float, labels long
        t_on["boxes"] = t["boxes"].to(device) if t["boxes"].numel() > 0 else t["boxes"].to(device)
        t_on["labels"] = t["labels"].to(device) if t["labels"].numel() > 0 else t["labels"].to(device)
        targets_on_device.append(t_on)
    return images, targets_on_device


if __name__ == "__main__":
    ROOT = os.path.join(os.getcwd(), "Datasets", "Drive_India") 

    train_dataset = DriveIndiaDataset(ROOT, split="train1", input_size=640, augment=True)
    val_dataset = DriveIndiaDataset(ROOT, split="val", input_size=640, augment=False)
    test_dataset = DriveIndiaDataset(ROOT, split="test", input_size=640, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2,
                              pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2,
                            pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=2,
                             pin_memory=True, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    
    for images, targets in train_loader:
        images, targets = move_batch_to_device((images, targets), device)
        print("images:", images.shape, images.device)               # (B,3,H,W) on device
        print("num targets:", len(targets))
        if len(targets) > 0:
            print("first target boxes:", targets[0]['boxes'].shape, targets[0]['boxes'].device)
            print("first target labels:", targets[0]['labels'].shape, targets[0]['labels'].device)
        break
