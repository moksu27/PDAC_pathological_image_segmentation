import argparse
import yaml
import glob
import torch
import numpy as np
import torch.multiprocessing as mp
import random
import os
from albumentations.pytorch.transforms import ToTensorV2
import albumentations as A
from train_worker import main_worker
from dataset import CustomDataset


parser = argparse.ArgumentParser(description="pathology_project")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--save_path", required=True)

args = parser.parse_args()

cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

device = "cuda" if torch.cuda.is_available() else "cpu"


log_dir = f"{args.save_path}/log_dir"
os.makedirs(log_dir, exist_ok=True)

pth_path = f"{args.save_path}/pth"
os.makedirs(pth_path, exist_ok=True)


# Seed Fixing
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(cfg["seed"])

# Load Data
train_path = cfg["train_path"]
train_path = sorted(glob.glob(f"{train_path}/*.png"))
train_img_path = []
train_label_path = []

for file in train_path:
    if file.endswith("labelled.png"):
        train_label_path.append(file)
    elif file.endswith(".png"):
        train_img_path.append(file)

val_path = cfg["val_path"]
val_path = sorted(glob.glob(f"{val_path}/*.png"))
val_img_path = []
val_label_path = []

for file in val_path:
    if file.endswith("labelled.png"):
        val_label_path.append(file)
    elif file.endswith(".png"):
        val_img_path.append(file)

# transform
train_transform = A.Compose(
    [
        A.Resize(cfg["img_size"], cfg["img_size"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        A.OneOf(
            [A.HorizontalFlip(p=0.3), A.RandomRotate90(p=0.3), A.VerticalFlip(p=0.3)],
            p=0.3,
        ),
        ToTensorV2(transpose_mask=True),
    ]
)
val_transform = A.Compose(
    [
        A.Resize(cfg["img_size"], cfg["img_size"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True),
    ]
)


train_set = CustomDataset(
    img_path=train_img_path,
    mask_path=train_label_path,
    cfg=cfg,
    transform=train_transform,
)

val_set = CustomDataset(
    img_path=val_img_path, mask_path=val_label_path, cfg=cfg, transform=val_transform
)

# Distributed Data Parallel(DDP) RUN
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(world_size, train_set, val_set, cfg, pth_path, log_dir),
        join=True,
    )
