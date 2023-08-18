import glob
import torch
import numpy as np
import random
import albumentations as A
import datetime
import pytz
from albumentations.pytorch.transforms import ToTensorV2
from dataset import CustomDataset
from train_worker import main_worker
import os

CFG = {
    "IMG_SIZE": 1024,
    "EPOCHS": 50,
    "LEARNING_RATE": 1e-6,
    "BATCH_SIZE": 16,
    "SEED": 41,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
    "num_worker":0
}

# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["SEED"])


kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")

magnification=CFG["train_magnification"]
log_dir = f"C:/Users/kim/Desktop/bsm/pathology_image_project/Unet_smp/log_dir/{day}/"
run_name = f"experiment_{magnification}_resnet18"

log_dir = os.path.join(log_dir,run_name)

os.makedirs(log_dir, exist_ok=True)


device = "cuda" if torch.cuda.is_available() else "cpu"

pth_path = f"C:/Users/kim/Desktop/bsm/pathology_image_project/Unet_smp/pthfile/{day}/"
os.makedirs(pth_path, exist_ok=True)
pth_name = os.path.join(pth_path, run_name)


train_data_path = f"C:/Users/kim/Desktop/bsm/pathology_image_project/git_ignore/PDA_labeled_tile/train/{CFG['train_magnification']}/**/*.png"
val_data_path = f"C:/Users/kim/Desktop/bsm/pathology_image_project/git_ignore/PDA_labeled_tile/validation/{CFG['train_magnification']}/**/*.png"

# 데이터 불러오기
train_path_list = sorted(glob.glob(train_data_path))
train_mask_path = train_path_list[0::2]
train_img_path = train_path_list[1::2]

val_path_list = sorted(glob.glob(val_data_path))
val_mask_path = val_path_list[0::2]
val_img_path = val_path_list[1::2]

# transform
train_transform = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        A.OneOf(
            [A.HorizontalFlip(p=0.3), A.RandomRotate90(p=0.3), A.VerticalFlip(p=0.3)],
            p=0.3,
        ),
        ToTensorV2(transpose_mask=True),
    ]
)
val_transform = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        ToTensorV2(transpose_mask=True),
    ]
)

train_set = CustomDataset(
    img_path=train_img_path,
    mask_path=train_mask_path,
    CFG=CFG,
    transform=train_transform,
)

val_set = CustomDataset(
    img_path=val_img_path, mask_path=val_mask_path, CFG=CFG, transform=val_transform
)


# 분산 학습 RUN
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    main_worker(world_size,train_set,val_set,CFG,pth_path,log_dir)
