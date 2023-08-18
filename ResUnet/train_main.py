import glob
import torch
import numpy as np
import torch.multiprocessing as mp
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime
import pytz
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from Unet_smp.train_worker import main_worker
from dataset import CustomDataset
from torch.utils.tensorboard import SummaryWriter

log_dir = "C:/Users/kim/Desktop/bsm/pathology_image_project/ResUnet/log_dir"
writer = SummaryWriter(log_dir)
device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "IMG_SIZE": 1024,
    "EPOCHS": 30,
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 30,
    "SEED": 41,
    "num_worker": 12,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
}

# 경로 및 변수 지정
kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")

pth_path = f"C:/Users/kim/Desktop/bsm/pathology_image_project/ResUnet/pthfile/{day}"
train_data_path = f"C:/Users/kim/Desktop/bsm/pathology_image_project/git_ignore/PDA_labeled_tile/train/{CFG['train_magnification']}/**/*.png"
val_data_path = f"C:/Users/kim/Desktop/bsm/pathology_image_project/git_ignore/PDA_labeled_tile/validation/{CFG['train_magnification']}/**/*.png"
pth_name = f"{pth_path}/M:{CFG['train_magnification']}_E:{CFG['EPOCHS']}_{hour}.pth"


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
world_size = torch.cuda.device_count()
magnification = CFG["train_magnification"]
if __name__ == "__main__":
    writer.add_scalar("IMG_SIZE", CFG["IMG_SIZE"])
    writer.add_scalar("EPOCHS", CFG["EPOCHS"])
    writer.add_scalar("BATCH_SIZE", CFG["BATCH_SIZE"])
    writer.add_scalar("Magnification", CFG["train_magnification"])

    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(world_size, train_set, val_set, CFG, pth_path, pth_name, log_dir),
        join=True,
    )
    writer.close()