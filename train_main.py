import glob
import torch
import numpy as np
from tqdm.auto import tqdm
import torch.multiprocessing as mp
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime
import pytz
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from train_worker import main_worker
from dataset import CustomDataset
import mlflow
import pdb


device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "IMG_SIZE": 1024,
    "EPOCHS": 40,
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 40,
    "SEED": 41,
    "num_worker": 12,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "10X",
}

# 경로 및 변수 지정
kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")

output_path = f"/workspace/git_ignore/output/{day}"
pth_path = f"/workspace/git_ignore/pthfile/{day}"
trainframe_path = f"{output_path}/trainframe"
train_data_path = f"/workspace/git_ignore/PDA_mask_img(1024)/train/{CFG['train_magnification']}/**/*.png"
val_data_path = f"/workspace/git_ignore/PDA_mask_img(1024)/validation/{CFG['train_magnification']}/**/*.png"

trainframe_name = f"{trainframe_path}/train:{CFG['train_magnification']}_epoch:{CFG['EPOCHS']}_{hour}.xlsx"
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
if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 주소 설정
    with mlflow.start_run(run_name=CFG["train_magnification"]):
        mlflow.log_param("IMG_SIZE", CFG["IMG_SIZE"])
        mlflow.log_param("EPOCHS", CFG["EPOCHS"])
        mlflow.log_param("BATCH_SIZE", CFG["BATCH_SIZE"])
        mlflow.log_param("Magnification", CFG["train_magnification"])

        mp.spawn(
            main_worker,
            nprocs=world_size,
            args=(world_size, train_set, val_set, CFG, pth_path, pth_name),
            join=True,
        )
    mlflow.end_run()
