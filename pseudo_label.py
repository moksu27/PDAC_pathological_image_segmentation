import openslide
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
import random
import os
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime
import pytz
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import pandas as pd
import tifffile as tiff

device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "IMG_SIZE": 1024,
    "BATCH_SIZE": 20,
    "SEED": 41,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
}


# 시드 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["SEED"])  # Seed 고정

# 현재 시각
kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")
print(day, hour)

# path
svspath = r"/workspace/git_ignore/PDA_svs_img/C3L-01637-21.svs"
output_path = f"/workspace/git_ignore/output/{day}"
figure_path = f"{output_path}/figure"
test_data_path = r"/workspace/git_ignore/PDA_tile_img/C3L-01637-21/C3L-01637-21_tiles"
tsv_path = r"/workspace/git_ignore/PDA_tile_img/C3L-01637-21/tile_selection.tsv"
pth_path = r"/workspace/git_ignore/pthfile/2023_06_19/train:20X_epoch:20_03:25_PM.pth"
