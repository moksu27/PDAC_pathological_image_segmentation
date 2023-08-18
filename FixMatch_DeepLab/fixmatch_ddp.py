import torch
from glob import glob
import datetime
import pytz
import torch.multiprocessing as mp
from fixmatch_worker import main_worker
from dataset.semi import SemiDataset
import os

CFG = {
    "IMG_SIZE": 1024,
    "EPOCHS": 10,
    "LEARNING_RATE": 1e-3,
    "BATCH_SIZE": 10,
    "SEED": 41,
    "num_worker": 4,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
    "nclass": 1,
}

kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")

log_dir = f"/workspace/FixMatch_DeepLab/log_dir/{day}/"
run_name = f"experiment1"

log_dir = os.path.join(log_dir, run_name)

os.makedirs(log_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"


pth_path = f"/workspace/FixMatch_DeepLab/pthfile/{day}/"
os.makedirs(pth_path, exist_ok=True)
pth_path = os.path.join(pth_path, run_name)

labeled_data_path = f"/workspace/git_ignore/PDA_labeled_tile/train/{CFG['train_magnification']}/**/*.png"
unlabeled_data_path = f"/workspace/git_ignore/PDA_unlabeled_tile/**/*_tiles/*.png"
val_data_path = f"/workspace/git_ignore/PDA_labeled_tile/validation/{CFG['train_magnification']}/**/*.png"


labeled_train_list = sorted(glob(labeled_data_path))
labeled_train_img = labeled_train_list[1::2][:10]
labeled_train_mask = labeled_train_list[0::2][:10]

val_path_list = sorted(glob(val_data_path))
val_img = val_path_list[1::2][:10]
val_mask = val_path_list[0::2][:10]

unlabeled_train_img = (sorted(glob(unlabeled_data_path)))[:10]


trainset_u = SemiDataset(img_root=unlabeled_train_img, mode="train_u")

trainset_l = SemiDataset(
    img_root=labeled_train_img, mask_root=labeled_train_mask, mode="train_l"
)

valset = SemiDataset(img_root=val_img, mask_root=val_mask, mode="val")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()

    mp.spawn(
        main_worker,
        nprocs=world_size,
        args=(
            world_size,
            trainset_u,
            trainset_l,
            valset,
            CFG,
            pth_path,
            log_dir,
        ),
        join=True,
    )
