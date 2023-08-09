import torch
from glob import glob
import datetime
import pytz
import mlflow
import torch.multiprocessing as mp
from fixmatch_worker import main_worker
from dataset.semi import SemiDataset
import os

CFG = {
    "IMG_SIZE": 1024,
    "crop_size": 256,
    "EPOCHS": 40,
    "LEARNING_RATE": 1e-5,
    "BATCH_SIZE": 20,
    "SEED": 41,
    "num_worker": 4,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
    "nclass": 1,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# 경로 및 변수 지정
kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
# hour = current_datetime.strftime("%I:%M_%p")

pth_path = f"/workspace/FixMatch_Unet/pthfile/{day}"
pth_name = f"{pth_path}/M:{CFG['train_magnification']}_E:{CFG['EPOCHS']}"

labeled_data_path = f"/workspace/git_ignore/PDA_labeled_tile/train/{CFG['train_magnification']}/**/*.png"
unlabeled_data_path = f"/workspace/git_ignore/PDA_unlabeled_tile/**/*_tiles/*.png"
val_data_path = f"/workspace/git_ignore/PDA_labeled_tile/validation/{CFG['train_magnification']}/**/*.png"


labeled_train_list = sorted(glob(labeled_data_path))
labeled_train_img = labeled_train_list[1::2]
labeled_train_mask = labeled_train_list[0::2]

val_path_list = sorted(glob(val_data_path))
val_img = val_path_list[1::2]
val_mask = val_path_list[0::2]

unlabeled_train_img = sorted(glob(unlabeled_data_path))


trainset_u = SemiDataset(
    img_root=unlabeled_train_img, mode="train_u", size=CFG["crop_size"]
)

trainset_l = SemiDataset(
    img_root=labeled_train_img,
    mask_root=labeled_train_mask,
    mode="train_l",
    size=CFG["crop_size"],
)

valset = SemiDataset(img_root=val_img, mask_root=val_mask, mode="val")

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 주소 설정
    magnification = CFG["train_magnification"]

    with mlflow.start_run(
        run_name=f"{magnification}_{day}_Fixmatch_Unet",
        experiment_id=535374782000415794,
    ) as run:
        run_id = run.info.run_id
        mlflow.log_param("IMG_SIZE", CFG["IMG_SIZE"])
        mlflow.log_param("EPOCHS", CFG["EPOCHS"])
        mlflow.log_param("BATCH_SIZE", CFG["BATCH_SIZE"])
        mlflow.log_param("Magnification", CFG["train_magnification"])
        mlflow.end_run()

    world_size = torch.cuda.device_count()

    os.makedirs(os.path.join(pth_path), exist_ok=True)

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
            pth_name,
            run_id,
        ),
        join=True,
    )
