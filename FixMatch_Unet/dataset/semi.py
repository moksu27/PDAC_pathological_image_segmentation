from dataset.transform import *
from copy import deepcopy
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


CFG = {
    "IMG_SIZE": 1024,
    "crop_size": 256,
    "EPOCHS": 1,
    "LEARNING_RATE": 1e-5,
    "lr_multi": 10.0,
    "BATCH_SIZE": 5,
    "SEED": 41,
    "num_worker": 12,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
}

transform_labeled = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        A.HorizontalFlip(p=0.3),
        ToTensorV2(transpose_mask=True),
    ]
)

transform_val = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        ToTensorV2(transpose_mask=True),
    ]
)

weak_transform = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        ToTensorV2(),
    ]
)

strong_transform = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.5,
        ),
        A.RandomBrightnessContrast(p=0.5),
        A.OneOf(
            [
                A.ColorJitter(
                    p=0.5, brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25
                ),
                A.RandomGamma(p=0.5),
                A.NoOp(p=0.5),
            ],
            p=0.5,
        ),
        A.OneOf(
            [
                A.Blur(blur_limit=(3, 7), p=0.5),
                A.GaussNoise(p=0.5),
                A.NoOp(p=0.5),
            ],
            p=0.5,
        ),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        ToTensorV2(),
    ]
)


class SemiDataset(Dataset):
    def __init__(self, img_root, mode, mask_root=None, size=None):
        self.img_root = img_root
        self.mask_root = mask_root
        self.mode = mode
        self.size = size

    def __getitem__(self, i):
        img = np.array(Image.open(self.img_root[i]))
        if self.mask_root is not None:
            mask = np.array(Image.open(self.mask_root[i]))

        if self.mode == "val":
            data = transform_val(image=img, mask=mask)
            image = data["image"]
            mask = data["mask"]
            return image, mask

        elif self.mode == "train_l":
            data = transform_labeled(image=img, mask=mask)
            image = data["image"]
            mask = data["mask"]
            return image, mask

        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        data_w = weak_transform(image=img_w)
        img_w = data_w["image"]

        data_s1 = strong_transform(image=img_s1)
        img_s1 = data_s1["image"]
        cutmix_box1 = obtain_cutmix_box(CFG["IMG_SIZE"], p=0.5)

        data_s2 = strong_transform(image=img_s2)
        img_s2 = data_s2["image"]
        cutmix_box2 = obtain_cutmix_box(CFG["IMG_SIZE"], p=0.5)

        return img_w, img_s1, img_s2, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.img_root)
