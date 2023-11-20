import glob
import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import ToPILImage
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse
import yaml
import matplotlib as mpl
import matplotlib.pyplot as plt
from dataset import CustomDataset
from collections import OrderedDict
import segmentation_models_pytorch as smp
from models.resunet import ResUNet
from util import *


parser = argparse.ArgumentParser(description="pathology_project")
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--save_path", required=True)
parser.add_argument("--pth_path", required=True)


args = parser.parse_args()
cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)


device = "cuda" if torch.cuda.is_available() else "cpu"

figure_path = f"{args.save_path}/figure"
os.makedirs(figure_path, exist_ok=True)


# Seed 고정
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(cfg["seed"])

test_path = cfg["test_path"]
test_path = sorted(glob.glob(f"{test_path}/*.png"))
test_img_path = []
test_label_path = []

for file in test_path:
    if file.endswith("labelled.png"):
        test_label_path.append(file)
    elif file.endswith(".png"):
        test_img_path.append(file)

test_transform = A.Compose(
    [
        A.Resize(cfg["img_size"], cfg["img_size"]),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(transpose_mask=True),
    ]
)

# test data set
test_set = CustomDataset(
    img_path=test_img_path, mask_path=test_label_path, cfg=cfg, transform=test_transform
)
test_loader = DataLoader(test_set, batch_size=cfg["batch_size"])


# Set Model
if cfg["model"] == "unet":
    model = ResUNet(num_classes=1, output_size=cfg["img_size"])

elif cfg["model"] == "fpn":
    model = smp.FPN(
        encoder_name=cfg[
            "backbone"
        ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation=None,
    )

elif cfg["model"] == "deeplabv3+":
    model = smp.DeepLabV3Plus(
        encoder_name=cfg["backbone"],
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
        activation=None,
        decoder_atrous_rates=cfg["dilations"],
    )
elif cfg["model"] == "pspnet+":
    model = smp.PSPNet(
        encoder_name=cfg[
            "backbone"
        ],  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation=None,
        psp_use_batchnorm=True,
    )

model = model.to(device)
criterion = DiceLoss().to(device)
loss_meter = AverageMeter()
score_meter = AverageMeter()

loaded_state_dict = torch.load(args.pth_path)
new_state_dict = OrderedDict()
for n, v in loaded_state_dict["model"].items():
    name = n.replace("module.", "")
    new_state_dict[name] = v

model.load_state_dict(new_state_dict)


# Figure output
def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


mpl.rcParams["image.cmap"] = "inferno"


def Test(model, criterion, test_loader, device):
    count = 0
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(test_loader):
            out = model(img.to(device))
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).to(device)
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).to(device))

            loss_meter.update(loss.item())
            score_meter.update(score.item())

            for i in range(len(img)):
                # 오리지널 이미지
                plt.subplot(1, 3, 1)
                plt.title("original")
                plt.imshow(ToPILImage()(denormalize(img[i].cpu())))
                plt.xticks([])
                plt.yticks([])

                # 마스크 이미지
                plt.subplot(1, 3, 2)
                plt.title("label")
                plt.imshow(label[i].cpu())
                plt.xticks([])
                plt.yticks([])

                # 마스크 예측 이미지
                plt.subplot(1, 3, 3)
                plt.title("predicted")
                plt.tight_layout()
                plt.imshow(pred[i].cpu())
                plt.xticks([])
                plt.yticks([])
                plt.savefig(
                    f"{figure_path}/figure_{count+1}.png",
                    bbox_inches="tight",
                )
                count += 1
        test_loss_mean = loss_meter.avg
        test_score_mean = score_meter.avg

        print(f"Test Score: {test_score_mean} Test Loss: {test_loss_mean}")
        loss_meter.reset()
        score_meter.reset()

    return


# RUN
if __name__ == "__main__":
    Test(model, criterion, test_loader, device)
