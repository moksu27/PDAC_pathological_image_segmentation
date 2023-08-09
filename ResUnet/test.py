import glob
import torch
import torch.nn as nn
import numpy as np
from torchvision.models import ResNet18_Weights
from tqdm.auto import tqdm
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
from torchvision.transforms import ToPILImage
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
import datetime
import pytz
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.pyplot as plt
import mlflow
from dataset import CustomDataset

device = "cuda" if torch.cuda.is_available() else "cpu"

CFG = {
    "IMG_SIZE": 1024,
    "BATCH_SIZE": 40,
    "SEED": 41,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "test_magnification": "20X",
    "pth": "/workspace/ResUnet/pthfile/2023_07_27/M:20X_E:30_03:42_PM.pth",
}

kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")
output_path = f"/workspace/ResUnet/output/{day}"
figure_path = f"{output_path}/figure"
pth_name = CFG["pth"].split("/")[5][:-13]
figure_name = f"{pth_name}_test"
run_id = "0af0a9b34ede464ea1aae4ae4a43c112"


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

test_data_path = (
    f"/workspace/git_ignore/PDA_labeled_tile/test/{CFG['test_magnification']}/**/*.png"
)
test_path_list = sorted(glob.glob(test_data_path))
test_mask_path = test_path_list[0::2]
test_img_path = test_path_list[1::2]

test_transform = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        ToTensorV2(transpose_mask=True),
    ]
)

# test data set
test_set = CustomDataset(
    img_path=test_img_path, mask_path=test_mask_path, CFG=CFG, transform=test_transform
)

test_loader = DataLoader(test_set, batch_size=CFG["BATCH_SIZE"])


# modeling
# Modeling
class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()
        self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x1)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)

        # Decoder
        x = self.upconv1(x5)
        x = torch.cat((x, x4), dim=1)
        x = self.relu(self.conv1(x))

        x = self.upconv2(x)
        x = torch.cat((x, x3), dim=1)
        x = self.relu(self.conv2(x))

        x = self.upconv3(x)
        x = torch.cat((x, x2), dim=1)
        x = self.relu(self.conv3(x))

        x = self.conv4(x)

        # Resize
        x = nn.functional.interpolate(
            x, size=(1024, 1024), mode="bilinear", align_corners=False
        )

        return x


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)  # sigmoid를 통과한 출력이면 주석처리

        inputs = inputs.view(-1)
        targets = targets.view(-1)

        intersection = (inputs * targets).sum()
        dice = (2.0 * intersection + smooth) / (inputs.sum() + targets.sum() + smooth)

        return 1 - dice


def dice_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


model = ResUNet(num_classes=1).to(device)
criterion = DiceLoss().to(device)
loss_meter = AverageMeter()
score_meter = AverageMeter()

model.load_state_dict(torch.load(CFG["pth"], map_location=device))


def Test(model, criterion, test_loader, device, run_id):
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

        test_loss_mean = loss_meter.avg
        test_score_mean = score_meter.avg
        mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 주소 설정
        with mlflow.start_run(run_id=run_id, experiment_id=0):
            mlflow.log_metric("Test Loss", test_loss_mean, step=1)
            mlflow.log_metric("Test Score", test_score_mean, step=1)
            mlflow.end_run()

        loss_meter.reset()
        score_meter.reset()

    return


# RUN
if __name__ == "__main__":
    Test(model, criterion, test_loader, device, run_id)


# Figure output
def denormalize(tensor, mean=CFG["MEAN"], std=CFG["STD"]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


mpl.rcParams["image.cmap"] = "inferno"

# os.makedirs(figure_path, exist_ok=True)

# with mlflow.start_run(run_id=run_id):
for i in tqdm(range(len(test_set))):
    data, label = test_set[i]
    label = torch.squeeze(label)

    with torch.no_grad():
        out = model(torch.unsqueeze(data, dim=0).to(device))
    out = torch.squeeze(out).sigmoid().to("cpu")
    pred = torch.ge(out, 0.5).float().to("cpu")

    # 오리지널 이미지
    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.imshow(ToPILImage()(denormalize(data)))
    plt.xticks([])
    plt.yticks([])

    # 마스크 이미지
    plt.subplot(1, 3, 2)
    plt.title("mask")
    plt.imshow(label)
    plt.xticks([])
    plt.yticks([])

    # 마스크 예측 이미지
    plt.subplot(1, 3, 3)
    plt.title("predicted")
    plt.tight_layout()
    plt.imshow(pred)
    plt.xticks([])
    plt.yticks([])
    # plt.savefig(f"{figure_path}/test_figure_{i+1}.png", bbox_inches="tight")

    #     mlflow.log_artifact(f"{figure_path}/test_figure_{i+1}.png", "Test_Figure")
    # mlflow.end_run()
