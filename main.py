import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data.dataset import Dataset
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from torchvision.transforms import ToPILImage
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import torchvision.models as models
import random
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from IPython.core.debugger import set_trace
import pandas as pd
import datetime
import pytz
import matplotlib as mpl

# parameter
CFG = {
    "IMG_SIZE": 512,
    "EPOCHS": 10,
    "LEARNING_RATE": 3e-4,
    "BATCH_SIZE": 64,
    "SEED": 41,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": 20,
    "test_magnification": 20,
}


# Seed setting
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


seed_everything(CFG["SEED"])


# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, img_path, mask_path, transform=None):
        self.image = img_path
        self.mask = mask_path
        n_samples = len(self.image)

        # mix data
        np.random.seed(CFG["SEED"])
        idxs = np.random.permutation(range(n_samples))

        self.image = np.array(self.image)[idxs]
        self.mask = np.array(self.mask)[idxs]
        self.transform = transform

    def __len__(self):
        return len(self.image)

    def __getitem__(self, i):
        image = np.array(Image.open(self.image[i]))
        mask = np.array(Image.open(self.mask[i]))
        data = self.transform(image=image, mask=mask)
        image = data["image"]
        mask = data["mask"]
        return image, mask


# ResNet + UNet Model
class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.encoder = models.resnet18(weights="DEFAULT")
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

        # Resize to 512x512
        x = nn.functional.interpolate(
            x, size=(512, 512), mode="bilinear", align_corners=False
        )
        return x


# dice score function
def dice_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


# Average function
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


# EarlyStop function
class EarlyStop:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_score):
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.counter = 0


# denormalize
def denormalize(tensor, mean=CFG["MEAN"], std=CFG["STD"]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


# cuda setting
device = "cuda" if torch.cuda.is_available() else "cpu"

# time variable declaration
kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
formatted_datetime = current_datetime.strftime("%Y_%m_%d_%I:%M_%p")

# server path
pth_name = f"/data/pthfile/train:{CFG['train_magnification']}X_test:{CFG['test_magnification']}X_epoch:{CFG['EPOCHS']}_({formatted_datetime}).pth"
output_name = f"train:{CFG['train_magnification']}X_test:{CFG['test_magnification']}X_epoch:{CFG['EPOCHS']}"
output_path = f"/data/output/output_{output_name}_({formatted_datetime})"
plot_path = f"/data/plot/plot_{output_name}_({formatted_datetime})"
train_data_path = f"/data/PDA_mask_img/train/{CFG['train_magnification']}X/**/*.png"
test_data_path = f"/data/PDA_mask_img/test_mask/{CFG['test_magnification']}X/**/*.png"
val_data_path = (
    f"/data/PDA_mask_img/validation_mask/{CFG['train_magnification']}X/**/*.png"
)

# local path
"""
pth_name=f"git_ignore/pthfile/train:{CFG['train_magnification']}X_test:{CFG['test_magnification']}X_epoch:{CFG['EPOCHS']}_({formatted_datetime}).pth"
output_name = f"train:{CFG['train_magnification']}X_test:{CFG['test_magnification']}X_epoch:{CFG['EPOCHS']}"
output_path = f"git_ignore/output/output_{output_name}_({formatted_datetime})"
plot_path = f"git_ignore/plot/plot_{output_name}_({formatted_datetime})"
train_data_path = f"git_ignore/PDA_mask_img/train/{CFG['train_magnification']}X/**/*.png"
test_data_path = f"git_ignore/PDA_mask_img/test/{CFG['test_magnification']}X/**/*.png"
val_data_path = f"git_ignore/PDA_mask_img/validation/{CFG['train_magnification']}X/**/*.png"
"""

# train path
train_path_list = sorted(glob.glob(train_data_path))
train_mask_path = train_path_list[0::2]
train_img_path = train_path_list[1::2]

# test path
test_path_list = sorted(glob.glob(test_data_path))
test_mask_path = test_path_list[0::2]
test_img_path = test_path_list[1::2]

# validation path
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
            [A.HorizontalFlip(p=0.5), A.RandomRotate90(p=0.5), A.VerticalFlip(p=0.5)],
            p=0.5,
        ),
        A.OneOf(
            [A.MotionBlur(p=0.5), A.OpticalDistortion(p=0.5), A.GaussNoise(p=0.5)],
            p=0.5,
        ),
        ToTensorV2(transpose_mask=True),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(CFG["IMG_SIZE"], CFG["IMG_SIZE"]),
        A.Normalize(mean=CFG["MEAN"], std=CFG["STD"]),
        ToTensorV2(transpose_mask=True),
    ]
)

# data load
train_set = CustomDataset(
    img_path=train_img_path, mask_path=train_mask_path, transform=train_transform
)
val_set = CustomDataset(
    img_path=val_img_path, mask_path=val_mask_path, transform=test_transform
)
test_set = CustomDataset(
    img_path=test_img_path, mask_path=test_mask_path, transform=test_transform
)

train_loader = DataLoader(train_set, batch_size=CFG["BATCH_SIZE"], shuffle=True)
val_loader = DataLoader(val_set, batch_size=CFG["BATCH_SIZE"])
test_loader = DataLoader(test_set, batch_size=CFG["BATCH_SIZE"])

# learning parameter
model = UNet(num_classes=1).to(device)
model = nn.DataParallel(model)
optimizer = Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2,
    threshold_mode="abs",
    min_lr=1e-8,
    verbose=True,
)
criterion = nn.BCEWithLogitsLoss().to(device)

loss_meter = AverageMeter()
score_meter = AverageMeter()
early_stopping = EarlyStop(patience=20, delta=0)


# validation model
def validation(model, criterion, val_loader, device):
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(val_loader):
            out = model(img.to(device))
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).to(device)
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).to(device))

            loss_meter.update(loss.item())
            score_meter.update(score.item())

        val_loss_mean = loss_meter.avg
        val_score_mean = score_meter.avg
        loss_meter.reset()
        score_meter.reset()
    return val_loss_mean, val_score_mean


# train model
def train(model, criterion, optimizer, train_loader, val_loader, scheduler, device):
    best_score = 0
    best_model = None
    result_arr = np.empty((0, 4), float)
    columns = []
    model.train()
    for epoch in range(CFG["EPOCHS"]):
        try:
            for img, label in tqdm(train_loader):
                optimizer.zero_grad()
                out = model(img.to(device))
                out = torch.squeeze(out)
                pred = torch.ge(out.sigmoid(), 0.5).float()
                label = torch.squeeze(label).to(device)
                score = dice_score(pred, label)
                loss = criterion(out, label.type(torch.FloatTensor).to(device))

                loss_meter.update(loss.item())
                score_meter.update(score.item())

                loss.backward()
                optimizer.step()

            train_loss_mean = loss_meter.avg
            train_score_mean = score_meter.avg
            loss_meter.reset()
            score_meter.reset()
            val_loss, val_score = validation(model, criterion, val_loader, device)

            print(
                f"epoch{epoch+1}: Train_loss:{train_loss_mean} Train_score:{train_score_mean} Val_loss:{val_loss} Val_score:{val_score}"
            )
            result_arr = np.append(
                result_arr,
                np.array([[train_loss_mean, train_score_mean, val_loss, val_score]]),
                axis=0,
            )
            if scheduler is not None:
                scheduler.step(val_score)

            if best_score < val_score:
                best_score = val_score
                best_model = model

            early_stopping(val_score)
            if early_stopping.early_stop:
                columns.append(f"epoch:{epoch+1}")
                print("Early stopping!")
                break
        except KeyboardInterrupt:
            best_model = model
        columns.append(f"epoch:{epoch+1}")
    return best_model, result_arr, columns


# Run
infer_model, result, columns = train(
    model, criterion, optimizer, train_loader, val_loader, scheduler, device
)

# result save
result_df = pd.DataFrame(
    data=result,
    index=columns,
    columns=[
        "Train Loss",
        "Train Dice Score",
        "Validation Loss",
        "Validation Dice Score",
    ],
)
result_df.to_excel(f"{output_path}.xlsx")

# model save
os.makedirs(output_path, exist_ok=True)
result_df.to_excel(f"{output_path}/{output_name}.xlsx")


# test model
def Test(model, criterion, test_loader, device):
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
        loss_meter.reset()
        score_meter.reset()

    return test_loss_mean, test_score_mean


# test
test_loss, test_score = Test(infer_model, criterion, test_loader, device)
print(test_loss, test_score)

# output_save
mpl.rcParams["image.cmap"] = "inferno"
os.makedirs(output_path, exist_ok=True)

for i in range(5):
    data, label = test_set[i]
    label = torch.squeeze(label)

    with torch.no_grad():
        out = model(torch.unsqueeze(data, dim=0).to(device))
    out = torch.squeeze(out).sigmoid().to("cpu")
    pred = torch.ge(out, 0.5).float().to("cpu")

    # original image
    plt.subplot(1, 3, 1)
    plt.title("original")
    plt.imshow(ToPILImage()(denormalize(data)))
    plt.xticks([])
    plt.yticks([])
    # mask image
    plt.subplot(1, 3, 2)
    plt.title("mask")
    plt.imshow(label)
    plt.xticks([])
    plt.yticks([])
    # predict mask image
    plt.subplot(1, 3, 3)
    plt.title("predicted")
    plt.tight_layout()
    plt.imshow(pred)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(f"{output_path}/{output_name}_{i+1}.png")
