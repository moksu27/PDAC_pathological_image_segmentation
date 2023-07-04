import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights


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


def validation(model, criterion, val_loader, device):
    loss_meter2 = AverageMeter()
    score_meter2 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(val_loader):
            out = model(img.to(device))
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).to(device)
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).to(device))

            loss_meter2.update(loss.item())
            score_meter2.update(score.item())

        val_loss_mean = loss_meter2.avg
        val_score_mean = score_meter2.avg
        loss_meter2.reset()
        score_meter2.reset()
    return val_loss_mean, val_score_mean


def train(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    scheduler,
    device,
    CFG,
    train_sampler,
):
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    early_stopping = EarlyStop(patience=20, delta=0)
    best_score = 0
    best_model = None
    result_arr = np.empty((0, 4), float)
    columns = []
    model.train()

    for epoch in range(CFG["EPOCHS"]):
        train_sampler.set_epoch(epoch)
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


def main_worker(gpu, world_size, train_set, val_set, CFG):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://0.0.0.0:12345",
        world_size=world_size,
        rank=gpu,
    )
    torch.cuda.set_device(gpu)
    torch.distributed.barrier()
    model = ResUNet(num_classes=1)
    model = model.cuda(gpu)
    model = DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )

    train_sampler = DistributedSampler(
        dataset=train_set, num_replicas=world_size, shuffle=True
    )
    val_sampler = DistributedSampler(
        dataset=val_set, num_replicas=world_size, shuffle=False
    )

    batch_size = int(CFG["BATCH_SIZE"] / world_size)
    num_worker = int(CFG["num_worker"] / world_size)
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        sampler=train_sampler,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        sampler=val_sampler,
        pin_memory=True,
    )

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
    criterion = DiceLoss().to(gpu)
    infer_model, result, columns = train(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        gpu,
        CFG,
        train_sampler,
    )

    return infer_model, result, columns
