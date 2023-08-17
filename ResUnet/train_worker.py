import torch
import torch.nn as nn
import os
from tqdm.auto import tqdm
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
import torchvision.models as models
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights
from torch.utils.tensorboard import SummaryWriter


#Modeling
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
        #Encoder
        x1 = self.encoder.conv1(x)
        x1 = self.encoder.bn1(x1)
        x1 = self.encoder.relu(x1)
        x1 = self.encoder.maxpool(x1)

        x2 = self.encoder.layer1(x1)
        x3 = self.encoder.layer2(x2)
        x4 = self.encoder.layer3(x3)
        x5 = self.encoder.layer4(x4)

        #Decoder
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

        #Resize
        x = nn.functional.interpolate(
            x, size=(1024, 1024), mode="bilinear", align_corners=False
        )

        return x


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
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
    pth_path,
    pth_name,
    log_dir,
):
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    best_score = 0

    model.train()
    for epoch in range(CFG["EPOCHS"]):
        train_loader.sampler.set_epoch(epoch)
        for img, label in tqdm(train_loader):
            optimizer.zero_grad()
            out = model(img.to(device))
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).to(device)
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).to(device))
            loss.backward()
            optimizer.step()

            loss_meter.update(loss.item())
            score_meter.update(score.item())

        train_loss_mean = loss_meter.avg
        train_score_mean = score_meter.avg
        loss_meter.reset()
        score_meter.reset()
        val_loss, val_score = validation(model, criterion, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_score)

        if device == 0:
            print(
                f"epoch{epoch+1}: Train_loss:{train_loss_mean} Train_score:{train_score_mean} Val_loss:{val_loss} Val_score:{val_score}"
            )
            writer = SummaryWriter(log_dir)
            writer.add_scalar("Train_Loss", train_loss_mean, global_step=epoch + 1)
            writer.add_scalar("Train_Score", train_score_mean, global_step=epoch + 1)
            writer.add_scalar("Validation_Loss", val_loss, global_step=epoch + 1)
            writer.add_scalar("Validation_Score", val_score, global_step=epoch + 1)

            if best_score < val_score:
                best_score = val_score
                best_model = model

        torch.distributed.barrier()

    if device == 0:
        os.makedirs(f"{pth_path}", exist_ok=True)
        torch.save(best_model.module.state_dict(), pth_name)

    return


def main_worker(gpu, world_size, train_set, val_set, CFG, pth_path, pth_name, log_dir):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://0.0.0.0:12345",
        world_size=world_size,
        rank=gpu,
    )
    torch.cuda.set_device(gpu)
    model = ResUNet(num_classes=1)
    model = model.cuda(gpu)
    model = DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )

    batch_size = int(CFG["BATCH_SIZE"] / (world_size))
    num_worker = int(CFG["num_worker"] / (world_size))

    train_sampler = DistributedSampler(
        dataset=train_set, num_replicas=world_size, shuffle=True
    )
    val_sampler = DistributedSampler(
        dataset=val_set, num_replicas=world_size, shuffle=False
    )

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
        min_lr=1e-10,
        verbose=True,
    )
    criterion = DiceLoss().to(gpu)

    train(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        gpu,
        CFG,
        pth_path,
        pth_name,
        log_dir,
    )

    return
