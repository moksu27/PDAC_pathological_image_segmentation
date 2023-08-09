import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
import mlflow
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from util.utils import AverageMeter
import torch.distributed as dist
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet18_Weights
import pdb


class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()
        self.encoder = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU(inplace=False)
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


def dice_score(pred, target, smooth=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice


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


def validation(model, criterion, val_loader, gpu):
    loss_meter2 = AverageMeter()
    score_meter2 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(val_loader):
            out = model(img.to(gpu))
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).to(gpu)
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).to(gpu))

            loss_meter2.update(loss.item())
            score_meter2.update(score.item())

        val_loss_mean = loss_meter2.avg
        val_score_mean = score_meter2.avg
        loss_meter2.reset()
        score_meter2.reset()
    return val_loss_mean, val_score_mean


def train(
    model,
    criterion_x,
    criterion_u,
    optimizer,
    trainloader_l,
    trainloader_u,
    valloader,
    scheduler,
    gpu,
    CFG,
    pth_path,
    pth_name,
    run_id,
):
    previous_best = 0.0
    epoch = -1

    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()

    if os.path.exists(os.path.join(pth_path, pth_name + "_latest.pth")):
        checkpoint = torch.load(os.path.join(pth_path, pth_name + "_latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

    for epoch in range(epoch + 1, CFG["EPOCHS"] + 1):
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        for (
            (img_x, mask_x),
            (img_u_w, img_u_s, _, cutmix_box, _),
            (img_u_w_mix, img_u_s_mix, _, _, _),
        ) in zip(tqdm(trainloader_l), trainloader_u, trainloader_u):
            mask_x = torch.squeeze(mask_x).to(gpu)
            with torch.no_grad():
                model.eval()
                out_u_w_mix = model(img_u_w_mix.to(gpu))
                pred_u_w_mix = torch.squeeze(out_u_w_mix)
                mask_u_w_mix = torch.ge(pred_u_w_mix.sigmoid(), 0.5).float()

                img_u_s[
                    cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1
                ] = img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            torch.distributed.barrier()

            model.train()
            out_u_w = model(img_u_w.to(gpu))
            pred_u_w = torch.squeeze(out_u_w)
            mask_u_w = torch.ge(pred_u_w.sigmoid(), 0.5).float()
            mask_u_w_cutmixed = mask_u_w.clone()
            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[
                cutmix_box == 1
            ]  # psuedo label

            out_u_s = model(img_u_s.to(gpu))
            pred_u_s = torch.squeeze(out_u_s)

            out_x = model(img_x.to(gpu))
            pred_x = torch.squeeze(out_x)

            loss_x = criterion_x(pred_x, mask_x.type(torch.FloatTensor).to(gpu))
            loss_u_s = criterion_u(
                pred_u_s, mask_u_w_cutmixed.type(torch.FloatTensor).to(gpu)
            )
            # loss = (loss_x + loss_u_s) / 2.0

            loss = loss_x.add(loss_u_s) * 0.5

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())

        train_loss_mean = total_loss.avg
        total_loss.reset()
        total_loss_x.reset()
        total_loss_s.reset()

        val_loss, val_score = validation(model, criterion_x, valloader, gpu)

        if scheduler is not None:
            scheduler.step(val_score)

        if gpu == 0:
            print(
                f"epoch{epoch+1}: Train_loss:{train_loss_mean} Val_loss:{val_loss} Val_score:{val_score}"
            )
            with mlflow.start_run(run_id=run_id, experiment_id=535374782000415794):
                mlflow.log_metric("Train_Loss", train_loss_mean, step=epoch + 1)
                mlflow.log_metric("Validation_Loss", val_loss, step=epoch + 1)
                mlflow.log_metric("Validation_Score", val_score, step=epoch + 1)
                mlflow.end_run()

        is_best = val_score > previous_best
        previous_best = max(val_score, previous_best)

        if gpu == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(pth_path, pth_name + "_latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(pth_path, pth_name + "_best.pth"))

        torch.distributed.barrier()

    return


def main_worker(
    gpu,
    world_size,
    trainset_u,
    trainset_l,
    valset,
    CFG,
    pth_path,
    pth_name,
    run_id,
):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://0.0.0.0:12345",
        world_size=world_size,
        rank=gpu,
    )

    cudnn.enabled = True
    cudnn.benchmark = True

    model = ResUNet(num_classes=1)
    model = model.to(gpu)

    model = DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True, broadcast_buffers=False
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
    criterion_x = DiceLoss().to(gpu)
    criterion_u = DiceLoss().to(gpu)

    if gpu == 0 or 2:
        batch_size = int(CFG["BATCH_SIZE"] / (world_size + 1))
        num_worker = int(CFG["num_worker"] / (world_size + 1))

    elif gpu == 1:
        batch_size = int(CFG["BATCH_SIZE"] / (world_size + 1)) * 3
        num_worker = int(CFG["num_worker"] / (world_size + 1)) * 3

    trainsampler_l = DistributedSampler(
        dataset=trainset_l, num_replicas=world_size, shuffle=True
    )
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_worker,
        drop_last=True,
        sampler=trainsampler_l,
    )

    trainsampler_u = DistributedSampler(
        dataset=trainset_u, num_replicas=world_size, shuffle=True
    )
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_worker,
        drop_last=True,
        sampler=trainsampler_u,
    )

    valsampler = DistributedSampler(
        dataset=valset, num_replicas=world_size, shuffle=False
    )
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_worker,
        drop_last=False,
        sampler=valsampler,
    )

    train(
        model,
        criterion_x,
        criterion_u,
        optimizer,
        trainloader_l,
        trainloader_u,
        valloader,
        scheduler,
        gpu,
        CFG,
        pth_path,
        pth_name,
        run_id,
    )

    return
