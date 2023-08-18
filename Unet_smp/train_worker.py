import torch
import os
from tqdm.auto import tqdm
from torch.optim.adam import Adam
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter

device = "cuda" if torch.cuda.is_available() else "cpu"


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


def validation(model, criterion, val_loader):
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
    CFG,
    pth_path,
    log_dir,
):
    previous_best = 0.0
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    epoch = CFG["EPOCHS"]

    if os.path.exists(pth_path + "_latest.pth"):
        checkpoint = torch.load(pth_path + "_latest.pth")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

    model.train()
    for epoch in range(epoch):
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

        val_loss, val_score = validation(model, criterion, val_loader)

        if scheduler is not None:
            scheduler.step(val_score)

        print(
            f"epoch{epoch+1}: Train_loss:{train_loss_mean} Train_score:{train_score_mean} Val_loss:{val_loss} Val_score:{val_score}"
        )
        writer = SummaryWriter(log_dir)

        writer.add_scalar("Loss/Train_Loss", train_loss_mean, global_step=(epoch + 1))
        writer.add_scalar("Score/Train_Score", train_score_mean, global_step=(epoch + 1))
        writer.add_scalar("Loss/Validation_Loss", val_loss, global_step=(epoch + 1))
        writer.add_scalar("Score/Validation_Score", val_score, global_step=(epoch + 1))
        writer.flush()


        is_best = val_score > previous_best
        previous_best = max(val_score, previous_best)

        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
        }
        torch.save(checkpoint, pth_path + "_latest.pth")
        if is_best:
            torch.save(checkpoint, pth_path + "_best.pth")

    writer.close()
    return

def main_worker(world_size, train_set, val_set, CFG, pth_path, log_dir):

    cudnn.enabled = True
    cudnn.benchmark = True

    model = smp.Unet(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1
    )

    model = model.to(device)
    model = nn.DataParallel(model)

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

    criterion = DiceLoss().to(device)

    batch_size = int(CFG["BATCH_SIZE"] / (world_size))
    num_worker = int(CFG["num_worker"] / (world_size))



    train_loader = DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
    )

    val_loader = DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_worker,
        pin_memory=True,
    )


    train(
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader,
        scheduler,
        CFG,
        pth_path,
        log_dir,
    )

    return
