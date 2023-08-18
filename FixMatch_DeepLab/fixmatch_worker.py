import os
import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from util.utils import AverageMeter
import torch.distributed as dist
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from torch.utils.tensorboard import SummaryWriter


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
    log_dir,
):
    previous_best = 0.0
    total_loss = AverageMeter()
    total_loss_x = AverageMeter()
    total_loss_s = AverageMeter()

    if os.path.exists(pth_path + "_latest.pth"):
        checkpoint = torch.load(pth_path + "_latest.pth")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

    for epoch in range(CFG["EPOCHS"]):
        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        for (img_x, mask_x), (img_u_w, img_u_s, _) in zip(
            tqdm(trainloader_l), trainloader_u
        ):
            mask_x = torch.squeeze(mask_x).to(gpu)

            model.train()
            out_u_w = model(img_u_w.to(gpu))
            pred_u_w = torch.squeeze(out_u_w)
            mask_u_w = torch.ge(pred_u_w.sigmoid(), 0.6).float()
            # psuedo label

            out_u_s = model(img_u_s.to(gpu))
            pred_u_s = torch.squeeze(out_u_s)

            out_x = model(img_x.to(gpu))
            pred_x = torch.squeeze(out_x)

            loss_x = criterion_x(pred_x, mask_x.type(torch.FloatTensor).to(gpu))
            loss_u_s = criterion_u(pred_u_s, mask_u_w.type(torch.FloatTensor).to(gpu))
            loss = (loss_x + loss_u_s) / 2.0

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

        torch.distributed.barrier()

        val_loss, val_score = validation(model, criterion_x, valloader, gpu)

        if scheduler is not None:
            scheduler.step(val_score)

        if gpu == 0:
            print(
                f"epoch{epoch+1}: Train_loss:{train_loss_mean} Val_loss:{val_loss} Val_score:{val_score}"
            )
            writer = SummaryWriter(log_dir)

            writer.add_scalar(
                "Loss/Train_Loss", train_loss_mean, global_step=(epoch + 1)
            )
            writer.add_scalar("Loss/Validation_Loss", val_loss, global_step=(epoch + 1))
            writer.add_scalar(
                "Score/Validation_Score", val_score, global_step=(epoch + 1)
            )
            writer.flush()

        is_best = val_score > previous_best
        previous_best = max(val_score, previous_best)

        if gpu == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, pth_path + "_latest.pth")
            if is_best:
                torch.save(checkpoint, pth_path + "_best.pth")

        torch.distributed.barrier()

    if gpu == 0:
        writer.close()
    return


def main_worker(
    gpu,
    world_size,
    trainset_u,
    trainset_l,
    valset,
    CFG,
    pth_path,
    log_dir,
):
    dist.init_process_group(
        backend="nccl",
        init_method="tcp://0.0.0.0:12345",
        world_size=world_size,
        rank=gpu,
    )

    cudnn.enabled = True
    cudnn.benchmark = True

    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=1,  # model output channels (number of classes in your dataset)
        activation=None,
        decoder_atrous_rates=(6, 12, 18),
    )

    model = model.to(gpu)

    model = DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=False, broadcast_buffers=False
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
        batch_size = int(CFG["BATCH_SIZE"] / (world_size + 1)) * 2
        num_worker = int(CFG["num_worker"] / (world_size + 1)) * 2

    trainsampler_l = DistributedSampler(
        dataset=trainset_l, num_replicas=world_size, shuffle=True
    )
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_worker,
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
        log_dir,
    )

    return
