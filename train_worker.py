import torch
import os
from tqdm.auto import tqdm
import segmentation_models_pytorch as smp
from torch.optim.adam import Adam
from torch.utils.data.dataloader import DataLoader
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from collections import OrderedDict
from models.resunet import ResUNet
from util import *


def validation(model, criterion, val_loader, device):
    val_loss_meter = AverageMeter()
    val_score_meter = AverageMeter()
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(val_loader):
            out = model(img.to(device))
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).to(device)
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).to(device))

            val_score_meter.update(score.item())
            val_loss_meter.update(loss.item())

        val_score_mean = val_score_meter.avg
        val_loss_mean = val_loss_meter.avg

        val_score_meter.reset()
        val_loss_meter.reset()
    return val_score_mean, val_loss_mean


def train(
    model,
    criterion,
    optimizer,
    train_loader,
    val_loader,
    scheduler,
    device,
    cfg,
    pth_path,
    log_dir,
    last_epoch,
):
    loss_meter = AverageMeter()
    score_meter = AverageMeter()
    previous_best = 0.0
    epoch = 0

    if os.path.exists(pth_path + "/latest.pth"):
        epoch = last_epoch + 1

    if device == 0:
        early_stopping = EarlyStop(patience=cfg["earlystop_patience"], delta=0.02)

    model.train()
    for epoch in range(epoch, cfg["epochs"]):
        train_loader.sampler.set_epoch(epoch)
        for img, label in tqdm(train_loader):
            optimizer.zero_grad()
            # predict
            out = model(img.to(device))
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).to(device)

            # evaluation
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).to(device))

            loss.backward()
            optimizer.step()

            score_meter.update(score.item())
            loss_meter.update(loss.item())

        train_score_mean = score_meter.avg
        train_loss_mean = loss_meter.avg
        score_meter.reset()
        loss_meter.reset()
        val_score, val_loss = validation(model, criterion, val_loader, device)

        if scheduler is not None:
            scheduler.step(val_score)

        if device == 0:
            print(
                f"epoch{epoch+1}: Train_score:{train_score_mean} Train_loss:{train_loss_mean} Val_score:{val_score} Val_loss:{val_loss}"
            )
            writer = SummaryWriter(log_dir)

            writer.add_scalar(
                "Score/Train_Score", train_score_mean, global_step=(epoch + 1)
            )
            writer.add_scalar(
                "Loss/Train_Loss", train_loss_mean, global_step=(epoch + 1)
            )
            writer.add_scalar(
                "Score/Validation_Score", val_score, global_step=(epoch + 1)
            )
            writer.add_scalar("Loss/Validation_Loss", val_loss, global_step=(epoch + 1))

            writer.flush()

        is_best = val_score > previous_best
        previous_best = max(val_score, previous_best)

        if device == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, pth_path + "/latest.pth")
            if is_best:
                torch.save(checkpoint, pth_path + "/best.pth")

            early_stopping(val_score)
            if early_stopping.early_stop:
                print("Early stopping!")
                break

        torch.distributed.barrier()

    if device == 0:
        writer.close()
    return


def main_worker(gpu, world_size, train_set, val_set, cfg, pth_path, log_dir):
    dist.init_process_group(
        backend="nccl",
        init_method=cfg["port"],
        world_size=world_size,
        rank=gpu,
    )
    torch.cuda.set_device(gpu)

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

    optimizer = Adam(params=model.parameters(), lr=cfg["lr"])
    last_epoch = 0
    if os.path.exists(pth_path + "/latest.pth"):
        checkpoint = torch.load(pth_path + "/latest.pth")
        model_state_dict = OrderedDict()

        for n, v in checkpoint["model"].items():
            model_name = n.replace("module.", "")
            model_state_dict[model_name] = v

        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(checkpoint["optimizer"])
        last_epoch = checkpoint["epoch"]

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(gpu)

    model = model.to(gpu)
    model = DistributedDataParallel(
        model, device_ids=[gpu], find_unused_parameters=True
    )

    batch_size = int(cfg["batch_size"] / world_size)
    num_worker = int(cfg["num_worker"] / world_size)

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
        cfg,
        pth_path,
        log_dir,
        last_epoch,
    )

    return
