import torch
from glob import glob
import datetime
import pytz
from dataset.semi import SemiDataset
import os
from tensorboardX import SummaryWriter
import torch.backends.cudnn as cudnn
from torch.optim.adam import Adam
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.nn.functional as F
import horovod.torch as hvd
from util.utils import AverageMeter
from tqdm import tqdm
import segmentation_models_pytorch as smp


hvd.init()

torch.cuda.set_device(hvd.local_rank())

kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")

log_dir = f"/workspace/FixMatch/log_dir/{day}/"
run_name = f"DeepLab_FixMatch_1"
log_dir = os.path.join(log_dir, run_name)

os.makedirs(log_dir, exist_ok=True)

writer = SummaryWriter(log_dir)

CFG = {
    "IMG_SIZE": 1024,
    "EPOCHS": 3,
    "LEARNING_RATE": 1e-6,
    "BATCH_SIZE": 10,
    "SEED": 41,
    "num_worker": 4,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
    "nclass": 1,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


pth_path = f"/workspace/FixMatch/pthfile/{day}"
pth_name = f"{pth_path}/M:{CFG['train_magnification']}_E:{CFG['EPOCHS']}"

labeled_data_path = f"/workspace/git_ignore/PDA_labeled_tile/train/{CFG['train_magnification']}/**/*.png"
unlabeled_data_path = f"/workspace/git_ignore/PDA_unlabeled_tile/**/*_tiles/*.png"
val_data_path = f"/workspace/git_ignore/PDA_labeled_tile/validation/{CFG['train_magnification']}/**/*.png"


labeled_train_list = sorted(glob(labeled_data_path))
labeled_train_img = labeled_train_list[1::2]
labeled_train_mask = labeled_train_list[0::2]

val_path_list = sorted(glob(val_data_path))
val_img = val_path_list[1::2]
val_mask = val_path_list[0::2]

unlabeled_train_img = (sorted(glob(unlabeled_data_path)))[:5000]


trainset_u = SemiDataset(img_root=unlabeled_train_img, mode="train_u")

trainset_l = SemiDataset(
    img_root=labeled_train_img, mask_root=labeled_train_mask, mode="train_l"
)

valset = SemiDataset(img_root=val_img, mask_root=val_mask, mode="val")

if __name__ == "__main__":
    magnification = CFG["train_magnification"]

    writer.add_scalar("IMG_SIZE", CFG["IMG_SIZE"])
    writer.add_scalar("EPOCHS", CFG["EPOCHS"])
    writer.add_scalar("BATCH_SIZE", CFG["BATCH_SIZE"])
    writer.add_scalar("Magnification", int(CFG["train_magnification"][:-1]))

    world_size = torch.cuda.device_count()

    os.makedirs(os.path.join(pth_path), exist_ok=True)


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


def validation(model, criterion, val_loader):
    loss_meter2 = AverageMeter()
    score_meter2 = AverageMeter()
    model.eval()
    with torch.no_grad():
        for img, label in tqdm(val_loader):
            out = model(img.cuda())
            out = torch.squeeze(out)
            pred = torch.ge(out.sigmoid(), 0.5).float()
            label = torch.squeeze(label).cuda()
            score = dice_score(pred, label)
            loss = criterion(out, label.type(torch.FloatTensor).cuda())

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
    CFG,
    pth_path,
    pth_name,
    writer,
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

        for (img_x, mask_x), (img_u_w, img_u_s, _) in zip(
            tqdm(trainloader_l), trainloader_u
        ):
            mask_x = torch.squeeze(mask_x).cuda()

            model.train()
            out_u_w = model(img_u_w.cuda())
            pred_u_w = torch.squeeze(out_u_w)
            mask_u_w = torch.ge(pred_u_w.sigmoid(), 0.5).float()
            # psuedo label

            out_u_s = model(img_u_s.cuda())
            pred_u_s = torch.squeeze(out_u_s)

            out_x = model(img_x.cuda())
            pred_x = torch.squeeze(out_x)

            loss_x = criterion_x(pred_x, mask_x.type(torch.FloatTensor).cuda())
            loss_u_s = criterion_u(pred_u_s, mask_u_w.type(torch.FloatTensor).cuda())
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

        val_loss, val_score = validation(model, criterion_x, valloader)

        if scheduler is not None:
            scheduler.step(val_score)

        if hvd.rank() == 0:
            print(
                f"epoch{epoch+1}: Train_loss:{train_loss_mean} Val_loss:{val_loss} Val_score:{val_score}"
            )
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

        if hvd.rank() == 0:
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

model = model.cuda()


optimizer = Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
optimizer = hvd.DistributedOptimizer(
    optimizer, named_parameters=model.named_parameters()
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
criterion_x = DiceLoss().cuda()
criterion_u = DiceLoss().cuda()

hvd.broadcast_parameters(model.state_dict(), root_rank=0)
hvd.broadcast_optimizer_state(optimizer, root_rank=0)


if hvd.rank() == 0 or 2:
    batch_size = int(CFG["BATCH_SIZE"] / (world_size + 1))
    num_worker = int(CFG["num_worker"] / (world_size + 1))

elif hvd.rank() == 1:
    batch_size = int(CFG["BATCH_SIZE"] / (world_size + 1)) * 3
    num_worker = int(CFG["num_worker"] / (world_size + 1)) * 3

trainsampler_l = DistributedSampler(
    dataset=trainset_l, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True
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
    dataset=trainset_u, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=True
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
    dataset=valset, num_replicas=hvd.size(), rank=hvd.rank(), shuffle=False
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
    CFG,
    pth_path,
    pth_name,
    writer,
)

writer.close()
