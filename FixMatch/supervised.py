import os
import torch
import numpy as np
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
from glob import glob
import mlflow

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import AverageMeter, intersectionAndUnion
from util.dist_helper import setup_distributed

CFG = {
    "IMG_SIZE": 1024,
    "crop_size": 256,
    "EPOCHS": 50,
    "lr": 0.004,
    "lr_multi": 10.0,
    "BATCH_SIZE": 10,
    "SEED": 41,
    "num_worker": 4,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
    "dataset": "pathology",
    "nclass": 2,
    "criterion": "CELoss",
    "ignore_index": 255,
    "conf_thresh": 0.95,
    "backbone": "xception",
    "dilations": [6, 12, 18],
}

labeled_data_path = f"/workspace/git_ignore/PDA_labeled_tile(1024)/train/{CFG['train_magnification']}/**/*.png"
unlabeled_data_path = f"/workspace/git_ignore/PDA_unlabeled_tile(1024)/**/*_tiles/*.png"
val_data_path = f"/workspace/git_ignore/PDA_labeled_tile(1024)/validation/{CFG['train_magnification']}/**/*.png"
pth_path = "/workspace/FixMatch/pthfile"

labeled_train_list = sorted(glob(labeled_data_path))
labeld_train_img = labeled_train_list[1::2]
labeld_train_mask = labeled_train_list[0::2]

val_path_list = sorted(glob(val_data_path))
val_img = val_path_list[1::2]
val_mask = val_path_list[0::2]

unlabeled_train_img = sorted(glob(unlabeled_data_path))
unlabeled_train_mask = None

device = "cuda" if torch.cuda.is_available() else "cpu"


def evaluate(model, loader, mode, cfg):
    model.eval()
    assert mode in ["original", "center_crop", "sliding_window"]
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    with torch.no_grad():
        for img, mask in tqdm(loader):
            img = img.cuda()

            if mode == "sliding_window":
                grid = cfg["crop_size"]
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(
                            img[
                                :, :, row : min(h, row + grid), col : min(w, col + grid)
                            ]
                        )
                        final[
                            :, :, row : min(h, row + grid), col : min(w, col + grid)
                        ] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == "center_crop":
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg["crop_size"]) // 2, (
                        w - cfg["crop_size"]
                    ) // 2
                    img = img[
                        :,
                        :,
                        start_h : start_h + cfg["crop_size"],
                        start_w : start_w + cfg["crop_size"],
                    ]
                    mask = mask[
                        :,
                        start_h : start_h + cfg["crop_size"],
                        start_w : start_w + cfg["crop_size"],
                    ]

                pred = model(img).argmax(dim=1)

            intersection, union, target = intersectionAndUnion(
                pred.cpu().numpy(), mask.numpy(), cfg["nclass"], 255
            )

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            dist.all_reduce(reduced_intersection)
            dist.all_reduce(reduced_union)
            dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    return mIOU


def main():
    rank, world_size = setup_distributed(port="tcp://0.0.0.0:12345")

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(CFG)

    optimizer = SGD(
        [
            {"params": model.backbone.parameters(), "lr": CFG["lr"]},
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "backbone" not in name
                ],
                "lr": CFG["lr"] * CFG["lr_multi"],
            },
        ],
        lr=CFG["lr"],
        momentum=0.9,
        weight_decay=1e-4,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )

    if CFG["criterion"] == "CELoss":
        criterion = nn.CrossEntropyLoss(CFG["ignore_index"]).cuda(local_rank)
    elif CFG["criterion"] == "OHEM":
        criterion = ProbOhemCrossEntropy2d(CFG["ignore_index"]).cuda(local_rank)

    trainset = SemiDataset(
        img=labeld_train_img,
        mask=labeld_train_mask,
        mode="train_l",
        size=CFG["crop_size"],
    )
    valset = SemiDataset(img=val_img, mask=val_mask, mode="val", size=CFG["crop_size"])

    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset,
        batch_size=CFG["batch_size"],
        pin_memory=True,
        num_workers=1,
        drop_last=True,
        sampler=trainsampler,
    )
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=1,
        pin_memory=True,
        num_workers=1,
        drop_last=False,
        sampler=valsampler,
    )

    iters = 0
    total_iters = len(trainloader) * CFG["EPOCHS"]
    previous_best = 0.0

    if os.path.exists(os.path.join(pth_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(pth_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"], map_location=device)
        optimizer.load_state_dict(checkpoint["optimizer"], map_location=device)
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

        if rank == 0:
            print(f"************ Load from checkpoint at epoch {epoch}")

    for epoch in range(1, CFG["EPOCHS"] + 1):
        if rank == 0:
            print(
                f'===========> Epoch: {epoch}, LR: {optimizer.param_groups[0]["lr"]:.5f}, Previous best: {previous_best:.2f}'
            )
        model.train()
        total_loss = AverageMeter()

        trainsampler.set_epoch(epoch)

        for i, (img, mask) in enumerate(trainloader):
            img, mask = img.cuda(), mask.cuda()

            pred = model(img)

            loss = criterion(pred, mask)

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())

            iters = epoch * len(trainloader) + i
            lr = CFG["lr"] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * CFG["lr_multi"]

        eval_mode = "sliding_window" if CFG["dataset"] == "cityscapes" else "original"
        mIoU = evaluate(model, valloader, eval_mode, CFG)

        if rank == 0:
            print(f"epoch{epoch}: Total Loss:{total_loss.avg:.2f} Val mIOU:{mIoU:.2f}")
            with mlflow.start_run(
                run_name=CFG["train_magnification"], experiment_id=459646067973468985
            ):
                mlflow.log_metric("Total_Loss", total_loss.avg)
                mlflow.log_metric("Val mIOU", mIoU)
                mlflow.end_run()

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(pth_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(pth_path, "best.pth"))


if __name__ == "__main__":
    main()
