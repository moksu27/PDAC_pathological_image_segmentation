import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
from glob import glob
import datetime
import pytz
import mlflow

from dataset.semi import SemiDataset
from model.semseg.deeplabv3plus import DeepLabV3Plus
from supervised import evaluate
from util.classes import CLASSES
from util.utils import AverageMeter
from util.dist_helper import setup_distributed

CFG = {
    "IMG_SIZE": 1024,
    "crop_size": 256,
    "EPOCHS": 3,
    "LEARNING_RATE": 1e-5,
    "lr_multi": 10.0,
    "BATCH_SIZE": 5,
    "SEED": 41,
    "num_worker": 12,
    "MEAN": [0.485, 0.456, 0.406],
    "STD": [0.229, 0.224, 0.225],
    "train_magnification": "20X",
    "backbone": "xception",
    "dilations": [6, 12, 18],
    "nclass": 2,
}

device = "cuda" if torch.cuda.is_available() else "cpu"

# 경로 및 변수 지정
kst = pytz.timezone("Asia/Seoul")
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")

pth_path = f"/workspace/FixMatch_new/pthfile/{day}"
pth_name = f"{pth_path}/M:{CFG['train_magnification']}_E:{CFG['EPOCHS']}_{hour}.pth"

labeled_data_path = f"/workspace/git_ignore/PDA_labeled_tile/train/{CFG['train_magnification']}/**/*.png"
unlabeled_data_path = f"/workspace/git_ignore/PDA_unlabeled_tile/**/*_tiles/*.png"
val_data_path = f"/workspace/git_ignore/PDA_labeled_tile/validation/{CFG['train_magnification']}/**/*.png"


labeled_train_list = sorted(glob(labeled_data_path))[0:10]
labeled_train_img = labeled_train_list[1::2]
labeled_train_mask = labeled_train_list[0::2]

val_path_list = sorted(glob(val_data_path))[0:10]
val_img = val_path_list[1::2]
val_mask = val_path_list[0::2]

unlabeled_train_img = sorted(glob(unlabeled_data_path))[0:10]
unlabeled_train_mask = None


def main():
    os.makedirs(f"{pth_path}", exist_ok=True)

    rank, world_size = setup_distributed(port="tcp://0.0.0.0:12345")

    cudnn.enabled = True
    cudnn.benchmark = True

    model = DeepLabV3Plus(CFG)

    optimizer = SGD(
        [
            {"params": model.backbone.parameters(), "lr": CFG["LEARNING_RATE"]},
            {
                "params": [
                    param
                    for name, param in model.named_parameters()
                    if "backbone" not in name
                ],
                "lr": CFG["LEARNING_RATE"] * CFG["lr_multi"],
            },
        ],
        lr=CFG["LEARNING_RATE"],
        momentum=0.9,
        weight_decay=1e-4,
    )

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        broadcast_buffers=False,
        output_device=local_rank,
        find_unused_parameters=False,
    )

    criterion_l = nn.BCEWithLogitsLoss().cuda(local_rank)

    criterion_u = nn.BCEWithLogitsLoss(reduction="none").cuda(local_rank)

    trainset_u = SemiDataset(
        unlabeled_train_img, unlabeled_train_mask, "train_u", CFG["crop_size"]
    )
    trainset_l = SemiDataset(
        labeled_train_img, labeled_train_mask, "train_l", CFG["crop_size"]
    )
    valset = SemiDataset(val_img, val_mask, "val")

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=CFG["BATCH_SIZE"],
        pin_memory=True,
        num_workers=CFG["num_worker"],
        drop_last=True,
        sampler=trainsampler_l,
    )
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=CFG["BATCH_SIZE"],
        pin_memory=True,
        num_workers=CFG["num_worker"],
        drop_last=True,
        sampler=trainsampler_u,
    )
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset,
        batch_size=CFG["BATCH_SIZE"],
        pin_memory=True,
        num_workers=CFG["num_worker"],
        drop_last=False,
        sampler=valsampler,
    )

    previous_best = 0.0
    epoch = -1

    if os.path.exists(os.path.join(pth_path, "latest.pth")):
        checkpoint = torch.load(os.path.join(pth_path, "latest.pth"))
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        epoch = checkpoint["epoch"]
        previous_best = checkpoint["previous_best"]

    mlflow.set_tracking_uri("http://127.0.0.1:5000")  # MLflow 서버 주소 설정
    with mlflow.start_run(
        run_name=f"{magnification}_{day}_Fixmatch", experiment_id=210481695216345952
    ) as run:
        run_id = run.info.run_id
        mlflow.log_param("IMG_SIZE", CFG["IMG_SIZE"])
        mlflow.log_param("EPOCHS", CFG["EPOCHS"])
        mlflow.log_param("BATCH_SIZE", CFG["BATCH_SIZE"])
        mlflow.log_param("Magnification", CFG["train_magnification"])
        mlflow.end_run()

    for epoch in range(epoch + 1, CFG["EPOCHS"] + 1):
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)

        loader = zip(trainloader_l, trainloader_u, trainloader_u)

        for i, (
            (img_x, mask_x),
            (img_u_w, img_u_s, _, cutmix_box, _),
            (img_u_w_mix, img_u_s_mix, _, _, _),
        ) in enumerate(loader):
            img_x = img_x.cuda()
            mask_x = torch.squeeze(mask_x).cuda()

            img_u_w, img_u_s = img_u_w.cuda(), img_u_s.cuda()
            cutmix_box = cutmix_box.cuda()
            img_u_w_mix, img_u_s_mix = img_u_w_mix.cuda(), img_u_s_mix.cuda()

            with torch.no_grad():
                model.eval()
                out_u_w_mix = model(img_u_w_mix)
                conf_u_w_mix = torch.squeeze(out_u_w_mix)
                pred_u_w_mix = torch.ge(conf_u_w_mix.sigmoid(), 0.5).float()
                mask_u_w_mix = torch.squeeze(pred_u_w_mix).cuda()
                img_u_s[
                    cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1
                ] = img_u_s_mix[cutmix_box.unsqueeze(1).expand(img_u_s.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            out_x, out_u_w = model(torch.cat((img_x, img_u_w))).split([num_lb, num_ulb])

            conf_u_w = torch.squeeze(out_u_w)
            pred_u_w = torch.ge(conf_u_w.sigmoid(), 0.5).float()
            mask_u_w = torch.squeeze(pred_u_w).cuda()

            out_u_s = model(img_u_s)

            mask_u_w_cutmixed, conf_u_w_cutmixed = (mask_u_w.clone(), conf_u_w.clone())

            mask_u_w_cutmixed[cutmix_box == 1] = mask_u_w_mix[
                cutmix_box == 1
            ]  # psuedo label

            conf_u_w_cutmixed[cutmix_box == 1] = conf_u_w_mix[cutmix_box == 1]

            loss_x = criterion_l(out_x, mask_x.type(torch.FloatTensor))

            loss_u_s = criterion_u(out_u_s, mask_u_w_cutmixed.type(torch.FloatTensor))

            loss = (loss_x + loss_u_s) / 2.0

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
        # ______________________________________evaluate Dice Score로 바꾸기________________________________________
        mIoU, iou_class = evaluate(model, valloader, "original", CFG)

        if rank == 0:
            for cls_idx, iou in enumerate(iou_class):
                logger.info(
                    "***** Evaluation ***** >>>> Class [{:} {:}] "
                    "IoU: {:.2f}".format(cls_idx, CLASSES[cfg["dataset"]][cls_idx], iou)
                )
            logger.info(
                "***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n".format(
                    eval_mode, mIoU
                )
            )

            writer.add_scalar("eval/mIoU", mIoU, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar(
                    "eval/%s_IoU" % (CLASSES[cfg["dataset"]][i]), iou, epoch
                )

        is_best = mIoU > previous_best
        previous_best = max(mIoU, previous_best)
        if rank == 0:
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "previous_best": previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, "latest.pth"))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, "best.pth"))
        """
        train_loss_mean = total_loss.avg
    
        if rank == 0:
            magnification = CFG["train_magnification"]
            print(f"epoch{epoch+1}: Train_loss:{train_loss_mean} Train_score:{train_score_mean} Val_loss:{val_loss} Val_score:{val_score}")
            with mlflow.start_run(run_name=CFG["train_magnification"], run_id=run_id, experiment_id=210481695216345952):
                mlflow.log_metric("Train_Loss", total_loss.avg, step=epoch + 1)
                mlflow.log_metric("Train_Score", , step=epoch + 1)
                mlflow.log_metric("Validation_Loss", loss_u_s.item(), step=epoch + 1)
                mlflow.log_metric("Validation_Score", val_score, step=epoch + 1)
                mlflow.end_run()
        """


if __name__ == "__main__":
    main()
