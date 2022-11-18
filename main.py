# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import datetime
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict
from torch.utils.data import DataLoader, DistributedSampler

import datasets
import util.misc as utils
from datasets import build_dataset, get_coco_api_from_dataset
from engine import evaluate, train_one_epoch
from models import build_model


def get_args_parser():
    parser = argparse.ArgumentParser("Set transformer detector", add_help=False)
    parser.add_argument("--config", default="./config.yaml", type=str)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument(
        "--device", default="cuda", help="device to use for training / testing"
    )
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    return parser


def main(args):
    utils.init_distributed_mode(args)
    print("git:\n  {}\n".format(utils.get_sha()))

    with open(args.config) as f:
        config = EasyDict(yaml.load(f, Loader=yaml.FullLoader))

    if config.saver.output_dir:
        Path(config.saver.output_dir).mkdir(parents=True, exist_ok=True)

    if config.saver.frozen_weights is not None:
        assert config.model.head.masks, "Frozen training is meant for segmentation only"
    print(args)
    print(config)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = config.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    train_backbone = config.trainer.lr_backbone > 0
    model, criterion, postprocessors = build_model(
        config.model, args, config.data.dataset_type, train_backbone
    )
    model.to(device)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)

    param_dicts = [
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" not in n and p.requires_grad
            ]
        },
        {
            "params": [
                p
                for n, p in model_without_ddp.named_parameters()
                if "backbone" in n and p.requires_grad
            ],
            "lr": config.trainer.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(
        param_dicts, lr=config.trainer.lr, weight_decay=config.trainer.weight_decay
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config.trainer.lr_drop)

    dataset_train = build_dataset(image_set="train", config=config.data)
    dataset_val = build_dataset(image_set="val", config=config.data)

    if args.distributed:
        sampler_train = DistributedSampler(dataset_train)
        sampler_val = DistributedSampler(dataset_val, shuffle=False)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, config.trainer.batch_size, drop_last=True
    )

    data_loader_train = DataLoader(
        dataset_train,
        batch_sampler=batch_sampler_train,
        collate_fn=utils.collate_fn,
        num_workers=config.trainer.num_workers,
    )
    data_loader_val = DataLoader(
        dataset_val,
        config.trainer.batch_size,
        sampler=sampler_val,
        drop_last=False,
        collate_fn=utils.collate_fn,
        num_workers=config.trainer.num_workers,
    )

    if config.data.dataset_type == "coco_panoptic":
        # We also evaluate AP during panoptic training, on original coco DS
        coco_val = datasets.coco.build("val", args)
        base_ds = get_coco_api_from_dataset(coco_val)
    else:
        base_ds = get_coco_api_from_dataset(dataset_val)

    if config.saver.frozen_weights is not None:
        checkpoint = torch.load(config.saver.frozen_weights, map_location="cpu")
        model_without_ddp.detr.load_state_dict(checkpoint["model"])

    output_dir = Path(config.saver.output_dir)

    if config.saver.load_pretrain:
        utils.load_pretrain(config.saver.load_pretrain, model_without_ddp)

    if config.saver.resume:
        if config.saver.resume.startswith("https"):
            checkpoint = torch.hub.load_state_dict_from_url(
                config.saver.resume, map_location="cpu", check_hash=True
            )
        else:
            checkpoint = torch.load(config.saver.resume, map_location="cpu")
        model_without_ddp.load_state_dict(checkpoint["model"])
        if (
            not args.eval
            and "optimizer" in checkpoint
            and "lr_scheduler" in checkpoint
            and "epoch" in checkpoint
        ):
            optimizer.load_state_dict(checkpoint["optimizer"])
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
            config.trainer.start_epoch = checkpoint["epoch"] + 1

    if args.eval:
        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            config.saver.output_dir,
        )
        if config.saver.output_dir:
            utils.save_on_master(
                coco_evaluator.coco_eval["bbox"].eval, output_dir / "eval.pth"
            )
        return

    print("Start training")
    start_time = time.time()
    for epoch in range(config.trainer.start_epoch, config.trainer.epochs):
        if args.distributed:
            sampler_train.set_epoch(epoch)
        train_stats = train_one_epoch(
            model,
            criterion,
            data_loader_train,
            optimizer,
            device,
            epoch,
            config.trainer.clip_max_norm,
        )
        lr_scheduler.step()
        if config.saver.output_dir:
            checkpoint_paths = [output_dir / "checkpoint.pth"]
            # extra checkpoint before LR drop and every 100 epochs
            if (epoch + 1) % config.trainer.lr_drop == 0 or (epoch + 1) % 100 == 0:
                checkpoint_paths.append(output_dir / f"checkpoint{epoch:04}.pth")
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master(
                    {
                        "model": model_without_ddp.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "lr_scheduler": lr_scheduler.state_dict(),
                        "epoch": epoch,
                        "args": args,
                    },
                    checkpoint_path,
                )

        test_stats, coco_evaluator = evaluate(
            model,
            criterion,
            postprocessors,
            data_loader_val,
            base_ds,
            device,
            config.saver.output_dir,
        )

        log_stats = {
            **{f"train_{k}": v for k, v in train_stats.items()},
            **{f"test_{k}": v for k, v in test_stats.items()},
            "epoch": epoch,
            "n_parameters": n_parameters,
        }

        if config.saver.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

            # for evaluation logs
            if coco_evaluator is not None:
                (output_dir / "eval").mkdir(exist_ok=True)
                if "bbox" in coco_evaluator.coco_eval:
                    filenames = ["latest.pth"]
                    if epoch % 50 == 0:
                        filenames.append(f"{epoch:03}.pth")
                    for name in filenames:
                        torch.save(
                            coco_evaluator.coco_eval["bbox"].eval,
                            output_dir / "eval" / name,
                        )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        "DETR training and evaluation script", parents=[get_args_parser()]
    )
    args = parser.parse_args()
    main(args)
