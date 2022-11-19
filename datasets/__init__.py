# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch.utils.data
import torchvision

from .coco import build as build_coco
from .custom import build as build_custom


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


def build_dataset(image_set, config):
    if config.dataset_type == "coco":
        return build_coco(image_set, config)
    if config.dataset_type == "coco_panoptic":
        # to avoid making panopticapi required for coco
        from .coco_panoptic import build as build_coco_panoptic

        return build_coco_panoptic(image_set, config)
    if config.dataset_type == "custom":
        return build_custom(image_set, config)
    raise ValueError(f"dataset {config.dataset_type} not supported")
