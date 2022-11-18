# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(config, args, dataset_file, train_backbone):
    return build(config, args, dataset_file, train_backbone)
