# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from .detr import build


def build_model(args, **kwargs):
    return build(args, **kwargs)