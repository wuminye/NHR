import torch
from .perceptual_loss import Perceptual_loss


def make_loss(cfg):
    return Perceptual_loss(cfg)
