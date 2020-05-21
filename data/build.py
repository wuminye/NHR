# encoding: utf-8

import os
import torch
from torch.utils import data

import logging


from .datasets import IBRDynamicDataset
from .datasets import IBRDynamicTotalDataset
from .transforms import build_transforms
from .collate_batch import static_collate

def build_dataset(cfg, transforms,  is_train=True, use_mask = False, is_need_all_data = False):

    if is_train:
        datasets = IBRDynamicTotalDataset(cfg, transforms,is_train)
        if cfg.DATASETS.RANDOM_NOISY>0:
            if cfg.MODEL.RESUME_EPOCH > 0 :
                datasets.load_vs(cfg)
            else:
                datasets.save_vs(cfg)
    else:
        datasets = IBRDynamicTotalDataset(cfg, transforms,is_train)
        if cfg.DATASETS.RANDOM_NOISY>0:
            datasets.load_vs(cfg)


    logger = logging.getLogger("rendering_model.train")
    logger.info("Load {} datasets.".format(len(datasets.datasets)))
    return datasets


def make_data_loader(cfg, is_train=True, is_need_all_data = False, is_center =False):
    
    batch_size = cfg.SOLVER.IMS_PER_BATCH


    transforms = build_transforms(cfg, is_train,is_center = is_center)
    datasets = build_dataset(cfg, transforms,  is_train, is_need_all_data)

    num_workers = cfg.DATALOADER.NUM_WORKERS
    if is_train:
        data_loader = data.DataLoader(
            datasets, collate_fn=static_collate, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
    else:
        data_loader = data.DataLoader(
            datasets, collate_fn=static_collate, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )

    return data_loader, datasets
