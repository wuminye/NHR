#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
import os
import sys
from os import mkdir
from apex import amp
import shutil


# In[ ]:


import torch.nn.functional as F

sys.path.append('..')
from config import cfg
from data import make_data_loader
from engine.trainer import do_train
from modeling import build_model
from solver import make_optimizer, WarmupMultiStepLR, build_scheduler 
from layers import make_loss

from utils.logger import setup_logger

from torch.utils.tensorboard import SummaryWriter
import torch
torch.cuda.set_device(int(sys.argv[1]))


# In[ ]:
 

cfg.merge_from_file('../configs/config.yml')
cfg.freeze()


# In[ ]:


output_dir = cfg.OUTPUT_DIR

writer = SummaryWriter(log_dir=os.path.join(output_dir,'tensorboard'))

logger = setup_logger("rendering_model", output_dir, 0)
logger.info("Running with config:\n{}".format(cfg))

shutil.copy('../configs/config.yml', os.path.join(cfg.OUTPUT_DIR,'configs.yml'))


# In[ ]:


train_loader, dataset = make_data_loader(cfg, is_train=True,is_center = cfg.DATASETS.CENTER)
model = build_model(cfg)


# In[ ]:


optimizer = make_optimizer(cfg, model)

sscheduler = WarmupMultiStepLR(optimizer, cfg.SOLVER.STEPS, cfg.SOLVER.GAMMA, cfg.SOLVER.WARMUP_FACTOR,
                               cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.WARMUP_METHOD)
#scheduler = build_scheduler(optimizer, cfg.SOLVER.WARMUP_ITERS, cfg.SOLVER.START_ITERS, cfg.SOLVER.END_ITERS, cfg.SOLVER.LR_SCALE)



loss_fn = make_loss(cfg)





model, optimizer = amp.initialize(model, optimizer, opt_level="O1")





do_train(
        cfg,
        model,
        train_loader,
        None,
        optimizer,
        scheduler,
        loss_fn,
        writer
    )






