# encoding: utf-8


import torch
import logging
import os

def find_pretrain_file(path, resume_point):
    files = os.listdir(path)
    files = [i for i in files if 'nr_optimizer' in i]
    files.sort()

    if ('nr_optimizer_%d.pth' %resume_point) not in files:
        return None
    return 'nr_optimizer_%d.pth' % resume_point


def make_optimizer(cfg, model, pre_load = True):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg.SOLVER.BASE_LR
        weight_decay = cfg.SOLVER.WEIGHT_DECAY
        if "bias" in key:
            lr = cfg.SOLVER.BASE_LR * cfg.SOLVER.BIAS_LR_FACTOR
            weight_decay = cfg.SOLVER.WEIGHT_DECAY_BIAS
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
        
    if cfg.SOLVER.OPTIMIZER_NAME == 'SGD':
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params, momentum=cfg.SOLVER.MOMENTUM)
    else:
        optimizer = getattr(torch.optim, cfg.SOLVER.OPTIMIZER_NAME)(params)

    if pre_load:
        pretrain_optimizer = find_pretrain_file(cfg.OUTPUT_DIR, cfg.MODEL.RESUME_EPOCH)
        if pretrain_optimizer is not None:
            logger = logging.getLogger("rendering_model.train")
            logger.info("Load pretrain optimizer {}.".format(pretrain_optimizer))
            optimizer.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR,pretrain_optimizer),map_location='cpu'))

    return optimizer
