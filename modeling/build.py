

from .UNet import UNet
from .PCPR import PCPRender
from .PCPR import PCPRParameters
from .generatic_model import Generatic_Model
import logging
import torch
import os
def find_pretrain_file(path, resume_point):
    files = os.listdir(path)
    files = [i for i in files if 'nr_model' in i]
    files.sort()

    if ('nr_model_%d.pth' %resume_point) not in files:
        return None
    return 'nr_model_%d.pth' % resume_point



def build_model(cfg,isTrain= True,dataset_num_overwrite = None):

    if dataset_num_overwrite is None:
        dataset_num_overwrite = len(cfg.DATASETS.TRAIN)

    if isTrain:
        model = Generatic_Model(cfg, cfg.INPUT.SIZE_TRAIN[0], cfg.INPUT.SIZE_TRAIN[1],
                 cfg.MODEL.FEATURE_DIM,  use_dir= cfg.INPUT.USE_DIR, dataset_num = dataset_num_overwrite,use_rgb = cfg.INPUT.USE_RGB,
                 use_mask = cfg.DATASETS.MASK, use_pointnet=not cfg.MODEL.NO_FEATURE_ONLY_RGB, use_depth = cfg.MODEL.USE_DEPTH,use_pc_norm = cfg.MODEL.USE_PC_NORM)
        pretrain_model = find_pretrain_file(cfg.OUTPUT_DIR, cfg.MODEL.RESUME_EPOCH)
        if pretrain_model is not None:
            logger = logging.getLogger("rendering_model.train")
            logger.info("Load pretrain model {}.".format(pretrain_model))
            model.load_state_dict(torch.load(os.path.join(cfg.OUTPUT_DIR,pretrain_model),map_location='cpu'))
    else:
        model = Generatic_Model(cfg, cfg.INPUT.SIZE_TEST[0], cfg.INPUT.SIZE_TEST[1],
                 cfg.MODEL.FEATURE_DIM, use_dir= cfg.INPUT.USE_DIR,dataset_num = dataset_num_overwrite, use_rgb = cfg.INPUT.USE_RGB, 
                 use_mask = cfg.DATASETS.MASK, use_pointnet=not cfg.MODEL.NO_FEATURE_ONLY_RGB, use_depth = cfg.MODEL.USE_DEPTH,use_pc_norm = cfg.MODEL.USE_PC_NORM)

    return model
