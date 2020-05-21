from yacs.config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.MODEL = CN()
_C.MODEL.DEVICE = "cuda"
_C.MODEL.NUM_CLASSES = 10
_C.MODEL.FEATURE_DIM = 20
_C.MODEL.RESUME_EPOCH = -1
_C.MODEL.NO_FEATURE_ONLY_RGB = False
_C.MODEL.LOSS = "MSE"
_C.MODEL.USE_DEPTH = False
_C.MODEL.USE_PC_NORM = False
_C.MODEL.MUL_POINTNET = False
_C.MODEL.UNET_LAYERS = [32, 64,128,512,512, 256,64,64,42]
_C.MODEL.POINTNET_NPOINTS = [4096, 1024, 256, 64]
_C.MODEL.POINTNET_RADIUS = [[0.1, 0.5], [0.5, 1.0], [1.0, 2.0], [2.0, 4.0]]
_C.MODEL.NO_MODIFY = False
_C.MODEL.STATIC_MODE = False

# -----------------------------------------------------------------------------
# INPUT
# -----------------------------------------------------------------------------
_C.INPUT = CN()
# Size of the image during training
_C.INPUT.SIZE_TRAIN = [400,250]
# Size of the image during test
_C.INPUT.SIZE_TEST = [400,250]
# Minimum scale for the image during training
_C.INPUT.MIN_SCALE_TRAIN = 0.5
# Maximum scale for the image during test
_C.INPUT.MAX_SCALE_TRAIN = 1.2
# Random probability for image horizontal flip
_C.INPUT.PROB = 0.5
# Values to be used for image normalization
_C.INPUT.PIXEL_MEAN = [0.485, 0.456, 0.406]
# Values to be used for image normalization
_C.INPUT.PIXEL_STD = [0.229, 0.224, 0.225]
_C.INPUT.NEAR_FAR_SIZE = [[1000, 3600, 1.5]]
_C.INPUT.USE_RGB = True
_C.INPUT.RGB_MAP = False

_C.INPUT.USE_DIR = "MAPS"

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
# List of the dataset names for training, as present in paths_catalog.py
_C.DATASETS.TRAIN = ["./"]
# List of the dataset names for testing, as present in paths_catalog.py
_C.DATASETS.TEST = ()
_C.DATASETS.FRAME_NUM = [20]
_C.DATASETS.MASK = False
_C.DATASETS.SHIFT = 0
_C.DATASETS.MAXRATION = 0.0
_C.DATASETS.ROTATION = 0.0
_C.DATASETS.SKIP_STEP = [1]
_C.DATASETS.RANDOM_NOISY = 0.0
_C.DATASETS.CENTER = False
_C.DATASETS.HOLES = ["None"]
_C.DATASETS.IGNORE_FRAMES = []

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of data loading threads
_C.DATALOADER.NUM_WORKERS = 10

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()
_C.SOLVER.OPTIMIZER_NAME = "SGD"

_C.SOLVER.MAX_EPOCHS = 100

_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.BIAS_LR_FACTOR = 2

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.WEIGHT_DECAY_BIAS = 0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 3
_C.SOLVER.WARMUP_ITERS = 500
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 10
_C.SOLVER.LOG_PERIOD = 50

_C.SOLVER.START_ITERS=50
_C.SOLVER.END_ITERS=200
_C.SOLVER.LR_SCALE=0.1
_C.SOLVER.LOSS_WHOLE_IMAGE = False

# Number of images per batch
# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.SOLVER.IMS_PER_BATCH = 7

# This is global, so if we have 8 GPUs and IMS_PER_BATCH = 16, each GPU will
# see 2 images per batch
_C.TEST = CN()
_C.TEST.IMS_PER_BATCH = 8
_C.TEST.WEIGHT = ""

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = ""
