import yaml
import numpy as np
from easydict import EasyDict as edict

config = edict()

config.output_path = ''
config.symbol = ''
config.gpus = ''
config.CLASS_AGNOSTIC = True
config.SCALES = [(600, 1000)]  # first is scale (the shorter side); second is max size
config.TEST_SCALES = [(600, 1000)]
# default training
config.default = edict()

# network related params
config.network = edict()
config.network.pretrained = ''
config.network.pretrained_epoch = 0
config.network.PIXEL_MEANS = np.array([103.06, 115.90, 123.15])
config.network.IMAGE_STRIDE = 32
config.network.RPN_FEAT_STRIDE = [4, 8, 16, 32, 64]
config.network.RCNN_FEAT_STRIDE = 16
config.network.FIXED_PARAMS = ['conv1', 'bn_conv1', 'res2', 'bn2', 'gamma', 'beta']
config.network.FIXED_PARAMS_SHARED = ['conv1', 'bn_conv1', 'res2', 'bn2', 'res3', 'bn3', 'res4', 'bn4', 'gamma', 'beta']
config.network.ANCHOR_SCALES = [8]
config.network.ANCHOR_RATIOS = [0.5, 1, 2]
config.network.NUM_ANCHORS = len(config.network.ANCHOR_SCALES) * len(config.network.ANCHOR_RATIOS)

# dataset related params
config.dataset = edict()
config.dataset.dataset = 'PascalVOC'
config.dataset.image_set = '2007_trainval'
config.dataset.test_image_set = '2007_test'
config.dataset.root_path = './data'
config.dataset.NUM_CLASSES = 21


config.TRAIN = edict()

config.TRAIN.lr = 0
config.TRAIN.lr_step = ''
config.TRAIN.lr_factor = 0.1
config.TRAIN.warmup = False
config.TRAIN.warmup_lr = 0
config.TRAIN.warmup_step = 0
config.TRAIN.momentum = 0.9
config.TRAIN.wd = 0.0005
config.TRAIN.begin_epoch = 0
config.TRAIN.end_epoch = 0
config.TRAIN.model_prefix = 'model'
config.TRAIN.nms_loss_scale = 1.0
config.TRAIN.nms_pos_scale = 4.0

config.TRAIN.ALTERNATE = edict()
config.TRAIN.ALTERNATE.RPN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.RCNN_BATCH_IMAGES = 0
config.TRAIN.ALTERNATE.rpn1_lr = 0
config.TRAIN.ALTERNATE.rpn1_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn1_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn1_lr = 0
config.TRAIN.ALTERNATE.rfcn1_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn1_epoch = 0      # recommend 8
config.TRAIN.ALTERNATE.rpn2_lr = 0
config.TRAIN.ALTERNATE.rpn2_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn2_epoch = 0       # recommend 3
config.TRAIN.ALTERNATE.rfcn2_lr = 0
config.TRAIN.ALTERNATE.rfcn2_lr_step = ''   # recommend '5'
config.TRAIN.ALTERNATE.rfcn2_epoch = 0      # recommend 8
# optional
config.TRAIN.ALTERNATE.rpn3_lr = 0
config.TRAIN.ALTERNATE.rpn3_lr_step = ''    # recommend '2'
config.TRAIN.ALTERNATE.rpn3_epoch = 0       # recommend 3

# whether resume training
config.TRAIN.RESUME = False
config.TRAIN.FLIP = True
config.TRAIN.SHUFFLE = True
config.TRAIN.BATCH_IMAGES = 1
# group images with similar aspect ratio
config.TRAIN.ASPECT_GROUPING = True

# R-CNN
# rcnn rois batch size
config.TRAIN.BATCH_ROIS = -1
config.TRAIN.BATCH_ROIS_OHEM = 512
# rcnn rois sampling params
config.TRAIN.FG_FRACTION = 0.25
config.TRAIN.FG_THRESH = 0.5
config.TRAIN.BG_THRESH_HI = 0.5
config.TRAIN.BG_THRESH_LO = 0.0
# rcnn bounding box regression params
config.TRAIN.BBOX_REGRESSION_THRESH = 0.5
config.TRAIN.BBOX_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.0])

# RPN anchor loader
# rpn anchors batch size
config.TRAIN.RPN_BATCH_SIZE = 256
# rpn anchors sampling params
config.TRAIN.RPN_FG_FRACTION = 0.5
config.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
config.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
config.TRAIN.RPN_CLOBBER_POSITIVES = False
# rpn bounding box regression params
config.TRAIN.RPN_BBOX_WEIGHTS = [1.0, 1.0, 1.0, 1.0]
config.TRAIN.RPN_POSITIVE_WEIGHT = -1.0

# RPN proposal
config.TRAIN.RPN_NMS_THRESH = 0.7
config.TRAIN.RPN_PRE_NMS_TOP_N = 6000
config.TRAIN.RPN_POST_NMS_TOP_N = 1000
config.TRAIN.RPN_MIN_SIZE = 0
# approximate bounding box regression
config.TRAIN.BBOX_MEANS = [0.0, 0.0, 0.0, 0.0]
config.TRAIN.BBOX_STDS = [0.1, 0.1, 0.2, 0.2]

config.TEST = edict()

# R-CNN testing
# size of images for each device
config.TEST.BATCH_IMAGES = 1

# RPN proposal
config.TEST.RPN_NMS_THRESH = 0.7
config.TEST.RPN_PRE_NMS_TOP_N = 6000
config.TEST.RPN_POST_NMS_TOP_N = 1000
config.TEST.RPN_MIN_SIZE = 0

# RPN generate proposal
config.TEST.PROPOSAL_NMS_THRESH = 0.7
config.TEST.PROPOSAL_PRE_NMS_TOP_N = 20000
config.TEST.PROPOSAL_POST_NMS_TOP_N = 2000
config.TEST.PROPOSAL_MIN_SIZE = 0

# RCNN nms
config.TEST.NMS = 0.3

config.TEST.max_per_image = 300

# Test Model Epoch
config.TEST.test_epoch = 0

config.TEST.USE_SOFTNMS = False


def update_config(config_file):
    with open(config_file) as f:
        exp_config = edict(yaml.load(f))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    if k == 'TRAIN':
                        if 'BBOX_WEIGHTS' in v:
                            v['BBOX_WEIGHTS'] = np.array(v['BBOX_WEIGHTS'])
                    elif k == 'network':
                        if 'PIXEL_MEANS' in v:
                            v['PIXEL_MEANS'] = np.array(v['PIXEL_MEANS'])
                    for vk, vv in v.items():
                        config[k][vk] = vv
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("key must exist in config.py")
