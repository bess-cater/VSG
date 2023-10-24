from easydict import EasyDict as edict
import torch
#optimizer = optim.Adam(model.parameters(),lr=config.TRAIN.LR, betas=(0.9, 0.999), weight_decay=config.TRAIN.WEIGHT_DECAY)
config = edict()


#opthers
config.WORKERS = 16
config.LOG_DIR = ''
config.MODEL_DIR = ''
config.RESULT_DIR = ''
config.DATA_DIR = ''

#dataset
config.DATASET = edict()
config.DATASET.dir = "../data/tacos/tall_c3d_features.hdf5"

config.DATASET.n_clips = 512
config.DATASET.n_frames = 16


# common params for NETWORK
config.MODEL = edict()
config.MODEL.proj = 1000
config.MODEL.enc_layer=3
config.MODEL.optimizer = torch.optim.Adam
config.MODEL.NAME = ''
config.MODEL.CHECKPOINT = '' # The checkpoint for the best performance
config.MODEL.temp = 0.2
config.MODEL.hidden_size=256
config.MODEL.PYRAMID = edict()
config.MODEL.PYRAMID.chann1 = 450
config.MODEL.PYRAMID.chann2 = 700
config.MODEL.PYRAMID.chann3 = 300

#train
config.TRAIN=edict()
config.TRAIN.LR=0.0001 
config.TRAIN.BS = 16


config.LOSS = edict()
config.LOSS.NAME = 'bce_loss'
config.LOSS.PARAMS = None

# test
config.TEST = edict()
config.TEST.RECALL = []
config.TEST.TIOU = []
config.TEST.NMS_THRESH = 0.4
config.TEST.INTERVAL = 1
config.TEST.EVAL_TRAIN = False
config.TEST.BATCH_SIZE = 1
config.TEST.TOP_K = 10
