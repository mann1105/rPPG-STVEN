from yacs.config import CfgNode

def get_config():
    config = CfgNode()
    
    # Dataset settings
    config.DATASET = CfgNode()
    config.DATASET.NAME = "UBFC-rPPG"
    config.DATASET.ROOT = "data/UBFC-rPPG"
    config.DATASET.BATCH_SIZE = 8
    config.DATASET.NUM_WORKERS = 4
    config.DATASET.CLIP_LENGTH = 64  # number of frames per clip
    
    # Model settings
    config.MODEL = CfgNode()
    config.MODEL.STVEN_PRETRAINED = ""  # path to pretrained STVEN if available
    config.MODEL.RPPG_PRETRAINED = ""   # path to pretrained rPPGNet if available
    
    # Training settings
    config.TRAIN = CfgNode()
    config.TRAIN.EPOCHS = 50
    config.TRAIN.LR_STVEN = 1e-4
    config.TRAIN.BETA1 = 0.9
    config.TRAIN.BETA2 = 0.999
    config.TRAIN.SAVE_INTERVAL = 5
    config.TRAIN.LOG_INTERVAL = 100
    
    # Loss weights
    config.LOSS = CfgNode()
    config.LOSS.BINARY = 0.1
    config.LOSS.ECG = 1.0
    config.LOSS.ECG_AUX = 0.5
    config.LOSS.PERCEPTUAL = 1.0
    config.LOSS.STVEN = 0.0001
    
    return config
