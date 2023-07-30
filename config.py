import torch
BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 1024 #416 #1024 # resize the image for training and transforms
NUM_EPOCHS = 50 # number of epochs to train for
NUM_WORKERS = 4
LR= 0.001 #0.001

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# training images and XML files directory
# TRAIN_DIR = '/users/local/datasets/iSAID/Train/images/train'
# TRAIN_DIR = '/users2/local/datasets/iSAID/Train_'
TRAIN_DIR = '/users2/local/h22elhaf/Faster_RCNN_for_DOTA/train'#/images'
# TRAIN_DIR = '/users2/local/datasets/h22elhaf/datasets/iSAID/train'
VALID_DIR = '/users2/local/h22elhaf/Faster_RCNN_for_DOTA/val'#/images'
# VALID_DIR = '/users2/local/datasets/h22elhaf/datasets/iSAID/val'
# ANNOT_DIR = '/users/local/datasets/iSAID/Train/labelTxt-v1.5/DOTA-v1.5_train/'
# validation images and XML files directory
# VALID_DIR = '/users2/local/datasets/iSAID/Val_' #'/users/local/datasets/iSAID/Train/images/val'
# classes: 0 index is reserved for background
# CLASSES = ('__background__', 'vehicle', 'plane', 'harbor', 'storage-tank') # ship instead of harbor
CLASSES = ['__background__', 'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',\
            'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',\
                  'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
# CLASSES = ['__background__', 'roundabout', 'small-vehicle', 'ship']
idx_to_class = {i:j for i,j in enumerate(CLASSES)}
class_to_idx = {value:key for key, value in idx_to_class.items()}
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False #True
# location to save model and plots
OUT_DIR = 'outputs'
model_version = 'default_images_stat_1024_lr_001_from_basic_inter'
# model_version = 'default_images_stat_416_lr_001'
# model_version = 'directe'
# model_version = 'basic_1024'