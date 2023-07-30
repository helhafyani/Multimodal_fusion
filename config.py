import torch
BATCH_SIZE = 8 # increase / decrease according to GPU memeory
RESIZE_TO = 1024 #416 #1024 # resize the image for training and transforms
NUM_EPOCHS = 50 # number of epochs to train for
NUM_WORKERS = 4
LR= 0.001 #0.001

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

TRAIN_DIR = '/users2/local/h22elhaf/Faster_RCNN_for_DOTA/train' 

VALID_DIR = '/users2/local/h22elhaf/Faster_RCNN_for_DOTA/val'#/images'

CLASSES = ['__background__', 'plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',\
            'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',\
                  'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']

idx_to_class = {i:j for i,j in enumerate(CLASSES)}
class_to_idx = {value:key for key, value in idx_to_class.items()}
NUM_CLASSES = len(CLASSES)
# whether to visualize images after crearing the data loaders
VISUALIZE_TRANSFORMED_IMAGES = False 
# location to save model and plots
OUT_DIR = 'outputs'
model_version = 'direct'
