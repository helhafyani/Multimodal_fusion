import torch
import torchvision
import torch.nn as nn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def create_model(num_classes, n_image_channels = 18):
    
    # load Faster RCNN pre-trained model
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # Initialize a new convolutional layer with random weights for the additional channels
    random_conv_layer = torch.nn.Conv2d(n_image_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

    # Copy the pre-trained weights of the RGB channels to the new layer
    random_conv_layer.weight.data[:, :3, :, :] = model.backbone.body.conv1.weight.data

    # Replace the first layer in the model with the new layer
    model.backbone.body.conv1 = random_conv_layer

    model.transform.image_mean = [0.485, 0.456, 0.406, 0.095,0.195,0.092,0.061, 0.145,0.049,0.083,0.169, 0.092, 0.349,0.517,0.016,0.071,0.079,0.107]
    model.transform.image_std = [0.229, 0.224, 0.225, 0.249, 0.322, 0.225, 0.193, 0.29, 0.173, 0.22, 0.318, 0.231, 0.361, 0.344, 0.099,0.201,0.218,0.244] 
    

    # model.transform.image_mean = [0.485, 0.456, 0.406]#, 0.095,0.195,0.092,0.061, 0.145,0.049,0.083,0.169, 0.092, 0.349,0.517,0.016,0.071,0.079,0.107]
    # model.transform.image_std = [0.229, 0.224, 0.225]


    # get the number of input features 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # define a new head for the detector with required number of classes
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes) 
    return model