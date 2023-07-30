# Multimodal_fusion

Please change the path to your data in config.file

In order to create probability maps, please refer to img_prob_exp.py.

datasets_direct.py: reads the images and prepare the dataloader for the direct fusion model.

model_direct.py: calls the pretrained faster-rcnn model and changes the first layer to accept 18 channels.

train_direct.py: finetune the faster-rcnn model on the DOTA dataset.

