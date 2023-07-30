from config import (
    DEVICE, NUM_CLASSES, NUM_EPOCHS, OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, NUM_WORKERS,model_version, LR
)
from model_fusion import create_model
# from model_fusion_feature_map import create_model
from custom_utils import Averager, SaveBestModel, save_model, save_loss_plot
from tqdm.auto import tqdm
from datasetsV2_fusion_exp import (
    create_train_dataset, create_valid_dataset, 
    create_train_loader, create_valid_loader
)
import torch
import matplotlib.pyplot as plt
import time
import os
plt.style.use('ggplot')

# OUT_DIR = '/users/local/h22elhaf/datasets/iSAID/outputs/outputs_subset'
# OUT_DIR = '/users2/local/datasets/h22elhaf/datasets/iSAID/outputs/output_all_whole_images'

OUT_DIR = '/users2/local/datasets/h22elhaf/datasets/iSAID/outputs/output_fusion_exp_'+model_version
# OUT_DIR = '/users2/local/datasets/h22elhaf/datasets/iSAID/outputs/output_final_'+model_version
if os.path.exists(OUT_DIR):
    print('========================')
    print('Directory already exists')
    print('========================')
else:
    os.makedirs(OUT_DIR)

# function for running training iterations
def train(train_data_loader, model):
    print('Training')
    global train_itr
    global train_loss_list
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        # print(i)
        # try:
        # for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        # print('check 1')
        # faire un tensor au lieu de list avec
        #images = torch.stack(images).to(DEVICE)#
        images = list(image.to(DEVICE) for image in images)
        # print('check 2')
        # images = images.to(DEVICE)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        # print('check 3')
        # print(len(images))
        # print(images[0].shape) #conccat
        loss_dict = model(images, targets)
        # print('check 4')
        losses = sum(loss for loss in loss_dict.values())
        # print('check 5')
        loss_value = losses.item()
        # print('check 6')
        train_loss_list.append(loss_value)
        # print('check 7')
        train_loss_hist.send(loss_value)
        # print('check 8')
        losses.backward()
        # print('check 9')
        optimizer.step()
        # print('check 10')
        train_itr += 1
        # print('check 11')
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        # except ValueError:
        #     print('Invalid value!')
      
    return train_loss_list

    # function for running validation iterations
def validate(valid_data_loader, model):
    print('Validating')
    global val_itr
    global val_loss_list
    
    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    
    # print('avant')
    for i, data in enumerate(prog_bar):
        # print(i)
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list
if __name__ == '__main__':
    train_dataset = create_train_dataset()
    valid_dataset = create_valid_dataset()
    train_loader = create_train_loader(train_dataset, NUM_WORKERS)
    valid_loader = create_valid_loader(valid_dataset, NUM_WORKERS)
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}\n")

    # initialize the model and move to the computation device
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    # get the model parameters
    params = [p for p in model.parameters() if p.requires_grad]
    # define the optimizer
    # optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.SGD(params, lr=LR, momentum=0.9, weight_decay=0.0005)
    # optimizer = torch.optim.SGD(params, lr=1e-5, momentum=0.9, weight_decay=0.0005)
    # initialize the Averager class
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

    # name to save the trained model with
    MODEL_NAME = 'model'
    # whether to show transformed images from data loader or not
    if VISUALIZE_TRANSFORMED_IMAGES:
        from custom_utils import show_tranformed_image
        show_tranformed_image(train_loader)
    # initialize SaveBestModel class
    
    save_best_model = SaveBestModel()
    
    # start the training epochs
    for epoch in range(NUM_EPOCHS):
        print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()
        # start timer and carry out training and validation
        start = time.time()
        train_loss = train(train_loader, model)
        val_loss = validate(valid_loader, model)
        print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
        print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
        # save the best model till now if we have the least loss in the...
        # ... current epoch
        
        save_best_model(
            OUT_DIR, val_loss_hist.value, epoch, model, optimizer
        )
        # save the current epoch model
        save_model(epoch, OUT_DIR, model, optimizer)
        # save loss plot
        save_loss_plot(OUT_DIR, train_loss, val_loss)
        
        # sleep for 5 seconds after each epoch'''
        time.sleep(5)