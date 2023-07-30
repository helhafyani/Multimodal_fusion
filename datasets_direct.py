import torch
import cv2
import numpy as np
import os
import glob as glob
import pandas as pd
#from xml.etree import ElementTree as et
from config import (
    CLASSES,class_to_idx, RESIZE_TO, TRAIN_DIR, VALID_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from custom_utils import collate_fn, get_train_transform, get_valid_transform, NormalizeData, transform_shift

# the dataset class
class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None, transform_shift = transform_shift()):
        self.transforms = transforms
        self.dir_path = dir_path
        # self.dir_path_annot = dir_path_annot
        self.height = height
        self.width = width
        self.classes = classes
        self.transform_shift=transform_shift
        
        # get all the image paths in sorted order
        self.image_paths = glob.glob(f"{self.dir_path}/images/*.png") # change this according to your dataset path
        self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.image_paths]
        self.all_images = sorted(self.all_images)
        self.class_to_idx = class_to_idx


    def read_metadata(self, image_prefix, image_suffix, classe):

        metadata_link = os.path.join(self.dir_path, 'Prob_images_exp', image_prefix + '_' + classe + '_instance_color_RGB'+ image_suffix )
        image = cv2.imread(metadata_link)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height),interpolation = cv2.INTER_AREA)

        image_resized /= 255.0
        image_resized = np.expand_dims(image_resized, axis=0)
        image_resized = np.transpose(image_resized, (2,1,0)).astype(np.float32)

        '''add the shift to the masks'''
        image_resized = torch.from_numpy(image_resized)
        image_resized = self.transform_shift(image_resized)
        image_resized = image_resized.detach().numpy()
        return image_resized


    def __getitem__(self, idx):

        image_name = self.all_images[idx]
        image_prefix = image_name[:5]
        image_suffix = image_name[5:] 
        image_path = os.path.join(self.dir_path, 'images', image_name)
        # read the image
        image = cv2.imread(image_path)
        # convert BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0

        # read the metadata
        plane = self.read_metadata(image_prefix,image_suffix,'plane')
        ship = self.read_metadata(image_prefix,image_suffix,'ship')
        storage_tank = self.read_metadata(image_prefix,image_suffix,'storage_tank')
        baseball_diamond = self.read_metadata(image_prefix,image_suffix,'baseball_diamond')
        tennis_court = self.read_metadata(image_prefix,image_suffix,'tennis_court')
        basketball_court = self.read_metadata(image_prefix,image_suffix,'basketball_court')
        ground_track_field = self.read_metadata(image_prefix,image_suffix,'ground_track_field')
        harbor = self.read_metadata(image_prefix,image_suffix,'harbor')
        bridge = self.read_metadata(image_prefix,image_suffix,'bridge')
        large_vehicle = self.read_metadata(image_prefix,image_suffix,'large_vehicle')
        small_vehicle = self.read_metadata(image_prefix,image_suffix,'small_vehicle')
        helicopter = self.read_metadata(image_prefix,image_suffix,'helicopter')
        roundabout = self.read_metadata(image_prefix,image_suffix,'roundabout')
        soccer_ball_field = self.read_metadata(image_prefix,image_suffix,'soccer_ball_field')
        swimming_pool = self.read_metadata(image_prefix,image_suffix,'swimming_pool')


        
        # capture the corresponding TXT file for getting the annotations
        annot_filename = image_name[:-4] + '.txt'
        annot_file_path = os.path.join(self.dir_path, 'labelTxt', annot_filename)
        
        
        boxes = []
        labels = []

        image_width = image.shape[1]
        image_height = image.shape[0]
        
        with open(annot_file_path) as f:
            start = 0
            for line in f:
                splits = line.split(" ")
                labels.append(class_to_idx[splits[8]])
		    
                x1 = float(splits[0]) #- 1
                y1 = float(splits[1]) #- 1
                x2 = float(splits[2]) #- 1
                y2 = float(splits[3]) #- 1
                x3 = float(splits[4]) #- 1
                y3 = float(splits[5]) #- 1
                x4 = float(splits[6]) #- 1
                y4 = float(splits[7]) #- 1


                xmin = max(min(x1, x2, x3, x4), 0)
                xmax = max(x1, x2, x3, x4)
                ymin = max(min(y1, y2, y3, y4), 0)
                ymax = max(y1, y2, y3, y4)

                xmin_final = max((xmin/image_width)*self.width - 1, 0)
                xmax_final = max((xmax/image_width)*self.width, 0)
                ymin_final = max((ymin/image_height)*self.height - 1, 0)
                ymax_final = max((ymax/image_height)*self.height, 0)

                boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # bounding box to tensor
        if len(boxes) == 0:
            boxes = torch.zeros((0,4), dtype=torch.float32)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # area of the bounding boxes
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        neg_area_idx = np.where(area < 0)[0]
        if neg_area_idx.size:
            print("Negative areas founds in: ", annot_filename)

        # no crowd instances
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # labels to tensor
        labels = torch.as_tensor(labels, dtype=torch.int64)
        # prepare the final `target` dictionary
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        images = [image_resized, plane, ship, storage_tank, baseball_diamond, tennis_court,
                                basketball_court, ground_track_field, harbor, bridge, large_vehicle, small_vehicle,
                                helicopter, roundabout, soccer_ball_field, swimming_pool]
        
        images_cat = np.concatenate(images, axis=2)

        # apply the image transforms
        if self.transforms:
            sample = self.transforms(image = images_cat,
                                    bboxes = target['boxes'],
                                    labels = labels)
            images_cat = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        if len(target['boxes']) == 0:
            target['boxes'] = torch.zeros((0,4), dtype=torch.float32)

        return images_cat, target

    
    def __len__(self):
        return len(self.all_images)

# prepare the final datasets and data loaders
def create_train_dataset():
    train_dataset = CustomDataset(TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform())
    #train_dataset = CustomDataset(TRAIN_DIR,ANNOT_DIR, RESIZE_TO, RESIZE_TO, CLASSES)
    return train_dataset
def create_valid_dataset():
    valid_dataset = CustomDataset(VALID_DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform())
    return valid_dataset
def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    return valid_loader
# execute datasets.py using Python command from Terminal...
# ... to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    dataset = CustomDataset(
        TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    )
    print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    def visualize_sample(image, target):
        for box_num in range(len(target['boxes'])):
            box = target['boxes'][box_num]
            label = CLASSES[target['labels'][box_num]]
            cv2.rectangle(
                image, 
                (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                (0, 255, 0), 2
            )
            cv2.putText(
                image, label, (int(box[0]), int(box[1]-5)), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2
            )
        cv2.imshow('Image', image)
        cv2.waitKey(0)
        
    NUM_SAMPLES_TO_VISUALIZE = 5
    for i in range(NUM_SAMPLES_TO_VISUALIZE):
        image, target = dataset[i]
        visualize_sample(image, target)
