import numpy as np
import cv2
import torch
import glob as glob
import os
import time
from model_fusion import create_model
from config import (
    NUM_CLASSES, DEVICE, CLASSES,model_version
)
from custom_utils import NormalizeData
# this will help us create a different color for each class
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
OUT_DIR = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_fusion_exp_'+model_version
# OUT_DIR = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_final_'+model_version
if os.path.exists(OUT_DIR):
    print('========================')
    print('Directory already exists')
    print('========================')
else:
    os.makedirs(OUT_DIR)

# load the best model and trained weights
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('/users2/local/datasets/h22elhaf/datasets/iSAID/outputs/output_fusion_exp_'+model_version+'/last_model.pth', map_location=DEVICE)
# checkpoint = torch.load('/users2/local/datasets/h22elhaf/datasets/iSAID/outputs/output_fusion_'+model_version+'/last_model.pth', map_location=DEVICE)
# checkpoint = torch.load('/users2/local/datasets/h22elhaf/datasets/iSAID/outputs/output_final_'+model_version+'/last_model.pth', map_location=DEVICE)
# checkpoint = torch.load('/users2/local/datasets/h22elhaf/datasets/iSAID/outputs/output_all_50epochs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()
# directory where all the images are present
DIR_TEST = '/users2/local/h22elhaf/Faster_RCNN_for_DOTA/test/images'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/test/images'
test_images = glob.glob(f"{DIR_TEST}/*.png")
print(f"Test instances: {len(test_images)}")
# define the detection threshold...
# ... any detection having score below this will be discarded
detection_threshold = 0.5 #0.8
# to count the total number of images iterated through
frame_count = 0
# to keep adding the FPS for each image
total_fps = 0 

def metadata_(image_prefix,image_suffix,classe,x,y): #,metadata_(image_prefix,image_suffix,classe,x,y)
    img_link = os.path.join('/users2/local/h22elhaf/Faster_RCNN_for_DOTA/val/Prob_images_exp' ,image_prefix + '_' + classe + '_instance_color_RGB' + image_suffix)
    # img_link = os.path.join('/users2/local/datasets/h22elhaf/datasets/iSAID/val/Prob_images' ,image_prefix + '_' + classe + '.png')
    # print(img_link)
    img = cv2.imread(img_link)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    img = cv2.resize(img, (x,y),interpolation = cv2.INTER_AREA)
    # img=NormalizeData(img)
    # img = 1 - img
    img /= 255.0
    return img

for i in range(len(test_images)):
    # get the image file name for saving output later on
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = cv2.imread(test_images[i])
    w=image.shape[0]
    h=image.shape[1]
    orig_image = image.copy()
    # BGR to RGB
    image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # make the pixel range between 0 and 1
    image /= 255.0
    # bring color channels to front
    image = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_prefix = test_images[i].split('/')[-1][:5]
    image_suffix = test_images[i].split('/')[-1][5:]

    plane = metadata_(image_prefix,image_suffix,'plane',h,w)
    ship = metadata_(image_prefix,image_suffix,'ship',h,w)
    storage_tank = metadata_(image_prefix,image_suffix,'storage_tank',h,w)
    baseball_diamond = metadata_(image_prefix,image_suffix,'baseball_diamond',h,w)
    tennis_court = metadata_(image_prefix,image_suffix,'tennis_court',h,w)
    basketball_court = metadata_(image_prefix,image_suffix,'basketball_court',h,w)
    ground_track_field = metadata_(image_prefix,image_suffix,'ground_track_field',h,w)
    harbor = metadata_(image_prefix,image_suffix,'harbor',h,w)
    bridge = metadata_(image_prefix,image_suffix,'bridge',h,w)
    large_vehicle = metadata_(image_prefix,image_suffix,'large_vehicle',h,w)
    small_vehicle = metadata_(image_prefix,image_suffix,'small_vehicle',h,w)
    helicopter = metadata_(image_prefix,image_suffix,'helicopter',h,w)
    roundabout = metadata_(image_prefix,image_suffix,'roundabout',h,w)
    soccer_ball_field = metadata_(image_prefix,image_suffix,'soccer_ball_field',h,w)
    swimming_pool = metadata_(image_prefix,image_suffix,'swimming_pool',h,w)

    metadata = np.stack([plane, ship, storage_tank, baseball_diamond, tennis_court,
                        basketball_court, ground_track_field, harbor, bridge, large_vehicle, small_vehicle,
                        helicopter, roundabout, soccer_ball_field, swimming_pool])
    fusion = np.concatenate([image, metadata], axis=0)


    # convert to tensor
    fusion = torch.tensor(fusion, dtype=torch.float).cuda()
    # add batch dimension
    fusion = torch.unsqueeze(fusion, 0)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(fusion.to(DEVICE))
    end_time = time.time()
    # get the current fps
    fps = 1 / (end_time - start_time)
    # add `fps` to `total_fps`
    total_fps += fps
    # increment frame count
    frame_count += 1
    # load all detection to CPU for further operations
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # carry further only if there are detected boxes
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # filter out boxes according to `detection_threshold`
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        draw_boxes = boxes.copy()
        # get all the predicited class names
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        pred_classes_ = np.array(pred_classes)[scores >= detection_threshold]
        scores_ = scores[scores>= detection_threshold]
        # prediction = np.concatenate([boxes, pred_classes_[:,None]],scores_, axis=1)
        prediction = np.concatenate([boxes, pred_classes_[:,None],scores_[:,None]], axis=1)
        # np.savetxt(f"/users2/local/datasets/h22elhaf/datasets/iSAID/inference_all_50epochs_test/{image_prefix}.txt", prediction, delimiter=",",fmt='%s',header="x_min,y_min,x_max,y_max,class,score")
        np.savetxt(f"{OUT_DIR}/{image_name}.txt", prediction, delimiter=",",fmt='%s',header="x_min,y_min,x_max,y_max,class,score")
        
        # draw the bounding boxes and write the class name on top of it
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            cv2.rectangle(orig_image,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color, 2)
            cv2.putText(orig_image, class_name, 
                        (int(box[0]), int(box[1]-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 
                        2, lineType=cv2.LINE_AA)
        # cv2.imshow('Prediction', orig_image)
        # cv2.waitKey(1)
        # cv2.imwrite(f"/users2/local/datasets/h22elhaf/datasets/iSAID/inference_all_50epochs_test/{image_name}.png", orig_image)
        cv2.imwrite(f"{OUT_DIR}/{image_name}.png", orig_image)
    else:
        prediction = np.empty((0,6))
        np.savetxt(f"{OUT_DIR}/{image_name}.txt", prediction, delimiter=",",fmt='%s',header="x_min,y_min,x_max,y_max,class,score")
        cv2.imwrite(f"{OUT_DIR}/{image_name}.png", orig_image)

    print(f"Image {i+1} done...")
    print('-'*50)
print('TEST PREDICTIONS COMPLETE')
# cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")