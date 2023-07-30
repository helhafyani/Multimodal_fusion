
import numpy as np
import cv2
import glob as glob
import os
from itertools import repeat
from config import model_version

DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_'+model_version

CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',\
            'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',\
                  'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
idx_to_class = {i:j for i,j in enumerate(CLASSES)}
class_to_idx = {value:key for key, value in idx_to_class.items()}

def read_data(image_txt):

    # load ground truth
 
    with open('/users2/local/h22elhaf/Faster_RCNN_for_DOTA/test/labelTxt/'+image_txt, 'r') as f:
        gt_lines = f.readlines()
    gt_data = []
    start = 0
    for line in gt_lines:
        data = line.strip().split()
        img_id = image_txt[:5]
        bbox = [min([float(data[i]) for i in [0,2,4,6]]), 
                min([float(data[i]) for i in [1,3,5,7]]),
                max([float(data[i]) for i in [0,2,4,6]]),
                max([float(data[i]) for i in [1,3,5,7]])]
        label = int(class_to_idx[data[8]])
        gt_data.append({
            'image_id': img_id,
            'category_id': label,
            'bbox': bbox,
        })

    #load prediction

    with open(os.path.join(DIR_TEST,image_txt), 'r') as f:
        pred_lines = f.readlines()
    pred_data = []
    start = 0
    for line in pred_lines:
        if start == 0:
            start +=1
        else:
            data = line.strip().split(',')
            img_id = image_txt[:5]
            bbox = [float(data[i]) for i in range(4)]
            label = int(class_to_idx[data[4]])
            # score = float(data[6])
            pred_data.append({
                'image_id': img_id,
                'category_id': label,
                'bbox': bbox#,
                # 'score': score,
            })

    return gt_data, pred_data


# Define IoU function
def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])


    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


# iterate over the data and compute IoU
# directory where all the images are present
test_files = glob.glob(f"{DIR_TEST}/*.txt")
test_files = sorted(test_files)
print(f"Test instances: {len(test_files)}")
print('test_files', test_files[0])

image_name = []
IoU = []
all_detections = {}
all_annotations = {}

for i in range(len(test_files)):
    #get the file name
    file_name = test_files[i].split(os.path.sep)[-1]
    # print(i, file_name)
    try:
        gt_data, pred_data = read_data(file_name)
        # image_name.extend(repeat(file_name,len(df)))
        # subset1 = df[['xmin_pred', 'ymin_pred', 'xmax_pred', 'ymax_pred']]
        # subset2 = df[['xmin', 'xmax','ymin', 'ymax', 'class',]]

        # all_detections = {}
        for data in pred_data:
            img_id = data['image_id']
            label = data['category_id']
            # score = data['score']
            bbox = data['bbox']
            if img_id not in all_detections:
                all_detections[img_id] = {}
            if label not in all_detections[img_id]:
                all_detections[img_id][label] = []
            # all_detections[img_id][label].append((score, bbox))
            all_detections[img_id][label].append((bbox))

        # all_annotations = {}
        for data in gt_data:
            img_id = data['image_id']
            label = data['category_id']
            bbox = data['bbox']
            if img_id not in all_annotations:
                all_annotations[img_id] = {}
            if label not in all_annotations[img_id]:
                all_annotations[img_id][label] = []
            all_annotations[img_id][label].append(bbox)
    except:
        print('no file :',file_name)
        continue

average_precisions = {}
for label in range(15):  # assuming there are 15 classes
    true_positives = []
    false_positives = []
    # scores = []
    num_annotations = 0
    for img_id in all_annotations:
        if label in all_annotations[img_id]:
            num_annotations += len(all_annotations[img_id][label])
        if img_id not in all_detections:
            continue
        if label not in all_detections[img_id]:
            continue
        detections = all_detections[img_id][label]
        detections = sorted(detections, reverse=True)
        for detection in detections:
            # score, bbox = detection
            bbox = detection
            if img_id not in all_annotations:
                false_positives.append(1)
                true_positives.append(0)
                # scores.append(score)
            
            else:
                if label in all_annotations[img_id].keys(): # if we have only one image
                    overlaps = [iou(bbox, annotation) for annotation in all_annotations[img_id][label]]
                    max_overlap = max(overlaps)
                    if max_overlap >= 0.5:
                        false_positives.append(0)
                        true_positives.append(1)
                    else:
                        false_positives.append(1)
                        true_positives.append(0)
                # scores.append(score)

    if num_annotations == 0:
        average_precisions[label] = 0, 0
        continue

    true_positives = np.array(true_positives)
    false_positives = np.array(false_positives)
    # scores = np.array(scores)

    # indices = np.argsort(-scores)
    # true_positives = true_positives[indices]
    # false_positives = false_positives[indices]

    true_positives = np.cumsum(true_positives)
    false_positives = np.cumsum(false_positives)

    recall = true_positives / num_annotations
    precision = true_positives / (true_positives + false_positives + 1e-8)

    average_precision = 0
    for i in range(len(recall) - 1):
        average_precision += (recall[i + 1] - recall[i]) * precision[i + 1]
    average_precisions[label] = average_precision

print(average_precisions)
mean_average_precision = np.mean(list(average_precisions.values()))
print(mean_average_precision)
