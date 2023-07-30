# import pandas as pd
import numpy as np
import cv2
import glob as glob
import os
from itertools import repeat
from config import model_version
# model_version = 'default_images_stat_416_lr_001'

# classe = 'swimming_pool'

# DIR_TEST = '/users/local/h22elhaf/datasets/iSAID/inference_direct_v3.0.1/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_all_50epochs/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_all_v2/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_whole_images/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_basic_'+model_version+'/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_fusion_'+model_version+'/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_indirect_'+model_version+'/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_basic_'+model_version+'/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_basic_best_model_'+model_version+'/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_basic_last_model_'+model_version+'/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_final_'+model_version+'/'
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_fusion_'+model_version+'_'+classe
# DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_indirect_exp_'+model_version
DIR_TEST = '/users2/local/datasets/h22elhaf/datasets/iSAID/inference_fusion_exp_'+model_version






CLASSES = ['plane', 'ship', 'storage-tank', 'baseball-diamond', 'tennis-court',\
            'basketball-court', 'ground-track-field', 'harbor', 'bridge', 'large-vehicle',\
                  'small-vehicle', 'helicopter', 'roundabout', 'soccer-ball-field', 'swimming-pool']
# CLASSES = ['__background__','small-vehicle','ship']
idx_to_class = {i:j for i,j in enumerate(CLASSES)}
class_to_idx = {value:key for key, value in idx_to_class.items()}

def read_data(image_txt):

    # load ground truth
    # with open('/users2/local/datasets/h22elhaf/datasets/iSAID/test/labelTxt/'+image_txt, 'r') as f:
    with open('/users2/local/h22elhaf/Faster_RCNN_for_DOTA/test/labelTxt/'+image_txt, 'r') as f:
        gt_lines = f.readlines()
    gt_data = []
    start = 0
    for line in gt_lines:
        # if (start == 0) | (start == 1):
        #     start +=1
        # else:
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

    # xmin1, ymin1, xmax1, ymax1 = box1
    # xmin2, ymin2, xmax2, ymax2 = box2

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
# DIR_TEST = '/users2/local/datasets/iSAID/Test_'
# DIR_TEST = '/users/local/h22elhaf/datasets/iSAID/inference_outputs_direct'
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
'''
{0: 'plane',1: 'ship',2: 'storage-tank', 3: 'baseball-diamond', 4: 'tennis-court', 5: 'basketball-court', 6: 'ground-track-field', 7: 'harbor',8: 'bridge',9: 'large-vehicle',10: 'small-vehicle',11: 'helicopter',12: 'roundabout',13: 'soccer-ball-field',14: 'swimming-pool'}
{0: 0.6949647588717811, 1: 0.4619148660909375, 2: 0.6263098348552442, 3: 0.3164640292493694, 4: 0.6799604553984966, 5: 0.1963373537334909, 6: 0.2799550734759615, 7: 0.41490289934520547, 8: 0.10784109906971856, 9: 0.4780358553940301, 10: 0.3088184424122428, 11: 0.018867924433962263, 12: 0.2004418636610234, 13: 0.3481980007642901, 14: 0.07412486118105148}
{'plane': 0.6949647588717811, 
'ship': 0.4619148660909375, 
'storage-tank': 0.6263098348552442, 
'baseball-diamond': 0.3164640292493694, 
'tennis-court': 0.6799604553984966, 
'basketball-court': 0.1963373537334909, 
'ground-track-field': 0.2799550734759615, 
'harbor': 0.41490289934520547, 
'bridge': 0.10784109906971856, 
'large-vehicle': 0.4780358553940301, 
'small-vehicle': 0.3088184424122428, 
'helicopter': 0.018867924433962263, 
'roundabout': 0.2004418636610234, 
'soccer-ball-field': 0.3481980007642901, 
'swimming-pool': 0.07412486118105148}
mAP=0.34714248786245366

11 point precision is  [1.0, 0.9025394646533974, 0.9025394646533974, 0.8905380333951762, 0.8714604236343366, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mAP is  0.41518885330330074

Latest model of inference_all_v2: 
{0: 0.7030600717711698, 1: 0.4694270796946401, 2: 0.6469598211435322, 3: 0.3650244808445513, 4: 0.7387576799849312, 5: 0.2889088568199779, 6: 0.1762505030545657, 7: 0.4174667095865503, 8: 0.1248436722332123, 9: 0.46900180129074986, 10: 0.28316481198673404, 11: 0.056603773380503146, 12: 0.338223188457783, 13: 0.24812464388047598, 14: 0.12396958695447863}
mAP = 0.363319112072257
11 point precision is  [1.0, 0.8792667509481669, 0.8792667509481669, 0.8462039752362333, 0.8255702917771883, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mAP is  0.4027552517190687

with 50 epochs:
{0: 0.7039296959793306, 1: 0.4706963938033582, 2: 0.6821764165212609, 3: 0.33063232308200124, 4: 0.7623110498236811, 5: 0.3115244553163375, 6: 0.2573114199420172, 7: 0.4895174595099506, 8: 0.17099471974514618, 9: 0.5367164939030915, 10: 0.31222056482530103, 11: 0.11613949876408959, 12: 0.3272620390260432, 13: 0.36086032349346986, 14: 0.15913952431361011}
mAP = 0.39942882520324596
11 point precision is  [1.0, 0.8921753723515838, 0.8921753723515838, 0.8684936228886591, 0.8476666312320612, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
mAP is  0.40913736352944435

inference on the images as a whole:
{0: 0.43640151112558223, 1: 0.1817520041320684, 2: 0.1406478313041122, 3: 0.3254561970599159, 4: 0.7484307515258236, 5: 0.30051675614002576, 6: 0.21916006579258243, 7: 0.2868790144385919, 8: 0.0463456996295952, 9: 0.3791217055516076, 10: 0.15610582768271847, 11: 0, 12: 0.04265027891242055, 13: 0.23940735406339686, 14: 0.08739088595182133}
0.23935105888735078
'''