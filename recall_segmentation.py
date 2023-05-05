import base64
import json
from io import BytesIO
import requests
from PIL import Image
import cv2
import os
import time
import numpy as np
import pandas as pd

def get_iou(gt:list, pred:list) ->float:
    """
    Calculate iou score between groundtruth and prediction
    INPUT:
        gt:     mask of groundtruth (numpy)
        pred:   mask of prediction  (numpy)
    OUTPUT:
        iou:    iou of gt over prediction
    """
    intersection = np.logical_and(gt, pred)
    union = np.logical_or(gt, pred)
    iou_score = np.sum(intersection) / np.sum(union)
    print(iou_score)

    return iou_score

def get_recall(gt_mask_file, pred_mask_file, IOU_THRESHOLD = 0.9):

    positive = 0
    true_positive = 0
    predictions = 0
    durations = []
    duration = 0
    recall = 0


    gt_mask = cv2.imread(gt_mask_file, 1)
    pred_mask = cv2.imread(pred_mask_file, 1)
    #iou
    iou = get_iou(gt_mask, pred_mask)
    print('IOU : {}'.format(iou))

    #compare
    if iou >= IOU_THRESHOLD:
        true_positive += 1
        predictions += 1
    else:
        predictions += 1
    
    # For each gt damage on image
    positive += 1

    print(f"==={iou}===")
    print("===============================================")
    
        
    recall = round(true_positive/positive,2)
    duration = np.round(np.mean(durations), 2)
    print("Recall: {}%".format(recall))
    print("Average Inference Time {}".format(duration))
    return recall, true_positive

if __name__ == '__main__':
    gt_mask_file = 'img1.png'
    gt_mask_file = 'img2.png'

    iou_threshold = [0.9]
    results = {
        'iou': [],
        'tp': [],
        'recall': []
    }
    for threshold in iou_threshold:
        recall, tp = get_recall(gt_mask_file, gt_mask_file, threshold)
        results['iou'].append(threshold)
        results['tp'].append(tp)
        results['recall'].append(recall)
    
    df = pd.DataFrame(results)
    df.to_csv('recall_result.csv')
    