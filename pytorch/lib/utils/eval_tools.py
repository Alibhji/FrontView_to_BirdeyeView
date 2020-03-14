import pandas as pd
import numpy as np

def IOU(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def mAP(meta_data ,iou_th = 0.5 , n_point=11):

    meta_data = pd.DataFrame(meta_data)
    
    ## Ap calculation
    total_objects = len(meta_data)
    meta_data['P_estimate'] = meta_data.apply(lambda row: True if(row.iou > iou_th) else False   ,axis=1)
    meta_data['TP']  = (meta_data['P_estimate'] == True).cumsum()
    meta_data['precision'] = meta_data.apply(lambda row: row.TP/(row.name+1)  ,axis=1)
    meta_data['recall'] = meta_data.apply(lambda row: row.TP/(total_objects)  ,axis=1)
    # Interpolated Precision
    meta_data['precision_IP'] = meta_data.groupby('recall')['precision'].transform(max)
    #print(meta_data)
    
    prec_at_rec=[]
    recall_level = np.linspace(0,1,11)
    for rl in recall_level:
        try:
            x = meta_data[meta_data['recall'] > rl]['precision_IP']
            prec = max(x)
        except:
            prec = 0

        prec_at_rec.append(prec)
    
    avg_prec = np.mean(prec_at_rec)
    #print('11 point precision is ', prec_at_rec)
    #print('mAP is ', avg_prec)
    return avg_prec
