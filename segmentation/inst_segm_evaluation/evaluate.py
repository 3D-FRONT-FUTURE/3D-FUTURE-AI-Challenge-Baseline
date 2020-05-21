# coding=utf-8
import json
import sys
import os
import os.path as osp
from coco import COCO
import numpy as np
import time
from future3d_eval import do_future3d_evaluation
import pdb
import logging

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)

def report_score(score_res, out_p):
    segm_s = score_res['segm']
    score = segm_s['AP']
    result = dict()
    result['score'] = score

    result['score_detail'] = {
        'score': score,
        'AP50': segm_s['AP50'],
        'AP75': segm_s['AP75'],
        'APs': segm_s['APs'],
        'APm': segm_s['APm'],
        'APl': segm_s['APl'],
    }

    dump_2_json(result,out_p)

if __name__=="__main__":
    '''
      evaluation
    '''
    pred_path = sys.argv[1] # segmentation file path
    gt_path = sys.argv[2] # ground-truth file path
    out_path = sys.argv[3] # json output path 

    try:
        dataset = COCO(gt_path)
        pred_file_path = pred_path
        eval_res = do_future3d_evaluation(
            dataset,
            pred_file_path,
            iou_types=["segm",],
        )
        report_score(eval_res.results, out_path)

    except Exception as e:
        print(e)

    

