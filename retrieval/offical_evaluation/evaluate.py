'''
evaluation test code for retrieval
'''
# coding=utf-8
import json
import sys
import os
import numpy as np
import time
import zipfile
import os
import logging
import shutil
import heapq
import open3d as o3d
from tqdm import tqdm
import pdb
import random

# 错误字典，这里只是示例
error_msg={
    1: "Wrong input file structu, please follow zip structure", 
    2: "Scientific notation is found in input file, please convert to decimal notitions",
    3: "Wrong sample number, please check or re-download dataset",
    4: "Input file not exists",
    5: "Wrong input format"
}

# 错误字典，这里只是示例
error_msg_detail={
    1: "Wrong input file structu, please follow zip structure", 
    2: "Scientific notation is found in input file, please convert to decimal notitions",
    3: "Wrong sample number, please check or re-download dataset",
    4: "Input file not exists",
    5: "Errors in main function"
}



def zip_data(standard_dir, standard_path, submit_dir, submit_path):
    if os.path.isdir(standard_dir) and len(os.listdir(standard_dir)) > 0:
        logging.info("no need to unzip %s", standard_path)
    else:
        with zipfile.ZipFile(standard_path, "r") as zip_ref:
            zip_ref.extractall(standard_dir)
    
    if os.path.isdir(submit_dir):
        shutil.rmtree(submit_dir)
    with zipfile.ZipFile(submit_path, "r") as zip_ref:
        zip_ref.extractall(submit_dir)
        zip_ref.close()

def get_submit_data(submit_file):
    print('collect submit data')
    with open(submit_file, 'r') as f:
        lines = f.readlines()

    submit_data = []
    for i, line in enumerate(lines):
        if 'e-' in line or 'E-' in line: 
            return False, None
        
        sample_id, dist_str_list = line.split(':')[0], line.split(':')[-1].strip().split(',')
        str2float_dist = map(float, dist_str_list)
        dist_list = list(str2float_dist)
        submit_data.append({'image_id': sample_id, 'dists': dist_list, 'index': i})
    return True, submit_data

def get_standard_data(standard_dir):
    '''
    random select
    '''
    print('collect standard data')
    standard_data = {}
    ply_filenames = os.listdir(os.path.join(standard_dir, 'norm_model_ply'))
    
    for i in range(16256):
        image_id = '{0:07d}'.format(i)
        model_id = random.choice(ply_filenames).split('.')[0]
        if image_id not in standard_data:
            standard_data[image_id] = {'image_id': image_id, 'model_id': model_id}
    
    return standard_data


def distChamfer(a, b):
    x, y = a.double(), b.double()
    bs, num_points_x, points_dim = x.size()
    bs, num_points_y, points_dim = y.size()

    xx = torch.pow(x, 2).sum(2)
    yy = torch.pow(y, 2).sum(2)
    zz = torch.bmm(x, y.transpose(2, 1))
    rx = xx.unsqueeze(1).expand(bs, num_points_y, num_points_x) # Diagonal elements xx
    ry = yy.unsqueeze(1).expand(bs, num_points_x, num_points_y) # Diagonal elements yy
    P = rx.transpose(2, 1) + ry - 2 * zz
    return torch.min(P, 2)[0].float(), torch.min(P, 1)[0].float(), torch.min(P, 2)[1].int(), torch.min(P, 1)[1].int()

def distChamfer_np(a, b):
    x, y = a.astype(np.float32), b.astype(np.float32)
    bs, num_points_x, points_dim = x.shape
    bs, num_points_y, points_dim = y.shape

    xx = np.power(x, 2).sum(2)
    yy = np.power(y, 2).sum(2)

    zz = np.dot(x[0], y[0].transpose(1, 0))
    zz = zz[np.newaxis, :]
    rx = xx[:, np.newaxis,:]
    rx = rx.repeat(num_points_y, axis=1)
    ry = yy[:, np.newaxis, :]
    ry = ry.repeat(num_points_x, axis=1)
    # a = rx.transpose(0, 2, 1)
    P = rx.transpose(0, 2, 1) + ry - 2 * zz
    return np.min(P, 2)[0], np.min(P, 1)[0], None ,None

def f_score_function(points, labels, dist1, dist2, threshold):
    len_points = points.shape[0]
    len_labels = labels.shape[0]

    num = len(np.where(dist1 <= threshold)[0]) + 0.0
    P = 100.0 * (num / len_points)
    num = len(np.where(dist2 <= threshold)[0]) + 0.0
    R = 100.0 * (num / len_labels)
    f_score = (2 * P * R) / (P + R + 1e-6)
    return np.array(f_score)

def downsample_pcd(pcd_points, desire_points = 2048):
    if len(pcd_points) <= desire_points:
        return pcd_points 

    ratio = int(len(pcd_points) / desire_points)
    downsample_pcd_points = pcd_points[::ratio, :]

    return downsample_pcd_points

def top5_f_score(submit_top5_min_index_list, standard_sample, standard_dir):
    # get standard pcd
    standard_model_index = int(standard_sample['model_id'])
    standard_model_file = os.path.join(standard_dir, 'norm_model_ply', '{0:07d}'.format(standard_model_index) + '.ply')
    if not os.path.exists(standard_model_file):
        return False, None
    standard_pcd = o3d.io.read_point_cloud(standard_model_file)
    standard_points = downsample_pcd(np.array(standard_pcd.points), desire_points=400)

    #import time
    top_5_f_score = 0
    
    for i in range(5):
        pred_model_index = submit_top5_min_index_list[i]
        pred_model_file = os.path.join(standard_dir, 'norm_model_ply', '{0:07d}'.format(pred_model_index) + '.ply')
        if not os.path.exists(pred_model_file):
            return False, None
        
        pred_pcd = o3d.io.read_point_cloud(pred_model_file)
        pred_points = downsample_pcd(np.array(pred_pcd.points), desire_points=400)
        
        # chamfer distance
        dist_pred, dist_standard, index_pred, index_standard = distChamfer_np(pred_points[np.newaxis, :], standard_points[np.newaxis, :])

        # f score
        top_5_f_score += f_score_function(pred_points, standard_points, dist_pred, dist_standard, threshold=0.04)
    return True, top_5_f_score/5

def evaluation_process(submit_data, standard_data, standard_dir):
    # 只选取一部分数据进行测试 - 1/3
    valid_number = 0
    f_score = 0
    top1_correct = 0
    continue_num = 0
    
    for i in tqdm(range(len(submit_data))):
        submit_sample = submit_data[i]
        if submit_sample['image_id'] not in standard_data: 
            continue_num += 1
            continue
        standard_sample = standard_data[submit_sample['image_id']]

        # 前5最小值索引
        submit_top5_min_index_list = []
        submit_top5_min_index_list = list(map(submit_sample['dists'].index, heapq.nsmallest(5, submit_sample['dists'])))
        submit_top1_min_index = submit_top5_min_index_list[0]

        # top 1 acc
        if submit_top1_min_index == int(standard_sample['model_id']):
            top1_correct += 1
        # f1 scores
        sucessed, mean_top5_f_score = top5_f_score(submit_top5_min_index_list, standard_sample, standard_dir)
        if sucessed == False:
            return False, None, None
        f_score += mean_top5_f_score
        valid_number += 1
        
    acc = top1_correct / valid_number
    mean_f_score = f_score / valid_number

    return True, acc, mean_f_score

def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file)

def report_error_msg(detail, showMsg, out_p):
    error_dict=dict()
    error_dict['errorDetail']=detail
    error_dict['errorMsg']=showMsg
    error_dict['score']=0
    error_dict['scoreJson']={}
    error_dict['success']=False
    dump_2_json(error_dict,out_p)

def report_score(score, acc, mean_f_score, out_p):
    result = dict()
    result['success']=True
    result['score'] = score

    # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
    # result['scoreJson'] = {'score': score, 'aaaa': 0.1}
    result['scoreJson'] = {
            'score': score,
            'top1_acc': acc,
            'mean_f_score': mean_f_score
            }

    dump_2_json(result,out_p)

if __name__=="__main__":
    '''
      online evaluation 
    '''
    in_param_path = sys.argv[1]
    out_path = sys.argv[2]

    # read submit and answer file from first parameter
    with open(in_param_path, 'r') as load_f:
        input_params = json.load(load_f)

    # 标准答案路径
    standard_path=input_params["fileData"]["standardFilePath"]
    print("Read standard from %s" % standard_path)

    # 选手提交的结果文件路径
    submit_path=input_params["fileData"]["userFilePath"]
    print("Read user submit file from %s" % submit_path)
     
    start_time = time.time()
    # 解压答案
    zip_standard_dir = 'standard_data'
    zip_submit_dir = 'retrieval_results'
    
    zip_data(zip_standard_dir, standard_path, zip_submit_dir, submit_path)
    
    # 执行测评与数据检验程序
    try:
        submit_file = os.path.join(zip_submit_dir, 'retrieval_results.txt')
        if not os.path.exists(submit_file):
            check_code = 1
            report_error_msg(error_msg[check_code],error_msg[check_code], out_path)
        sucessed, submit_data = get_submit_data(submit_file)
        standard_data = get_standard_data(zip_standard_dir)
        
        if sucessed == False:
            check_code = 2
            report_error_msg(error_msg[check_code],error_msg[check_code], out_path)
        # 输入的验证集数量和提供的不一致，check_code = 3
        if len(submit_data) != len(standard_data):
            check_code = 3
            report_error_msg(error_msg[check_code],error_msg[check_code], out_path)
        
        # 计算评价指标
        sucessed, acc, mean_f_score = evaluation_process(submit_data, standard_data, zip_standard_dir)
        if sucessed:
            score = acc*100*0.65 + mean_f_score*0.35
            report_score(score, acc, mean_f_score, out_path)
        else:
            check_code = 4
            report_error_msg(error_msg_detail[check_code],error_msg[check_code], out_path)
    except:
        check_code = 5
        report_error_msg(error_msg_detail[check_code],error_msg[check_code], out_path)

    print('process time: ', time.time() - start_time)
