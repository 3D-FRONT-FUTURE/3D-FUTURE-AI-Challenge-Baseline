'''
demo evaluation code for reconstruction
'''
# coding=utf-8
import json
import sys
import os
#from pycocotools.mask import *
import numpy as np
import time
import zipfile
import os
import logging
import shutil
import heapq
import open3d as o3d
import math
import scipy.linalg as linalg
import pdb
from tqdm import tqdm

# 错误字典，这里只是示例
error_msg={
    1: "Input File is empty or Path is in-correct, please check input file or re-zip input file follow our instruction", 
    2: "Input sample number is in-correct, please check or re-download data",
    3: "Input sample can not be found in ground true dataset, please check input sample filename and make sure it is same as image name",
    4: "Some input sample not exist, please check input file",
    5: "Some input File name is in-correct, please check or re-download data",
    6: "Vertex of input file is not required (range from 1000 to 2048, but recommend 2048)",
    7: "Wrong input file format"
}

error_msg_detail={
    1: "Input File is empty or Path is in-correct, please check input file or re-zip input file follow our instruction", 
    2: "Input sample number is in-correct, please check or re-download data",
    3: "Input sample can not be found in ground true dataset, please check input sample filename and make sure it is same as image name",
    4: "Some input sample not exist, please check input file",
    5: "Some input File name is in-correct, please check or re-download data",
    6: "Vertex of input file is not required (range from 1000 to 2048, but recommend 2048)",
    7: "Main function wrong"
}


        
def get_submit_data(submit_dir):
    print('collect submit data')
    submit_filenames = os.listdir(submit_dir)
    if len(submit_filenames) == 0:
        return False, None
    return True, submit_filenames

def get_standard_data(standard_dir):
    print('collect standard data')
    
    standard_data = {}
    validation_data = np.load(os.path.join(standard_dir, 'validation_set.npy'), allow_pickle=True)
    standard_ply_dir = os.path.join(standard_dir, 'norm_model_ply')
    for sample in validation_data:
        image_id = sample['image']
        model_id = sample['model']
        view = sample['view']
        
        if image_id not in standard_data:
            standard_data[image_id] = {}
        standard_data[image_id] = {'image_id': image_id, 'model_id': model_id, 'view': view}
   
    return standard_data

def rotate_mat(axis, radian):
    rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
    return rot_matrix

def rotate_standard_model(view, standard_points):
    axis_x, axis_y, axis_z = [1,0,0], [0,1,0], [0, 0, 1]
    
    angle_y = 330 - 120 - view*30
    radians_y = math.radians(angle_y)
    rot_matrix1 = rotate_mat(axis_y, radians_y)

    angle_x = 14.477512185929925
    radians_x = math.radians(angle_x)
    rot_matrix2 = rotate_mat(axis_x, radians_x)
    rot_matrix = np.dot(rot_matrix2, rot_matrix1)

    standard_points = np.dot(rot_matrix, np.transpose(standard_points))
    standard_points = np.transpose(standard_points)
    return standard_points


def find_max_axis(all_vertex):
    # find max axis
    all_vertex_np = np.array(all_vertex)
    def find_max_min(axis_vertex):
        min_num = min(axis_vertex)
        max_num = max(axis_vertex)
        return min_num, max_num

    x_min, x_max = find_max_min(all_vertex_np[:, 0])
    y_min, y_max = find_max_min(all_vertex_np[:, 1])
    z_min, z_max = find_max_min(all_vertex_np[:, 2])

    x_mean = (x_max + x_min) / 2
    y_mean = (y_max + y_min) / 2
    z_mean = (z_max + z_min) / 2
    all_vertex_np[:, 0] = all_vertex_np[:, 0] - x_mean
    all_vertex_np[:, 1] = all_vertex_np[:, 1] - y_mean
    all_vertex_np[:, 2] = all_vertex_np[:, 2] - z_mean

    x_min, x_max = find_max_min(all_vertex_np[:, 0])
    y_min, y_max = find_max_min(all_vertex_np[:, 1])
    z_min, z_max = find_max_min(all_vertex_np[:, 2])

    x_len = abs(x_min - x_max)
    y_len = abs(y_min - y_max)
    z_len = abs(z_min - z_max)
    max_num = max(x_len, y_len, z_len)

    if max_num == x_len:
        return x_min, x_max, all_vertex_np
    elif max_num == y_len:
        return y_min, y_max, all_vertex_np
    elif max_num == z_len:
        return z_min, z_max, all_vertex_np
    else:
        print('max_num can not be recongnised')
        exit(-1)

def normalize_points(raw_points):
    all_vertex = raw_points
    # find max axis
    min_val, max_val, all_vertex_numpy = find_max_axis(all_vertex)

    # norm
    # all_vertex_numpy = np.array(all_vertex)
    norm_all_vertex_numpy = 2 * (all_vertex_numpy - min_val) / (max_val - min_val) + (-1)

    return norm_all_vertex_numpy


def distChamfer_np(a, b):
    x, y = a.astype(np.float32), b.astype(np.float32)
    bs, num_points_x, points_dim = x.shape
    bs, num_points_y, points_dim = y.shape
    xx = np.power(x, 2).sum(2)
    yy = np.power(y, 2).sum(2)

    zz = np.dot(x[0], y[0].transpose(1, 0))[np.newaxis, :][0]
    rx = xx[:, np.newaxis, :].repeat(num_points_y, axis=1)[0]
    ry = yy[:, np.newaxis, :].repeat(num_points_x, axis=1)[0]

    P = np.einsum('ji', rx) + ry - 2*zz

    return np.min(P, 1), np.min(P, 0), None, None
   
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

def evaluation_process(submit_dir, submit_filenames, standard_dir, standard_data):
    valid_number = 0
    f_score = 0
    top1_correct = 0
    continue_num = 0
    
    total_chamfer_dist = 0
    total_f_score = 0
    check_code = 0
    for i, submit_filename in enumerate(tqdm(submit_filenames)):
        submit_ply_id = submit_filename.split('.')[0]
        submit_ply_file = os.path.join(submit_dir, submit_filename)
        standard_ply_file = os.path.join(standard_dir, 'norm_model_ply', standard_data[submit_ply_id]['model_id']+'.ply')
        
        # check file exists
        if not os.path.exists(submit_ply_file):
            check_code = 4
            return check_code, None, None
        if not os.path.exists(standard_ply_file):
            check_code = 5
            return check_code, None, None
        
        submit_ply = o3d.io.read_point_cloud(submit_ply_file)
        standard_ply = o3d.io.read_point_cloud(standard_ply_file)
        
         
        # make sure point clouds are 2048
        if len(submit_ply.points) > 2048 or len(submit_ply.points) < 1000:
            check_code = 6
            return check_code, None, None
        
        submit_points = np.array(submit_ply.points)

        if len(standard_ply.points) <= 2048:
            standard_points = np.array(standard_ply.points)
        else:
            standard_points = downsample_pcd(np.array(standard_ply.points), desire_points = 2048) 
        
        # rotate standard 
        view = standard_data[submit_ply_id]['view']
        standard_points = rotate_standard_model(view, standard_points)
        
        # nomalize ply
        standard_points = normalize_points(standard_points)
        pred_points = normalize_points(submit_points)
        
        #start_time = time.time()
        # comput chamfer distance and f-score
        dist_pred, dist_standard, index_pred, index_standard = distChamfer_np(pred_points[np.newaxis, :], standard_points[np.newaxis, :])
        # print('time: ', time.time() - start_time, len(submit_points), len(standard_points))
        sample_chamfer_dist = np.mean(dist_pred) + np.mean(dist_standard)
        sample_f_score = f_score_function(pred_points, standard_points, dist_pred, dist_standard, threshold=0.04)
        #print('time: ', time.time() - start_time, len(submit_points), len(standard_points))

        total_chamfer_dist += sample_chamfer_dist
        total_f_score += sample_f_score

    mean_chamfer_dist = total_chamfer_dist / len(submit_filenames)
    mean_f_score = total_f_score / len(submit_filenames)
    return check_code, mean_chamfer_dist, mean_f_score

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

def report_score(score, chamfer_dist, mean_f_score, out_p):
    result = dict()
    result['success']=True
    result['score'] = score

    # 这里{}里面的score注意保留，但可以增加其他key，比如这样：
    # result['scoreJson'] = {'score': score, 'aaaa': 0.1}
    result['scoreJson'] = {
            'score': score,
            'chamfer distance': chamfer_dist,
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
    
    # 解压答案
    zip_standard_dir = 'standard'
    zip_submit_dir = 'submit'
    
    try:
        if os.path.isdir(zip_standard_dir) and len(os.listdir(zip_standard_dir)) > 0:
            logging.info("no need to unzip %s", standard_path)
        else:
            with zipfile.ZipFile(standard_path, "r") as zip_ref:
                zip_ref.extractall(zip_standard_dir)
        
        if os.path.isdir(zip_submit_dir):
            shutil.rmtree(zip_submit_dir)
        with zipfile.ZipFile(submit_path, "r") as zip_ref:
            zip_ref.extractall(zip_submit_dir)
            zip_ref.close()
        
        # 执行测评与数据检验程序
        # 获取数据
        zip_submit_dir = os.path.join(zip_submit_dir, 'reconstruction_results')
        sucessed, submit_filenames = get_submit_data(zip_submit_dir)
        standard_data = get_standard_data(zip_standard_dir)
        
        if sucessed == False:
            check_code = 1
            report_error_msg(error_msg_detail[check_code],error_msg[check_code], out_path)
        if len(submit_filenames) != len(standard_data):
            check_code = 2
            report_error_msg(error_msg_detail[check_code],error_msg[check_code], out_path)
        if submit_filenames[0].split('.')[0] not in standard_data:
            check_code = 3
            report_error_msg(error_msg_detail[check_code],error_msg[check_code], out_path)
        
        # 计算评价指标
        check_code, chamfer_dist, mean_f_score = evaluation_process(zip_submit_dir, submit_filenames, zip_standard_dir, standard_data)
        if check_code != 0:
            report_error_msg(error_msg_detail[check_code],error_msg[check_code], out_path)
        else:
            score = ((2 - chamfer_dist) / 2 * 100 + mean_f_score) / 2
            report_score(score, chamfer_dist, mean_f_score, out_path)
    except:
        check_code = 7
        report_error_msg(error_msg_detail[check_code],error_msg[check_code], out_path)
