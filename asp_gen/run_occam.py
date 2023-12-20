'''
    this file is mostly based on occam_demo
'''
import sys
sys.path.append("..")
import os
import argparse
from copy import deepcopy
from locale import getpreferredencoding
from tkinter.ttk import LabeledScale
from unittest import result
import torch
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
from occam import OccAM
import numpy as np
import utils
import corruption_utils as corrupt_methods
from pcdet.ops.iou3d_nms.iou3d_nms_utils import boxes_iou3d_gpu
import math
import open3d
from atmos_models import LISA
from scipy.spatial.transform import Rotation
os.environ['CUDA_VISIBLE_DEVICES']='0'
import warnings
warnings.filterwarnings("ignore")
slope = 100
def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--model_cfg_file', type=str,
                        default='/home/yzy/occam/cfgs/kitti_models/pv_rcnn.yaml',
                        help='dataset/model config for the demo')    
    parser.add_argument('--occam_cfg_file', type=str,
                        default='/home/yzy/occam/cfgs/occam_configs/kitti_pointpillar.yaml',
                        help='specify the OccAM config')
    parser.add_argument('--ckpt', type=str, default='/home/yzy/OpenPCDet/pv_rcnn_8369.pth', 
                        help='path to pretrained model parameters')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size for OccAM creation')
    parser.add_argument('--workers', type=int, default=0,
                        help='number of workers for dataloader')
    parser.add_argument('--nr_it', type=int, default=2000,
                        help='number of sub-sampling iterations N')
    parser.add_argument('--level', default=0.002, type=float, help='this is corruption level')


    args = parser.parse_args()

    cfg_from_yaml_file(args.model_cfg_file, cfg)
    cfg_from_yaml_file(args.occam_cfg_file, cfg)

    return args, cfg
args, config = parse_config()
logger = common_utils.create_logger()
logger.info('---------------------OccAM Start--------------')
occam = OccAM(data_config=config.DATA_CONFIG, model_config=config.MODEL,
                occam_config=config.OCCAM, class_names=config.CLASS_NAMES,
                  model_ckpt_path=args.ckpt, nr_it=args.nr_it, logger=logger)

def point_in_box(point, box):
    '''
    input:
        point[x, y, z]
        box[cx, cy, cz, dx, dy, dz, heading]
    
    output: 
        True/False: bool
    '''
    #求偏移量
    shift_x = point[0] - box[0]
    shift_y = point[1] - box[1]
    shift_z = point[2] - box[2]

    cos_a = math.cos(box[6])
    sin_a = math.sin(box[6])
    dx, dy, dz = box[3], box[4], box[5]
    local_x = shift_x * cos_a + shift_y * sin_a
    local_y = shift_y * cos_a - shift_x * sin_a

    if(abs(shift_z)>dz/2.0  or abs(local_x)>dx/2.0 or abs(local_y)>dy/2.0):
        
        return False
    return True

def point_in_boxes(point, boxes):
    box_num, _ = boxes.shape
    for i in range(box_num):
        if point_in_box(point, boxes[i]) ==True :
            return True
    return False

def get_attr(pcl):
    base_det = occam.get_base_predictions(pcl=pcl)
    base_det_boxes, base_det_labels, base_det_scores = base_det
    attr_maps = occam.compute_attribution_maps(
        pcl=pcl, base_det_boxes=base_det_boxes,
        base_det_labels=base_det_labels, batch_size=args.batch_size,
        num_workers=args.workers)
    
    return attr_maps, base_det_boxes

# def get_attr(file_list):
#     '''
#         将数据集放入模型，得到对应的attr_map
#     '''
    
#     labels = []
#     boxes = []
#     maps = []
#     pcls = []
#     for i in range(len(file_list)):
#         file_name = file_list[i]
#         pcl = occam.load_and_preprocess_pcl(file_name)

#         base_det = occam.get_base_predictions(pcl=pcl)
#         base_det_boxes, base_det_labels, base_det_scores = base_det
#         attr_maps = occam.compute_attribution_maps(
#         pcl=pcl, base_det_boxes=base_det_boxes,
#         base_det_labels=base_det_labels, batch_size=args.batch_size,
#         num_workers=args.workers)
#         labels.append(base_det_labels)
#         boxes.append(base_det_boxes)
#         maps.append(attr_maps)
#         pcls.append(pcl)
#     print("----------------data upload successful!--------------")
#     return pcls, labels, boxes, maps

def get_attr_path(file_path):
    pcl = occam.load_and_preprocess_pcl(file_path)

    base_det = occam.get_base_predictions(pcl=pcl)
    base_det_boxes, base_det_labels, base_det_scores = base_det
    attr_maps = occam.compute_attribution_maps(
        pcl=pcl, base_det_boxes=base_det_boxes,
        base_det_labels=base_det_labels, batch_size=args.batch_size,
        num_workers=args.workers)
    
    return attr_maps


def get_prediction(pcl_path):
    '''
        只进行检测，不计算occam
        return: boxes, labels, scores
    '''
    detection = occam.get_base_predictions(pcl= pcl_path)
    det_boxes, det_labels, det_scores = detection
    return det_boxes, det_labels, det_scores

def prepare_pcl_on_path(pcl_path):
    '''
        针对文件路径上的pcl进行prepare
    '''
    pcl = occam.load_and_preprocess_pcl(pcl_path)

    return pcl


def target_score(map, level):
    
    #正序排序
    map_sort = sorted(map, reverse= True)
    length = len(map)
    target_length = int(length * level)
    score = map_sort[target_length]
    return score

def derive_map_out_box(maps, level = 0.7):
    """得到"""
    N, M = maps.shape
    # print("N", N)
    # print("M", M)
    maps_1 = np.ones((N, M))
    #print("maps_1.shape = ", maps_1.shape)
    for i in range(N):
        map = maps[i]
        # max_map = np.array(map).max()
        # min_map = np.array(map).min()
        # level_map = min_map + (max_map - min_map) * level
        level_map = target_score(map, level)
        for j in range(len(map)):
            #如果点的重要度不够高，将重要度设为0
            if map[j] < level_map:
                maps_1[i,j] = 0
    vertical_points = np.arange(M)
    #如果在所有map中这个点的重要度都是0，清除
    for i in range(M):
        maps_1_line = maps_1[:, i]
        if np.all(maps_1_line == 0):
            vertical_points[i] = 0
    
    zeros = np.argwhere(vertical_points == 0)
    vertical_points = np.delete(vertical_points, zeros)
    print("the shape of vertical point is", vertical_points.shape)
    return vertical_points

def get_index_list(maps, level):
    index_list = []
    for i in range(len(maps)):
        map_1 = maps[i]
        score = target_score(map_1, level)
        index = []
        for j in range(len(map_1)):
            if map_1[j] > score:
                index.append(j)
        index_list.append(index)
    return index_list

def slope_with_map(pcl, maps, level, boxes):
    index_list = get_index_list(maps, level)
    for indexes in index_list:
        for index in indexes:
            point = pcl[index]
            point = point[:3]
            if point_in_boxes(point, boxes) == True:
                limit = -5
                target_point = pcl[index, :3]
                x = target_point[0]
                y = target_point[1]
                z = target_point[2]
                d = np.sqrt(np.square(x)+np.square(y))
                alpha = np.arctan(y/x)
                beta = np.arctan(z/d)
                m = np.random.uniform(limit, -1)
                pcl[index][0] = x + m*np.cos(beta) * np.cos(alpha)
                pcl[index][1] = y + m*np.cos(beta) * np.sin(alpha)
                pcl[index][2] = z + m*np.sin(beta)
            else:
                indexes.remove(index)
    return pcl
    


def corrupt_inside_box(pcl, boxes, maps, corrupt, level):
    index_list = get_index_list(maps, level)
    print(len(index_list))
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcl[:, :3])
    crop_point_list = []
    corrupt_point_list = []
    for i in range(len(index_list)):
        indexes = index_list[i]
        box = boxes[i]
        rot_mat = Rotation.from_rotvec([0,0,box[6]]).as_matrix()
        bbox = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        cropped_pts = pcd.crop(bbox)
        box_points = np.array(cropped_pts.points)

        points1 = pcl[indexes, :3]
        corrupt_point = []
        for i in range(len(indexes)):
            point = points1[i]
            if point_in_boxes(point, boxes) == True:
                #print("in box")
                corrupt_point.append(point)
        if len(corrupt_point) == 0 or len(box_points) == 0:
            crop_num = []
        else:
            crop_num = np.random.choice(len(box_points), len(corrupt_point))
        crop_point = box_points[crop_num]
        corrupt_point = np.array(corrupt_point)
        #corrupt_point = corrupt_point[np.all(corrupt_point!=0, axis=1)]
        #print(corrupt_point.shape)
        corrupt_point = corrupt_point.reshape(-1, 3)

        #print("the shape of corrupt_point",corrupt_point.shape)
        crop_point_list.append(crop_point)
        corrupt_point_list.append(corrupt_point)
    result_points = pcl
    for i in range(len(corrupt_point_list)):
        corrupt_point = corrupt_point_list[i]
        #print(corrupt_point.shape)
        corrupt_point = corrupt_point[:, :3]
        if corrupt =='slope':
            slope_point = corrupt_methods.corrupt_slope(corrupt_point)
            slope_points, _ = utils.replace_points(result_points, corrupt_point, slope_point)
            result_points = slope_points
        if corrupt == 'jitter':
            jitter_point = corrupt_methods.corrupt_jitter(corrupt_point)
            jitter_points, _ = utils.replace_points(result_points, corrupt_point, jitter_point)
            result_points = jitter_points    
        if corrupt == 'scale':
            scale_point = corrupt_methods.corrupt_scale(corrupt_point)
            scale_points, _ = utils.replace_points(result_points, corrupt_point, scale_point)
            result_points = scale_points
    
        if corrupt == 'slope_random':
            crop_point = crop_point_list[i]
            crop_point = crop_point[:, :3]
            slope_random_point = corrupt_methods.corrupt_slope(crop_point)
            slope_random_points , _ = utils.replace_points(result_points, crop_point, slope_random_point)
            result_points = slope_random_points
    return result_points
        

def corrupt_out_of_box(pcl, boxes, maps, level = 0.7, corrupt = 'slope'):
    '''
        1、从单个attr(N, 1)中取出权重最大的关键点
        2、判断这些点是否在box中，如果在box内不动，不在的话替换。
        return: slope_boxes, slope_labels, crop_boxes, slope_labels
    '''
    new_index = derive_map_out_box(maps, level=level)
    
    corrupt_point = []
    #当前的map中，判断点是否在box中
    # for i in range(len(map_index)):
    for i in range(len(new_index)):
        #index = map_index[i, 1]
        index = new_index[i]
        point = pcl[np.int(index)]
        #得到需要进行操作的点的集合，这些点不在任何一个box内，所以不会影响结果
        if point_in_boxes(point, boxes) == False :
            '''
            目前有两种思路：
            1)把不在box内的点云当作一个整体，对这个整体进行简单的扰动
            2)遇到一个点就扣掉一个点，抠完回去再做检测
            '''
            #corrupt_point = np.append(corrupt_point, point)
            #corrupt_point[i, :] = point
            corrupt_point.append(point)
            #print(point)

    
    corrupt_point = np.array(corrupt_point)
    corrupt_point = corrupt_point[:, :3]
    print("corrupt_point.shape = ", corrupt_point.shape)
    #print("the len of corrupt_point is ", len(corrupt_point))
    print("next step")
    N = len(corrupt_point)
    if N == 0:
        crop_points = pcl
    else:
        crop_points = delete_out_of_box(pcl, corrupt_point)

    if N ==0:
        print("the number of point is 0")
        return pcl
    
        #1、得到corrupt之后的新点云(jitter, scale, slope)
        #2、替换点云，得到完整点云
    else:
        if corrupt == 'slope':
            slope_point = corrupt_methods.corrupt_slope(corrupt_point)
            slope_points, _ = utils.replace_points(pcl, corrupt_point, slope_point)
            print("replace over")
            result_points = slope_points   
            return result_points#, crop_points 
        if corrupt == 'jitter':
            jitter_point = corrupt_methods.corrupt_jitter(corrupt_point)
            print("corrupt over")
            jitter_points, _ = utils.replace_points(pcl, corrupt_point, jitter_point)
            print("replace over")
            result_points = jitter_points
            return result_points#, crop_points
        if corrupt == 'scale':
            scale_point = corrupt_methods.corrupt_scale(corrupt_point)
            print("corrupt over")
            scale_points, _ = utils.replace_points(pcl, corrupt_point, scale_point)
            print("replace over")
            result_points = scale_points
            return result_points#, crop_points
    
    # slope_boxes, slope_labels, _ = get_prediction(slope_points)
    # crop_boxes, crop_labels, _ = get_prediction(crop_points)
    # # scale_boxes, scale_labels, _ = get_prediction(pcl= scale_points)
    # # jitter_boxes, jitter_labels, _ = get_prediction(pcl= jitter_points)
    # boxes = torch.from_numpy(boxes).cuda()
    # slope_boxes = torch.from_numpy(slope_boxes).cuda()
    # # slope_iou = boxes_iou3d_gpu(boxes_a=boxes, boxes_b= slope_boxes)
    # # print("slope_iou = ", slope_iou)

    


def delete_out_of_box(pcl, points):
    pcl_copy = deepcopy(pcl)
    n,_ = pcl.shape
    n1, _ = points.shape
    for i in range(n):
        #print(" i = ", i)
        for j in range(n1):
            if (pcl[i,0] == points[j,0]) & (pcl[i, 1] == points[j, 1] )& (pcl[i, 2] == points[j, 2]):
                pcl_copy[i, :3] = [0, 0, 0]

    pcl_copy = pcl_copy[np.all(pcl_copy!=0,axis=1)]
    return pcl_copy

def corrupt_lisa(boxes, pcl):
    lisa = LISA(atm_model='snow')
    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(pcl[:, :3])
    pcl2 = pcl
    print(len(boxes))
    for i in range(len(boxes)):
        box = boxes[i]
        rot_mat = Rotation.from_rotvec([0,0,box[6]]).as_matrix()
        bbox = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
        cropped_pts = pcd.crop(bbox)
        box_points = np.array(cropped_pts.points)
        # print("box_points: ", box_points)
        #   corrupt_pcl = lisa.augment(box_points, 33)
        pcl_corrupt, indexes = utils.replace_points(pcl, box_points, box_points)
        target_pcl = pcl[indexes, :]
        # print(indexes)
        # corrupt_pcl = lisa.augment(target_pcl, 33)
        for i in len(indexes):
            index = indexes[i]
            # pcl2[index] = corrupt_pcl[i]
        pert_boxes, labels, scores = get_prediction(pcl2)
        print(labels)
    return pcl2

