import numpy as np
import pickle
import os
from run_occam import prepare_pcl_on_path, point_in_box, get_prediction
from atmos_models import LISA

load_path = '/data/dataset/kitti_lisa/training/velodyne/'
save_path = '/data/dataset/kitti_lisa2/training/velodyne/'
val_path = '/data/dataset/kitti_lisa/ImageSets/val.txt'
#攻击的目标
keys = ['Pedestrian', 'Car', 'Cyclist']

#加载database数据pkl = open('/data/dataset/kitti_lisa/kitti_dbinfos_train.pkl', 'rb')
# data = pickle.load(pkl)
# 

def refresh_data(load_path, save_path, val_path):
    '''
    input:
        load_path: clean velodyne path
        save_path: corrupt velodyne path
        val_path: validation velodyne list
    refresh corrupted data to previous version.
    '''
    img_idx_list = [x.strip() for x in open(val_path).readlines()]
    for i in range(len(img_idx_list)):
        img_idx = img_idx_list[i]
        PCL_path = load_path + '/' + img_idx + '.bin'
        PCL_path2 = save_path + '/' + img_idx +  '.bin'
        PCL = prepare_pcl_on_path(PCL_path)
        PCL.tofile(PCL_path2)
        print("refreshing %s"%img_idx)
    print("done.")

def replace_pcl(object_pcl, corrupt_obj, sc_pcl):
    '''
    input:
        object_pcl: object from gt_database
        corrupt_obj: object after corrupt 
        sc_pcl: source pcl(source database)
    '''
    n, c = sc_pcl.shape
    n1, c1 = object_pcl.shape
    for i in range(n):
        for j in range(n1):
            if (sc_pcl[i, 0] == object_pcl[j, 0]) and (sc_pcl[i, 1] == object_pcl[j, 1]) and (sc_pcl[i, 2] == object_pcl[j, 2]):
                sc_pcl[i, 0] = corrupt_obj[j, 0]
                sc_pcl[i, 1] = corrupt_obj[j, 1]
                sc_pcl[i, 2] = corrupt_obj[j, 2]
                sc_pcl[i, 3] = corrupt_obj[j, 3]
                #print("in")

    return sc_pcl

def get_points_from_box(pcl, box):
    '''
    input:
        pcl:场景点云
        box:gt中的3d_lidar_box信息
    output
        obj_pcl:需要扰动的物体对象
    '''
    obj_pcl =[]
    pcl_xyz = pcl[:, :3]
    for i in range(len(pcl)):
        point = pcl_xyz[i]
        if point_in_box(point, box):
            obj_pcl.append(pcl[i])
    
    obj_pcl = np.array(obj_pcl)
    return obj_pcl

def corrupt_lisa(pert_boxes, sc_pcl):
    '''
    input:
        pert_boxes: boxes from previous prediction(M, 7)
        sc_pcl: source pointcloud(N, 4)
    '''
    sv_pcl = sc_pcl
    for j in range(len(pert_boxes)):
        box = pert_boxes[j]
        obj_pcl= get_points_from_box(sc_pcl, box)
        if len(obj_pcl) == 0:
            print('pass')
            continue
        lisa = LISA(atm_model='snow')
        corrupt_obj = lisa.augment(obj_pcl, 33)
        corrupt_obj = corrupt_obj[:, :4]
        sv_pcl = replace_pcl(obj_pcl, corrupt_obj, sv_pcl)
    return sv_pcl

def main():
    img_idx_list = [x.strip() for x in open("/data/dataset/kitti_lisa/ImageSets/val.txt").readlines()]    
    for img_idx in img_idx_list:
        print(img_idx)
        sc_pcl_path = load_path + img_idx + '.bin'
        #提取的干净点云
        sc_pcl = prepare_pcl_on_path(sc_pcl_path)
        pert_boxes, pert_labels, pert_scores = get_prediction(sc_pcl)
        #用于保存的点云
        sv_pcl = sc_pcl
        #refresh_data(load_path, save_path, val_path)
        box_list =  []

        for j in range(len(pert_boxes)):
            box = pert_boxes[j]
            obj_pcl= get_points_from_box(sc_pcl, box)
            if len(obj_pcl) == 0:
                print('pass')
                continue
            lisa = LISA(atm_model='snow')
            corrupt_obj = lisa.augment(obj_pcl, 33)
            corrupt_obj = corrupt_obj[:, :4]
            sv_pcl = replace_pcl(obj_pcl, corrupt_obj, sv_pcl)

        
        corrupt_boxes, corrupt_labels, corrupt_scores = get_prediction(sv_pcl)

        print("the label of clean pcl: ", pert_labels)
        print("the label of corrupt pcl: ", corrupt_labels)
        sv_pcl_path = save_path + img_idx + '.bin'
        sv_pcl.tofile(sv_pcl_path)    
                