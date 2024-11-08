import os
import numpy as np
from run_occam import corrupt_inside_box, prepare_pcl_on_path, get_prediction, get_attr_path, get_attr
from run_occam import slope_with_map
from atmos_models import LISA
# from run_occam import augumentation
import argparse

velodyne_path = "/data/datasets/kitti_yzy/training/velodyne/"
name = 'slope'
# maps_path = "/data/datasets/kitti_yzy/maps"
save_path = "/data/datasets/kitti_yzy2/kitti/training/velodyne/"

os.environ['CUDA_VISIBLE_DEVICES']='0'
parser = argparse.ArgumentParser(description='add level')
parser.add_argument('--level', default=0.002, type=float, help='this is corruption level')
arg = parser.parse_args()

level1 = arg.level
img_idx_list = [x.strip() for x in open("/data/datasets/kitti_yzy2/kitti/ImageSets/val.txt").readlines()]
#000000
for i in range(len(img_idx_list)):
    print(i)
    img_idx = img_idx_list[i]
    sc_pcl_path =  velodyne_path + '/' + img_idx +'.bin'
    print(img_idx)
    pcl = prepare_pcl_on_path(sc_pcl_path)
    pert_boxes, pert_labels, pert_scores = get_prediction(pcl)

    map, pert_boxes = get_attr(pcl)
    # corrupt_pcl = corrupt_inside_box(pcl, pert_boxes, map, name, level=0.2)
    corrupt_pcl = slope_with_map(pcl, map, level1, pert_boxes)

    # lisa = LISA(atm_model='snow')
    # corrupt_pcl = lisa.augment(pcl, 33)
    # corrupt_pcl = corrupt_pcl[:, :4]


    corrupt_boxes, corrupt_labels, corrupt_scores = get_prediction(corrupt_pcl)
    print("the label of clean data: ", pert_labels)
    print("the label of corrupt data: ", corrupt_labels)
    crp_pcl_path = save_path + '/' + img_idx +'.bin'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    corrupt_pcl.tofile(crp_pcl_path)
    print("the corrupt point will be saved to:", crp_pcl_path)

