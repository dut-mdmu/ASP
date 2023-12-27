from run_occam import get_attr, get_attr_path
import os
import numpy as np

velodyn_path = "/data/datasets/kitti_yzy/training/velodyne/"
maps_path = "/data/datasets/kitti_yzy/pointrcnn_maps/"
img_idx_list = [x.strip() for x in open("/home/yzy/occam/experiment/demo_train.txt").readlines()]
velodyne_names = os.listdir(velodyn_path)
for i in range(len(img_idx_list)):
    img_idx = img_idx_list[i]

    img_idx_list.append(img_idx)
    pcl_path = velodyn_path + '/' + img_idx +'.bin'
    map_path = maps_path + '/' + img_idx
    map_file = map_path + '.npy'
    if os.path.exists(map_file):    
        print("the file has existed!")
        continue
    else:
        map = get_attr_path(pcl_path)
            
        print("the map of %s will be saved to: "%img_idx, map_path )