'''
    根据txt获得需要validate的velodyne, image_2, 以及calib
    对点云进行干扰。
    干扰后的点云放入folder
'''
import os
import shutil

'''
确定路径
'''
sc_kitti_root = '/data/datasets/kitti_yzy/training/'
sc_velodyne_root = sc_kitti_root + 'velodyne/'
sc_calib_root = sc_kitti_root + 'calib/'
sc_image_root = sc_kitti_root + 'image_2/'
sc_label_root = sc_kitti_root + 'label_2/'

tg_kitti_root = '/data/datasets/fog_kitti2/training/'
tg_velodyne_root = tg_kitti_root + 'velodyne/'
tg_calib_root = tg_kitti_root + 'calib/'
tg_image_root = tg_kitti_root + 'image_2/'
tg_label_root = tg_kitti_root + 'label_2/'

def copy_file(img_idx):
    calib = img_idx + '.txt'
    image_2 = img_idx + '.png'
    velodyne = img_idx + '.bin'
    label = img_idx + '.txt'
    shutil.copy(sc_velodyne_root+ velodyne, tg_velodyne_root + velodyne)
    shutil.copy(sc_calib_root + calib, tg_calib_root + calib)
    shutil.copy(sc_image_root + image_2, tg_image_root + image_2)
    shutil.copy(sc_label_root + label, tg_label_root +label )


#确定干扰方式
name = 'slope'
maps_path = '/data/datasets/voxe_rcnn_maps/'

img_idx_list = [x.strip() for x in open("/data/datasets/fog_kitti2/ImageSets/val.txt").readlines()]
for i in range(len(img_idx_list)):
    print(i)
    img_idx = img_idx_list[i]
    copy_file(img_idx)

    