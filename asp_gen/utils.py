import imp
import numpy as np
import open3d
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
import math
from copy import deepcopy
from numba import njit, jit, prange
import numba

def dele_attr(attri_map, level):
    '''
    删掉一部分只留Attr中的一部分，并且data里面的索引可以接到pcl上
    input: 
        attr_map: (N, 1)，当前点云每个点的重要程度
        level: 筛选点云的重要程度
    output: 
        data: (N*, 1)，新的attr_map，里面的attr值都是符合要求的
        pointlist：这些attr_map在pcl中的索引
    '''
    max_attr = np.array(attri_map).max()
    min_attr = np.array(attri_map).min()
    #attr的范围
    level_attr = min_attr + (max_attr - min_attr)*level
    point_list = np.array([[]])
    #data是新的attr_map，里面的点都是attr_map值大于最低要求的
    data = np.array([])
    for i in range(len(attri_map)):
        if(attri_map[i] > level_attr):
           data = np.append(data, attri_map[i])
           #point_list指的是这些高于标准值的attr_map在pcl中的索引。
           point_list = np.append(point_list, i)
    
    data = np.unique(data)
    point_list = np.unique(point_list)
    #print("point_list = ",point_list)
    return np.array(data), point_list

def new_pointcloud(pcl, pointlist):
    '''
    根据delet_attr得到的索引提取pcl中的点，建立新的pcl
    '''
    j = len(pointlist)
    new_pts = np.zeros((j, 4))
    for i in range(len(pointlist)):
        index = pointlist[i]
        index = np.int(index)
        new_pts[i] = pcl[index]
    #new_pts = new_pts.reshape(len(pointlist), 4)
    #print("new_points.shape = " ,new_pts.shape)
    return new_pts

def visualize_attr_map(points, box, attr_map, draw_origin=True):
    '''
    对源点云进行可视化
    '''
    turbo_cmap = plt.get_cmap('turbo')
    attr_map_scaled = attr_map - attr_map.min()
    attr_map_scaled /= attr_map_scaled.max()
    color = turbo_cmap(attr_map_scaled)[:, :3]
    print("shape of color: ",color.shape)

    vis = open3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().point_size = 4.0
    vis.get_render_option().background_color = np.ones(3) * 0.25

    if draw_origin:
        axis_pcd = open3d.geometry.TriangleMesh.create_coordinate_frame(
            size=1.0, origin=[0, 0, 0])
        vis.add_geometry(axis_pcd)

    rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
    bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
    bb.color = (1.0, 0.0, 1.0)
    vis.add_geometry(bb)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    pts.colors = open3d.utility.Vector3dVector(color)
    vis.add_geometry(pts)
    vis.run()
    vis.destroy_window()

def cropped_pcd(points, box):
    '''
    从点云中剪裁出检测框中的点
    '''
    rot_mat = Rotation.from_rotvec([0, 0, box[6]]).as_matrix()
    bb = open3d.geometry.OrientedBoundingBox(box[:3], rot_mat, box[3:6])
    bb.color = (1.0, 0.0, 1.0)

    pts = open3d.geometry.PointCloud()
    pts.points = open3d.utility.Vector3dVector(points[:, :3])
    cropped_pts = pts.crop(bb)
    cropped_points = np.array(cropped_pts.points)
    return  cropped_points

def replace_points(pcl, points1, points2):
    '''
    想办法把corrupt完成的点安回去
    '''
    n, c = pcl.shape
    n1, c1 = points1.shape
    index= []
    for i in range(n):
        for j in range(n1):
            if (pcl[i,0] == points1[j,0]) and (pcl[i, 1] == points1[j, 1] ) and (pcl[i, 2] == points1[j, 2]):
                #print(i)
                #print(j)
                pcl[i, 0] = points2[j, 0]
                pcl[i, 1] = points2[j, 1]
                pcl[i, 2] = points2[j, 2]
                
                index.append(i)
    index = np.array(index)
    return pcl, index

def corrupt_slope(pcl):
    pcl_corrupt = deepcopy(pcl)
    #print(pcl.shape)
    x = pcl[:, 0]
    y = pcl[:, 1]
    z = pcl[:, 2]
    d = np.sqrt(np.square(x)+np.square(y))
    alpha = np.arctan(y/x)
    beta = np.arctan(z/d)
    point_num, _ = pcl.shape
    for indicies in range(point_num):
        limit = -5
        x_1 = pcl_corrupt[indicies][0]
        y_1 = pcl_corrupt[indicies][1]
        z_1 = pcl_corrupt[indicies][2]
        distance = np.sqrt(x_1**2 + y_1**2 + z_1**2)
        if distance < np.abs(limit):
            limit = -distance
        m = np.random.uniform(limit, -1)
        pcl_corrupt[indicies][0] =pcl_corrupt[indicies][0] + m*np.cos(beta[indicies])*np.cos(alpha[indicies])
        pcl_corrupt[indicies][1] =pcl_corrupt[indicies][1] + m*np.cos(beta[indicies])*np.sin(alpha[indicies])
        pcl_corrupt[indicies][2] =pcl_corrupt[indicies][2]+ m*np.sin(beta[indicies])

    return pcl_corrupt

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