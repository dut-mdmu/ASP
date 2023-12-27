from copy import deepcopy
import numpy as np
import math
import numba
import cupy as cp

from numba import njit, prange

def _pc_normalize(pc):
    """
    Normalize the point cloud to a unit sphere
    :param pc: input point cloud
    :return: normalized point cloud
    """
    #求x, y, z的平均值
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    #点云归一化
    pc = pc / m
    return pc


def _shuffle_pointcloud(pcd):
    """
    Shuffle the points
    :param pcd: input point cloud
    :return: shuffled point clouds
    """
    idx = np.random.rand(pcd.shape[0], 1).argsort(axis=0)
    #匹配1d索引和数据片从输入数据中取值
    return np.take_along_axis(pcd, idx, axis=0)


def _gen_random_cluster_sizes(num_clusters, total_cluster_size):
    """
    Generate random cluster sizes
    :param num_clusters: number of clusters
    :param total_cluster_size: total size of all clusters
    :return: a list of each cluster size
    """
    rand_list = np.random.randint(num_clusters, size=total_cluster_size)
    cluster_size_list = [sum(rand_list == i) for i in range(num_clusters)]
    return cluster_size_list


def _sample_points_inside_unit_sphere(number_of_particles):
    """
    Uniformly sample points in a unit sphere
    :param number_of_particles: number of points to sample
    :return: sampled points
    """
    radius = np.random.uniform(0.0, 1.0, (number_of_particles, 1))
    radius = np.power(radius, 1 / 3)
    costheta = np.random.uniform(-1.0, 1.0, (number_of_particles, 1))
    theta = np.arccos(costheta)
    phi = np.random.uniform(0, 2 * np.pi, (number_of_particles, 1))
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)
    return np.concatenate([x, y, z], axis=1)


def corrupt_scale(pointcloud):
    """
    Corrupt the scale of input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    #print("the shape of scale before corrupt", pointcloud.shape)
    level = 2
    s = [1.6, 1.7, 1.8, 1.9, 2.0][level]
    xyz = np.random.uniform(low=1. / s, high=s, size=[3])
    return _pc_normalize(np.multiply(pointcloud, xyz).astype('float32'))


@njit(nopython=True)
def corrupt_jitter(pointcloud):
    """
    Jitter the input point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    #print("in")
    level = 5
    sigma = 0.01 * (level + 1)
    N, C = pointcloud.shape
    #print(N, C)
    pointcloud = pointcloud + sigma * np.random.randn(N, C)
    return pointcloud


def corrupt_rotate(pointcloud, level):
    """
    Randomly rotate the point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    angle_clip = math.pi / 6
    angle_clip = angle_clip / 5 * (level + 1)
    angles = np.random.uniform(-angle_clip, angle_clip, size=(3))
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    return np.dot(pointcloud, R)


def corrupt_dropout_global(pointcloud, level):
    """
    Drop random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    drop_rate = [0.25, 0.375, 0.5, 0.625, 0.75][level]
    num_points = pointcloud.shape[0]
    pointcloud = _shuffle_pointcloud(pointcloud)
    pointcloud = pointcloud[:int(num_points * (1 - drop_rate)), :]
    return pointcloud


def corrupt_dropout_local(pointcloud, level):
    """
    Randomly drop local clusters
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    total_cluster_size = 100 * (level + 1)
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    for i in range(num_clusters):
        K = cluster_size_list[i]
        pointcloud = _shuffle_pointcloud(pointcloud)
        dist = np.sum((pointcloud - pointcloud[:1, :]) ** 2, axis=1, keepdims=True)
        idx = dist.argsort(axis=0)[::-1, :]
        pointcloud = np.take_along_axis(pointcloud, idx, axis=0)
        num_points -= K
        pointcloud = pointcloud[:num_points, :]
    return pointcloud


def corrupt_add_global(pointcloud, level):
    """
    Add random points globally
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    npoints = 10 * (level + 1)
    additional_pointcloud = _sample_points_inside_unit_sphere(npoints)
    pointcloud = np.concatenate([pointcloud, additional_pointcloud[:npoints]], axis=0)
    return pointcloud


def corrupt_add_local(pointcloud, level):
    """
    Randomly add local clusters to a point cloud
    :param pointcloud: input point cloud
    :param level: severity level
    :return: corrupted point cloud
    """
    num_points = pointcloud.shape[0]
    total_cluster_size = 100 * (level + 1)
    num_clusters = np.random.randint(1, 8)
    cluster_size_list = _gen_random_cluster_sizes(num_clusters, total_cluster_size)
    pointcloud = _shuffle_pointcloud(pointcloud)
    add_pcd = np.zeros_like(pointcloud)
    num_added = 0
    for i in range(num_clusters):
        K = cluster_size_list[i]
        sigma = np.random.uniform(0.075, 0.125)
        add_pcd[num_added:num_added + K, :] = np.copy(pointcloud[i:i + 1, :])
        add_pcd[num_added:num_added + K, :] = add_pcd[num_added:num_added + K, :] + sigma * np.random.randn(
            *add_pcd[num_added:num_added + K, :].shape)
        num_added += K
    assert num_added == total_cluster_size
    dist = np.sum(add_pcd ** 2, axis=1, keepdims=True).repeat(3, axis=1)
    add_pcd[dist > 1] = add_pcd[dist > 1] / dist[dist > 1]  # ensure the added points are inside a unit sphere
    pointcloud = np.concatenate([pointcloud, add_pcd], axis=0)
    pointcloud = pointcloud[:num_points + total_cluster_size]
    return pointcloud

# @njit(parallel=True)
def slope_corrupt(pcl, index, limit=20):
    '''
    caculating the slope of critical points, and randomly move those 
position in limit distance 
    input: 
        pcl(N, 3)
        index(n)
    '''
    pcl_corrupt = deepcopy(pcl)

    x = pcl[:,0]
    y = pcl[:,1]
    z = pcl[:,2]
    d = np.sqrt(np.square(x)+np.square(y))
    alpha = np.arctan(y/x)
    beta = np.arctan(z/d)
    for num in prange(len(index)):
        m = np.random.uniform(-limit,limit)
        # n = np.random.uniform(-limit,limit)
        # o = np.random.uniform(-limit,limit)
        indicies = np.int(index[num])
        pcl_corrupt[indicies][0] =pcl_corrupt[indicies][0] + m*np.cos(beta[num])*np.cos(alpha[num])
        pcl_corrupt[indicies][1] =pcl_corrupt[indicies][1] + m*np.cos(beta[num])*np.sin(alpha[num])
        pcl_corrupt[indicies][2] =pcl_corrupt[indicies][2]+ m*np.sin(beta[num])

    #np.save("slope/slope_%s"%(limit), pcl_corrupt)

    return pcl_corrupt

# @njit(parallel=True)
def corrupt_slope(pcl):
    pcl_corrupt = pcl

    # pcl_corrupt = cp.asarray(pcl)
    # pcl = pcl_corrupt

    #print(pcl.shape)
    x = pcl[:, 0]
    y = pcl[:, 1]
    z = pcl[:, 2]
    d = np.sqrt(np.square(x)+np.square(y))
    # d = np.sqrt(np.square(x)+np.square(y))
    alpha = np.arctan(y/x)
    beta = np.arctan(z/d)
    point_num, _ = pcl.shape
    for indicies in prange(point_num):
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
    # pcl_corrupt = np.asnumpy(pcl_corrupt)

    return pcl_corrupt

def corrupt_slope_random(pcl):
    pcl_corrupt = pcl

    # pcl_corrupt = cp.asarray(pcl)
    # pcl = pcl_corrupt

    #print(pcl.shape)
    x = pcl[:, 0]
    y = pcl[:, 1]
    z = pcl[:, 2]
    d = np.sqrt(np.square(x)+np.square(y))
    # d = np.sqrt(np.square(x)+np.square(y))
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
        m = np.random.uniform(limit, 5)
        pcl_corrupt[indicies][0] =pcl_corrupt[indicies][0] + m*np.cos(beta[indicies])*np.cos(alpha[indicies])
        pcl_corrupt[indicies][1] =pcl_corrupt[indicies][1] + m*np.cos(beta[indicies])*np.sin(alpha[indicies])
        pcl_corrupt[indicies][2] =pcl_corrupt[indicies][2]+ m*np.sin(beta[indicies])
    # pcl_corrupt = np.asnumpy(pcl_corrupt)

    return pcl_corrupt