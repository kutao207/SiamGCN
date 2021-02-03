import os
import os.path as osp
import errno

from laspy.file import File
import numpy as np
import pandas as pd

def random_subsample(points,n_samples):
    if points.shape[0]==0:
        print('No points found at this center replacing with dummy')
        points = np.zeros((1,points.shape[1]))
    #No point sampling if already 
    if n_samples < points.shape[0]:
        random_indices = np.random.choice(points.shape[0],n_samples, replace=False)
        points = points[random_indices,:]
    return points

def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e

def to_list(x):
    if not isinstance(x, (tuple, list)):
        x = [x]
    return x

def files_exist(files):
    return len(files) != 0 and all(osp.exists(f) for f in files)

def find_file(file_list:list, scene_num: str):
    for f in file_list:
        if scene_num == osp.basename(f).split('_')[0]:
            return f
    raise f"{scene_num} not exist!"

def load_las(path):
    input_las = File(path, mode='r')
    point_records = input_las.points.copy()
    las_scaleX = input_las.header.scale[0]
    las_offsetX = input_las.header.offset[0]
    las_scaleY = input_las.header.scale[1]
    las_offsetY = input_las.header.offset[1]
    las_scaleZ = input_las.header.scale[2]
    las_offsetZ = input_las.header.offset[2]

    # px = point_records['point']['X'] * las_scaleX
    # py = point_records['point']['Y'] * las_scaleY
    # pz = point_records['point']['Z'] * las_scaleZ
    # r = input_las.red / 65535.0
    # g = input_las.red / 65535.0
    # b = input_las.red / 65535.0

    # points = np.hstack((px, py, pz, r, g, b))

    # return points, [las_scaleX, las_offsetX, las_scaleY, las_offsetY, las_scaleZ, las_offsetZ] 


    # calculating coordinates
    p_X = np.array((point_records['point']['X'] * las_scaleX) + las_offsetX)
    p_Y = np.array((point_records['point']['Y'] * las_scaleY) + las_offsetY)
    p_Z = np.array((point_records['point']['Z'] * las_scaleZ) + las_offsetZ)

    points = np.vstack((p_X,p_Y,p_Z,input_las.red,input_las.green,input_las.blue)).T
    
    return points, [las_scaleX, las_offsetX, las_scaleY, las_offsetY, las_scaleZ, las_offsetZ]


def extract_area(full_cloud,center,clearance,shape= 'cylinder'):
    if shape == 'square':
        x_mask = ((center[0]+clearance)>full_cloud[:,0]) &   (full_cloud[:,0] >(center[0]-clearance))
        y_mask = ((center[1]+clearance)>full_cloud[:,1]) &   (full_cloud[:,1] >(center[1]-clearance))
        mask = x_mask & y_mask
    elif shape == 'cylinder':
        mask = np.linalg.norm(full_cloud[:,:2]-center,axis=1) <  clearance
    out = full_cloud[mask]
    out[:,:2] -= center
    out[:,3:6] /= 65535.0
    return out

