import os
import os.path as osp
import errno

from laspy.file import File
import numpy as np
import pandas as pd

import argparse
import sys
from io import StringIO
import functools

import logging



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



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_logger(logFilename):
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s-%(levelname)s-%(message)s',
        datefmt='%y-%m-%d %H:%M',
        filename=logFilename,
        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def check_dirs(dirname):
    if osp.isdir(dirname) is False:
        os.makedirs(dirname)

# ```A decorator for parse string from print function ```

def get_print_string(func):
    @functools.wraps(func)
    def wrapper(*args, **kw):
        old_std_out = sys.stdout
        new_std_out = StringIO()
        sys.stdout = new_std_out
        func(*args, **kw)
        sys.stdout = old_std_out
        return new_std_out.getvalue().strip()
    return wrapper

@get_print_string
def get_string_from_print(*args, **kw):
    print(*args, **kw)

def ktprint(*args, **kw):    
    logging.info(get_string_from_print(*args, **kw))
