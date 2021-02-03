import os
import os.path as osp
import shutil
import glob
import copy
import re
import logging

from itertools import repeat, product
import errno

import numpy as np
import pandas as pd

import torch
import torch.utils.data

from torch_geometric.data import DataLoader, Data, InMemoryDataset


from utils import load_las, extract_area
from utils import makedirs, files_exist, to_list, find_file


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

class ChangeDataset(InMemoryDataset):
    def __init__(self, root, train=True, clearance = 3, transform=None, pre_transform=None, pre_filter=None):        
        self.root = root
        self.train = train
        self.clearance = clearance
        self.data_2016 = osp.join(root, '2016')
        self.data_2020 = osp.join(root, '2020')
        self.train_csv_dir = osp.join(root, 'train')

        self.class_labels = ['nochange','removed',"added",'change',"color_change","unfit"]
        self.labels_to_names_dict = {i:v for i, v in enumerate(self.class_labels)}
        self.names_to_labels_dict = {v:i for i, v in enumerate(self.class_labels)}

        super().__init__(root, transform, pre_transform, pre_filter)

        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)


    def labels_to_names(self, labels):
        return [self.class_labels[i] for i in labels]

    def names_to_labels(self, names):
        return [self.names_to_labels_dict[n] for n in names]

    @property
    def processed_file_names(self):
        # return ['training.pt', 'test.pt']
        return ['training_'+ f'{self.clearance}'+ '.pt', 'test_'+ f'{self.clearance}'+ '.pt']
    
    def process(self):
        if self.train is True:
            torch.save(self.process_set('train'), self.processed_paths[0])
        else:
            torch.save(self.process_set('test'), self.processed_paths[1])

    def _process(self):
        f = osp.join(self.processed_dir, 'pre_transoform_'+ f'{self.clearance}'+ '.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_transform):
            logging.warning(
                'The `pre_transform` argument differs from the one used in '
                'the pre-processed version of this dataset. If you really '
                'want to make use of another pre-processing technique, make '
                'sure to delete `{}` first.'.format(self.processed_dir))
        f = osp.join(self.processed_dir, 'pre_filter_'+ f'{self.clearance}'+ '.pt')
        if osp.exists(f) and torch.load(f) != __repr__(self.pre_filter):
            logging.warning(
                'The `pre_filter` argument differs from the one used in the '
                'pre-processed version of this dataset. If you really want to '
                'make use of another pre-fitering technique, make sure to '
                'delete `{}` first.'.format(self.processed_dir))

        path = self.processed_paths[0] if self.train else self.processed_paths[1]
        # if files_exist(self.processed_paths):  # pragma: no cover
        #     return
        if osp.exists(path):  # pragma: no cover
            return

        print('Processing...')

        makedirs(self.processed_dir)

        self.process()

        path = osp.join(self.processed_dir, 'pre_transform_'+ f'{self.clearance}'+ '.pt')
        torch.save(__repr__(self.pre_transform), path)
        path = osp.join(self.processed_dir, 'pre_filter_'+ f'{self.clearance}'+ '.pt')
        torch.save(__repr__(self.pre_filter), path)

        print('Done!')

    def process_set(self, dataset):
        data_2016 = osp.join(self.root, '2016')
        data_2020 = osp.join(self.root, '2020')
        csv_files = sorted(glob.glob(osp.join(self.root, dataset, '*.csv')))
        files_2016 = glob.glob(osp.join(data_2016, '*.laz'))
        files_2020 = glob.glob(osp.join(data_2020, '*.laz'))

        data_list = []
        i = 0
        for file in csv_files:
            # scene_num = re.findall(r'^[0-9]', osp.basename(file))

            scene_num = osp.basename(file).split('_')[0]

            # if len(scene_num) == 0:
            #     continue
            # else:
            #     scene_num = scene_num[0]

            f16 = find_file(files_2016, scene_num)
            f20 = find_file(files_2020, scene_num)

            print(f"Processing {i+1}/{len(csv_files)} --> {osp.basename(file)} scene_num={scene_num}")
            i+=1

            df = pd.read_csv(file)
            centers = df[["x", "y", "z"]].to_numpy()
            label_names = df["classification"].to_list()
            labels = self.names_to_labels(label_names)

            points_16, h16 = load_las(f16)
            points_20, h20 = load_las(f20)

            for center, label in zip(centers, labels):
                
                x1 = torch.tensor(extract_area(points_16, center[0:2], self.clearance))                
                data = Data(x=x1)
                data.x2 = torch.tensor(extract_area(points_20, center[0:2], self.clearance))
                data.y = torch.tensor([label])
                data.scene_num = torch.tensor([int(scene_num)])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

if __name__ == '__main__':
    root_dir = 'F:/shrec2021/data'

    train_dataset = ChangeDataset(root_dir, train=True)
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)

    for data in train_loader:
        print(data.x.shape)


