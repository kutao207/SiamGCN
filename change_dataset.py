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
from torch import Tensor
from torch_sparse import SparseTensor, cat
from torch.utils.data.dataloader import default_collate
from torch._six import container_abcs, string_classes, int_classes

import torch_geometric
from torch_geometric.data import DataLoader, Data, InMemoryDataset, Batch

from utils import load_las, extract_area
from utils import makedirs, files_exist, to_list, find_file

from imblearn.over_sampling import RandomOverSampler

class ChangeBatch(Batch):
    def __init__(self, batch=None, ptr=None, **kwargs):
        super().__init__(batch=batch, ptr=ptr, **kwargs)
    
    @classmethod
    def from_data_list(cls, data_list, follow_batch, exclude_keys):
        r'''
        datalist: A list object with `batch_size` elements, and each element is a `Data` object.
        '''
        keys = list(set(data_list[0].keys) - set(exclude_keys)) # ['y', 'x', 'x2', 'scene_num']
        assert 'batch' not in keys and 'ptr' not in keys

        batch = cls()
        for key in data_list[0].__dict__.keys(): 
            # ['x', 'edge_index', 'edge_attr', 'y', 'pos', 'normal', 'face', 'x2', 'scene_num']
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['batch'] + ['batch2']:
            batch[key] = []
        batch['ptr'] = [0]
        batch['ptr2'] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys: # keys: ['y', 'x', 'x2', 'scene_num']
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Treat 0-dimensional tensors as 1-dimensional.
                if isinstance(item, Tensor) and item.dim() == 0:
                    item = item.unsqueeze(0)

                batch[key].append(item)

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                cat_dims[key] = cat_dim
                if isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                    device = item.device()

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size, ), i, dtype=torch.long,
                                           device=device))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size, ), i, dtype=torch.long,
                                       device=device))

            if hasattr(data, '__num_nodes__'):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            num_nodes = data.num_nodes
            if num_nodes is not None:
                item = torch.full((num_nodes, ), i, dtype=torch.long,
                                  device=device)
                batch.batch.append(item)
                batch.ptr.append(batch.ptr[-1] + num_nodes)
            num_nodes2 = data.x2.size(0)
            if num_nodes2 is not None:
                item = torch.full((num_nodes2, ), i, dtype=torch.long,
                                  device=device)
                batch.batch2.append(item)
                batch.ptr2.append(batch.ptr2[-1] + num_nodes2)

        # Fix initial slice values:
        for key in keys:
            slices[key][0] = slices[key][1] - slices[key][1]

        batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.batch2 = None if len(batch.batch2) == 0 else batch.batch2
        batch.ptr2 = None if len(batch.ptr2) == 1 else batch.ptr2
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, ref_data.__cat_dim__(key, item))
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()

class ChangeCollater(object):
    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, Data):
            return ChangeBatch.from_data_list(batch, self.follow_batch,
                                        self.exclude_keys)
        elif isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int_classes):
            return torch.tensor(batch)
        elif isinstance(elem, string_classes):
            return batch
        elif isinstance(elem, container_abcs.Mapping):
            return {key: self.collate([d[key] for d in batch]) for key in elem}
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, container_abcs.Sequence):
            return [self.collate(s) for s in zip(*batch)]

        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)



class MyDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, follow_batch=[], exclude_keys=[], **kwargs):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for Pytorch Lightning...
        self.follow_batch = follow_batch

        super().__init__(dataset, batch_size, shuffle,
                             collate_fn=ChangeCollater(follow_batch,
                                                 exclude_keys), **kwargs)


def __repr__(obj):
    if obj is None:
        return 'None'
    return re.sub('(<.*?)\\s.*(>)', r'\1\2', obj.__repr__())

class ChangeDataset(InMemoryDataset):
    def __init__(self, root, train=True, clearance = 3, ignore_labels=[], transform=None, pre_transform=None, pre_filter=None):        
        self.root = root
        self.train = train
        self.clearance = clearance
        self.ignore_labels = ignore_labels
        self.data_2016 = osp.join(root, '2016')
        self.data_2020 = osp.join(root, '2020')
        self.train_csv_dir = osp.join(root, 'train')

        self.class_labels = ['nochange','removed',"added",'change',"color_change"]

        if len(self.ignore_labels) > 0:
            rm_labels = []
            for l in self.ignore_labels:
                if l in self.class_labels:
                    self.class_labels.remove(l)
                    rm_labels.append(l)
            print(f"Labels {rm_labels} have been removed!")
            if len(self.ignore_labels) == 0:
                raise ValueError("All labels have been ignored!!")

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
        torch.save(self.process_set('train'), self.processed_paths[0])        
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

        f = osp.join(self.processed_dir, 'label_names_'+ f'{self.clearance}'+ '.pt')
        if osp.exists(f) and torch.load(f) != '_'.join(self.class_labels):
            logging.warning('The `class_labels` argument differs from last used one. You may have ignored some class names.')

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

        path = osp.join(self.processed_dir, 'label_names_'+ f'{self.clearance}'+ '.pt')
        torch.save('_'.join(self.class_labels), path)

        print('Done!')

    def process_set(self, dataset):        
        csv_files = sorted(glob.glob(osp.join(self.root, dataset, '*.csv')))
        files_2016 = glob.glob(osp.join(self.data_2016, '*.laz'))
        files_2020 = glob.glob(osp.join(self.data_2020, '*.laz'))

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
            
            df = pd.read_csv(file)
            centers = df[["x", "y", "z"]].to_numpy()
            label_names = df["classification"].to_list()
            # labels = self.names_to_labels(label_names)

            points_16, h16 = load_las(f16)
            points_20, h20 = load_las(f20)

            i+=1
            print(f"Processing {dataset} set {i}/{len(csv_files)} --> {osp.basename(file)} scene_num={scene_num} finding {len(centers)} objects")

            for center, label in zip(centers, label_names):

                if label in self.ignore_labels:
                    continue
                else:
                    label = self.names_to_labels_dict[label]
                
                x1 = torch.tensor(extract_area(points_16, center[0:2], self.clearance), dtype=torch.float)                
                data = Data(x=x1)
                data.x2 = torch.tensor(extract_area(points_20, center[0:2], self.clearance), dtype=torch.float)
                data.y = torch.tensor([label])
                data.scene_num = torch.tensor([int(scene_num)])
                data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

if __name__ == '__main__':

    from transforms import NormalizeScale, SamplePoints
    pre_transform, transform = NormalizeScale(), SamplePoints(1024)

    root_dir = 'F:/shrec2021/data'

    train_dataset = ChangeDataset(root_dir, train=True, clearance=3, transform=transform, pre_transform=pre_transform)
    test_dataset = ChangeDataset(root_dir, train=False, clearance=3, transform=transform, pre_transform=pre_transform)

    print("Dataset finished!")

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)

    # for data in train_loader:
    #     print(data.x.shape)


