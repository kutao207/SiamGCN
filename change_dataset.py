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

from torch.utils.data.dataset import IterableDataset
from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler, BatchSampler
from torch.utils.data.dataloader import _InfiniteConstantSampler,_SingleProcessDataLoaderIter,_MultiProcessingDataLoaderIter
from torch.utils.data import _utils
import multiprocessing as python_multiprocessing
import torch.multiprocessing as multiprocessing

class _DatasetKind(object):
    Map = 0
    Iterable = 1

    @staticmethod
    def create_fetcher(kind, dataset, auto_collation, collate_fn, drop_last):
        if kind == _DatasetKind.Map:
            return _utils.fetch._MapDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)
        else:
            return _utils.fetch._IterableDatasetFetcher(dataset, auto_collation, collate_fn, drop_last)



class DataLoader(object):
    r"""
    Data loader. Combines a dataset and a sampler, and provides an iterable over
    the given dataset.

    The :class:`~torch.utils.data.DataLoader` supports both map-style and
    iterable-style datasets with single- or multi-process loading, customizing
    loading order and optional automatic batching (collation) and memory pinning.

    See :py:mod:`torch.utils.data` documentation page for more details.

    Arguments:
        dataset (Dataset): dataset from which to load the data.
        batch_size (int, optional): how many samples per batch to load
            (default: ``1``).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: ``False``).
        sampler (Sampler or Iterable, optional): defines the strategy to draw
            samples from the dataset. Can be any ``Iterable`` with ``__len__``
            implemented. If specified, :attr:`shuffle` must not be specified.
        batch_sampler (Sampler or Iterable, optional): like :attr:`sampler`, but
            returns a batch of indices at a time. Mutually exclusive with
            :attr:`batch_size`, :attr:`shuffle`, :attr:`sampler`,
            and :attr:`drop_last`.
        num_workers (int, optional): how many subprocesses to use for data
            loading. ``0`` means that the data will be loaded in the main process.
            (default: ``0``)
        collate_fn (callable, optional): merges a list of samples to form a
            mini-batch of Tensor(s).  Used when using batched loading from a
            map-style dataset.
        pin_memory (bool, optional): If ``True``, the data loader will copy Tensors
            into CUDA pinned memory before returning them.  If your data elements
            are a custom type, or your :attr:`collate_fn` returns a batch that is a custom type,
            see the example below.
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: ``False``)
        timeout (numeric, optional): if positive, the timeout value for collecting a batch
            from workers. Should always be non-negative. (default: ``0``)
        worker_init_fn (callable, optional): If not ``None``, this will be called on each
            worker subprocess with the worker id (an int in ``[0, num_workers - 1]``) as
            input, after seeding and before data loading. (default: ``None``)


    .. warning:: If the ``spawn`` start method is used, :attr:`worker_init_fn`
                 cannot be an unpicklable object, e.g., a lambda function. See
                 :ref:`multiprocessing-best-practices` on more details related
                 to multiprocessing in PyTorch.

    .. warning:: ``len(dataloader)`` heuristic is based on the length of the sampler used.
                 When :attr:`dataset` is an :class:`~torch.utils.data.IterableDataset`,
                 it instead returns an estimate based on ``len(dataset) / batch_size``, with proper
                 rounding depending on :attr:`drop_last`, regardless of multi-process loading
                 configurations. This represents the best guess PyTorch can make because PyTorch
                 trusts user :attr:`dataset` code in correctly handling multi-process
                 loading to avoid duplicate data.

                 However, if sharding results in multiple workers having incomplete last batches,
                 this estimate can still be inaccurate, because (1) an otherwise complete batch can
                 be broken into multiple ones and (2) more than one batch worth of samples can be
                 dropped when :attr:`drop_last` is set. Unfortunately, PyTorch can not detect such
                 cases in general.

                 See `Dataset Types`_ for more details on these two types of datasets and how
                 :class:`~torch.utils.data.IterableDataset` interacts with
                 `Multi-process data loading`_.
    """

    __initialized = False

    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=None,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None, multiprocessing_context=None,
                 generator=None):
        torch._C._log_api_usage_once("python.data_loader")

        if num_workers < 0:
            raise ValueError('num_workers option should be non-negative; '
                             'use num_workers=0 to disable multiprocessing.')

        if timeout < 0:
            raise ValueError('timeout option should be non-negative')

        self.dataset = dataset
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.timeout = timeout
        self.worker_init_fn = worker_init_fn
        self.multiprocessing_context = multiprocessing_context

        # Arg-check dataset related before checking samplers because we want to
        # tell users that iterable-style datasets are incompatible with custom
        # samplers first, so that they don't learn that this combo doesn't work
        # after spending time fixing the custom sampler errors.
        if isinstance(dataset, IterableDataset):
            self._dataset_kind = _DatasetKind.Iterable
            # NOTE [ Custom Samplers and `IterableDataset` ]
            #
            # `IterableDataset` does not support custom `batch_sampler` or
            # `sampler` since the key is irrelevant (unless we support
            # generator-style dataset one day...).
            #
            # For `sampler`, we always create a dummy sampler. This is an
            # infinite sampler even when the dataset may have an implemented
            # finite `__len__` because in multi-process data loading, naive
            # settings will return duplicated data (which may be desired), and
            # thus using a sampler with length matching that of dataset will
            # cause data lost (you may have duplicates of the first couple
            # batches, but never see anything afterwards). Therefore,
            # `Iterabledataset` always uses an infinite sampler, an instance of
            # `_InfiniteConstantSampler` defined above.
            #
            # A custom `batch_sampler` essentially only controls the batch size.
            # However, it is unclear how useful it would be since an iterable-style
            # dataset can handle that within itself. Moreover, it is pointless
            # in multi-process data loading as the assignment order of batches
            # to workers is an implementation detail so users can not control
            # how to batchify each worker's iterable. Thus, we disable this
            # option. If this turns out to be useful in future, we can re-enable
            # this, and support custom samplers that specify the assignments to
            # specific workers.
            if shuffle is not False:
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "shuffle option, but got shuffle={}".format(shuffle))
            elif sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "sampler option, but got sampler={}".format(sampler))
            elif batch_sampler is not None:
                # See NOTE [ Custom Samplers and IterableDataset ]
                raise ValueError(
                    "DataLoader with IterableDataset: expected unspecified "
                    "batch_sampler option, but got batch_sampler={}".format(batch_sampler))
        else:
            self._dataset_kind = _DatasetKind.Map

        if sampler is not None and shuffle:
            raise ValueError('sampler option is mutually exclusive with '
                             'shuffle')

        if batch_sampler is not None:
            # auto_collation with custom batch_sampler
            if batch_size != 1 or shuffle or sampler is not None or drop_last:
                raise ValueError('batch_sampler option is mutually exclusive '
                                 'with batch_size, shuffle, sampler, and '
                                 'drop_last')
            batch_size = None
            drop_last = False
        elif batch_size is None:
            # no auto_collation
            if drop_last:
                raise ValueError('batch_size=None option disables auto-batching '
                                 'and is mutually exclusive with drop_last')

        if sampler is None:  # give default samplers
            if self._dataset_kind == _DatasetKind.Iterable:
                # See NOTE [ Custom Samplers and IterableDataset ]
                sampler = _InfiniteConstantSampler()
            else:  # map-style
                if shuffle:
                    sampler = RandomSampler(dataset, generator=generator)
                else:
                    sampler = SequentialSampler(dataset)

        if batch_size is not None and batch_sampler is None:
            # auto_collation without custom batch_sampler
            batch_sampler = BatchSampler(sampler, batch_size, drop_last)

        self.batch_size = batch_size
        self.drop_last = drop_last
        self.sampler = sampler
        self.batch_sampler = batch_sampler
        self.generator = generator

        if collate_fn is None:
            if self._auto_collation:
                collate_fn = _utils.collate.default_collate
            else:
                collate_fn = _utils.collate.default_convert

        self.collate_fn = collate_fn
        self.__initialized = True
        self._IterableDataset_len_called = None  # See NOTE [ IterableDataset and __len__ ]

    @property
    def multiprocessing_context(self):
        return self.__multiprocessing_context

    @multiprocessing_context.setter
    def multiprocessing_context(self, multiprocessing_context):
        if multiprocessing_context is not None:
            if self.num_workers > 0:
                if not multiprocessing._supports_context:
                    raise ValueError('multiprocessing_context relies on Python >= 3.4, with '
                                     'support for different start methods')

                if isinstance(multiprocessing_context, string_classes):
                    valid_start_methods = multiprocessing.get_all_start_methods()
                    if multiprocessing_context not in valid_start_methods:
                        raise ValueError(
                            ('multiprocessing_context option '
                             'should specify a valid start method in {}, but got '
                             'multiprocessing_context={}').format(valid_start_methods, multiprocessing_context))
                    multiprocessing_context = multiprocessing.get_context(multiprocessing_context)

                if not isinstance(multiprocessing_context, python_multiprocessing.context.BaseContext):
                    raise TypeError(('multiprocessing_context option should be a valid context '
                                     'object or a string specifying the start method, but got '
                                     'multiprocessing_context={}').format(multiprocessing_context))
            else:
                raise ValueError(('multiprocessing_context can only be used with '
                                  'multi-process loading (num_workers > 0), but got '
                                  'num_workers={}').format(self.num_workers))

        self.__multiprocessing_context = multiprocessing_context

    def __setattr__(self, attr, val):
        if self.__initialized and attr in ('batch_size', 'batch_sampler', 'sampler', 'drop_last', 'dataset'):
            raise ValueError('{} attribute should not be set after {} is '
                             'initialized'.format(attr, self.__class__.__name__))

        super(DataLoader, self).__setattr__(attr, val)

    def __iter__(self):
        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @property
    def _auto_collation(self):
        return self.batch_sampler is not None

    @property
    def _index_sampler(self):
        # The actual sampler used for generating indices for `_DatasetFetcher`
        # (see _utils/fetch.py) to read data at each time. This would be
        # `.batch_sampler` if in auto-collation mode, and `.sampler` otherwise.
        # We can't change `.sampler` and `.batch_sampler` attributes for BC
        # reasons.
        if self._auto_collation:
            return self.batch_sampler
        else:
            return self.sampler

    def __len__(self):
        if self._dataset_kind == _DatasetKind.Iterable:
            # NOTE [ IterableDataset and __len__ ]
            #
            # For `IterableDataset`, `__len__` could be inaccurate when one naively
            # does multi-processing data loading, since the samples will be duplicated.
            # However, no real use case should be actually using that behavior, so
            # it should count as a user error. We should generally trust user
            # code to do the proper thing (e.g., configure each replica differently
            # in `__iter__`), and give us the correct `__len__` if they choose to
            # implement it (this will still throw if the dataset does not implement
            # a `__len__`).
            #
            # To provide a further warning, we track if `__len__` was called on the
            # `DataLoader`, save the returned value in `self._len_called`, and warn
            # if the iterator ends up yielding more than this number of samples.
            length = self._IterableDataset_len_called = len(self.dataset)
            if self.batch_size is not None:  # IterableDataset doesn't allow custom sampler or batch_sampler
                from math import ceil
                if self.drop_last:
                    length = length // self.batch_size
                else:
                    length = ceil(length / self.batch_size)
            return length
        else:
            return len(self._index_sampler)




class MyDataLoader(torch.utils.data.DataLoader):
# class MyDataLoader(DataLoader):
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
        # self.train_csv_dir = osp.join(root, 'train')

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

    data_root_path = '/home/kt/cyclomedia_disk/kt/shrec2021/data'
    if os.name == 'nt':
        data_root_path = 'F:/shrec2021/data'

    # train_dataset = ChangeDataset(root_dir, train=True, clearance=3, transform=transform, pre_transform=pre_transform)
    test_dataset = ChangeDataset(data_root_path, train=False, clearance=3, transform=transform, pre_transform=pre_transform)

    print("Dataset finished!")

    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=False, num_workers=0)

    # for data in train_loader:
    #     print(data.x.shape)


