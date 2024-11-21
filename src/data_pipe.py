from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Sampler
from typing import Tuple, Sequence, Callable
import numpy as np
import torch
import random
from utils._c_const import _c_strat_pad, _c_strat_skip
from utils._c_types import Properties, Globals, NdArrayLike, MapLike
import h5py
import utils.utilty as util

class __C_Data__:
    _d_group = dict()
    _d_all   = dict()
    _l_all   = list()
    _l_group = dict()

    def __init__(self, name, mode):
        self._f = h5py.File(name, mode)
        self._X = len(self._f)
        self._keys_rep = list(self._f.keys())
        self._n = [len(self._f[x]) for x in self._keys_rep]
        self._min = min(self._n)
        self._max = max(self._n)
        self._total = sum(self._n)
        
        self._f.visititems(__C_Data__.fill)

    def key(self, idx):
        return self._keys_rep[idx]
    
    @property
    def keys(self):
        return self._keys_rep
    
    @property
    def min_len(self):
        return self._min
    
    @property
    def max_len(self):
        return self._max
    
    @property
    def group_len(self):
        return self._n

    def __len__(self):
        return self._X

    def extract(self, exclude=None | Sequence[int]):
        if isinstance(exclude, Sequence) and \
            all(_ex_idx < len(self) for _ex_idx in exclude):
            return [[self._f[self._keys_rep[i]][x][()] 
                   for x in self._f[self._keys_rep[i]]] 
                   for i in range(self._X) if i not in exclude]

        return [[self._f[self._keys_rep[i]][x][()] 
                   for x in self._f[self._keys_rep[i]]] 
                   for i in range(self._X)]
    
    def reduce(self, image_count, exclude=None | Sequence[int]):
        if isinstance(exclude, Sequence) and \
            all(_ex_idx < len(self) for _ex_idx in exclude):
            image_count = min([self._n[i] for i in range(len(self._n)) if i not in exclude])
            return [[self._f[self._keys_rep[i]][list(self._f[self._keys_rep[i]].keys())[x]][()] 
                   for x in range(image_count)] 
                   for i in range(self._X) if i not in exclude]
        else:
            image_count = min(self._min, image_count)
            return [[self._f[self._keys_rep[i]][list(self._f[self._keys_rep[i]].keys())[x]][()] 
                   for x in range(image_count)] 
                   for i in range(self._X)]

    def global_explore(self, not_idx:Sequence[int]) -> Globals:
        _x = self.extract(not_idx)

        _total = sum([self._n[i] for i in range(self._X) if i not in not_idx])

        _g_std  = []
        _g_max  = []
        _g_min  = []
        _g_mean = [] 

        for _group in _x:
            for _group_i in _group:
                _g_max.append(np.max(_group_i))
                _g_min.append(np.min(_group_i))
                _g_mean.append(np.mean(_group_i))
                _g_std.append(np.std(_group_i))

        return (max(_g_max), min(_g_min), sum(_g_std) / _total, sum(_g_mean) / _total)

    def data_search(self):
        _min_images = []
        _max_images = []

        for i in range(self._X):
            _min_img_size = min(__C_Data__._d_group[self._keys_rep[i]].keys())
            _max_img_size = max(__C_Data__._d_group[self._keys_rep[i]].keys())
            
            _min_img_f = 0
            _max_img_f = 0

            for x in self._f[self._keys_rep[i]]:
                x = self._f[self._keys_rep[i]][x]
                if x.shape[0] * x.shape[1] == _min_img_size and _min_img_f == 0:
                    _min_images.append(x[()])
                    _min_img_f = 1
                if x.shape[0] * x.shape[1] == _max_img_size and _max_img_f == 0:
                    _max_images.append(x[()])
                    _max_img_f = 1

        return (_min_images, _max_images)

    def group_distribution(self):
        _d_group_sorted = [sorted(list(__C_Data__._d_group[self._keys_rep[i]].keys())) 
                           for i in range(self._X)]
        _d_group_p = [[__C_Data__._d_group[self._keys_rep[i]][x] / self._n[i] 
                       for x in _d_group_sorted[i]] for i in range(self._X)]

        return (_d_group_sorted, _d_group_p)

    def single_group_distribution(self, group):
        assert group < self._n

        _d_group_sorted = sorted(list(__C_Data__._d_group[self._keys_rep[group]].keys())) 
                           
        _d_group_p = [__C_Data__._d_group[self._keys_rep[group]][x] / self._n[group] 
                       for x in _d_group_sorted]
        
        return (_d_group_sorted, _d_group_p)

    def distribution(self):
        _d_all_sorted = sorted(list(__C_Data__._d_all.keys()))
        _d_all_p = [__C_Data__._d_all[x] / self._total for x in _d_all_sorted]

        return (_d_all_sorted, _d_all_p)

    @staticmethod
    def fill(name, item):
        if(hasattr(item, 'shape')):
            name = name.split('/', 1)[0]
        
        if name not in __C_Data__._d_group:
            __C_Data__._d_group[name] = dict()
            __C_Data__._l_group[name] = []

        _l_e = __C_Data__._l_group[name]
        _d = __C_Data__._d_group[name]

        if(not hasattr(item, 'shape')): # filter out groups
            return
        
        def all_based(item, _d):
            _img_size = item.shape[0]*item.shape[1]
            __C_Data__._l_all.append(_img_size)
            _l_e.append(_img_size)
            if _img_size in _d:  
                _d[_img_size] = _d[_img_size] + 1
            else:
                _d[_img_size] = 1

        all_based(item, _d)
        all_based(item, __C_Data__._d_all)

class DataDescriptor:
    def __init__(self, name):
        self.name = name

    def __get__(self, instance, owner=None):
        return (self.name, self.props)

    def __set__(self, instance, value):
        self.props = len(instance._X[value])

class __C_DataSet__(Dataset):
    #x1 = DataDescriptor("b1")
    #x2 = DataDescriptor("b2")
    #x3 = DataDescriptor("b3")
    #x4 = DataDescriptor("b4")
    #x5 = DataDescriptor("b5")

    def __init__(self, x, p:Properties):
        super().__init__()
        self._X = x
        #_data_descr_l = [self.x1, self.x2, self.x3, self.x4, self.x5]
        #for i in range(self._X):
        #    _data_descr_l[i] = i
        self._prop = p
        
    @property
    def dims(self):
        a = ((x, len(self._X[x])) for x in range(len(self._X)))

        _d = dict({"groups" : len(self._X), "group" : {}})
        for _entry in a:
            _d["group"][_entry[0]] = _entry[1]

        return _d
        
    def details(self, d_brain_x, d_point_x):
        return (len(self._X[d_brain_x][d_point_x]), len(self._X[d_brain_x][d_point_x][0]))

    def __getitem__(self, idx):
        assert isinstance(idx, Tuple) and len(idx) == 5 \
            and all(isinstance(x, int) for x in idx)
        
        _brain, _image, _row, _column, _tile_s = idx
        _data_prop = self.dims

        assert _data_prop["groups"] > _brain and _data_prop["group"][_brain] > _image and \
                len(self._X[_brain][_image]) > _row and len(self._X[_brain][_image][0]) > _column
        
        _l_img   = len(self._X[_brain][_image])
        _l_img_d = len(self._X[_brain][_image][0])
        _pad  = ((((_row + _tile_s) % _l_img) + 1) * ((_row  + _tile_s) // _l_img), 
                 (((_column + _tile_s) % _l_img_d) + 1) * ((_column + _tile_s) // _l_img_d))

        _r = np.asarray(
            self._X[_brain][_image][_row:_row+(_tile_s-_pad[0]), _column:_column+(_tile_s-_pad[1])], dtype='f4')
        
        if _pad[0] > 0:
            _r = np.vstack((_r, np.zeros((_pad[0], _tile_s))), dtype='f4')
        if _pad[1] > 0:    
            _r = np.hstack((_r, np.zeros((_tile_s, _pad[1]))), dtype='f4')

        _mean, _std = self._prop
        _standardized_data = (_r - _mean) / _std

        _standardized_data = _standardized_data[..., np.newaxis]
        _standardized_data = np.reshape(_standardized_data, 
                   (1, _standardized_data.shape[0], _standardized_data.shape[1]))

        return torch.from_numpy(_standardized_data)
    
    def __len__(self):
        return sum(self.dims["group"].values())
    
    def map(self, map_fnct:Callable[[NdArrayLike, MapLike], 
                                     NdArrayLike], **kw) -> '__C_DataSet__':
        _dims = self.dims
        _new_X = [[map_fnct(self._X[i][j], **kw) for j in range(_dims["group"][i])] 
                  for i in range(_dims["groups"])]

        return __C_DataSet__(_new_X, self._prop) # we have to adjust the properties
        
    def concat(self, dataset:'__C_DataSet__') -> '__C_DataSet__':
        for i in range(len(self._X)):
            self._X[i].extend(dataset._X[i])
        return self

    def augment_seq(self, map_fncts:Sequence[Callable[[NdArrayLike, MapLike], 
                                                       NdArrayLike]], **kw) -> '__C_DataSet__':
        _new_X = [self.map(map_fnct, **kw) for map_fnct in map_fncts]

        _d = self
        for x in _new_X:
            _d = _d.concat(x)
        return _d
    
class __C_DataSampler__(Sampler):

    # stategy resolves to either pad=default or skip
    # pad allows each row | column to be selected and 
    # pads the border accordingly with zeros
    # skip on the other side, just gets row | column 
    # that do not extend the bound

    def __init__(self, strategy, data, tile_size, samples):
        super().__init__()
        self._strategy = strategy
        self._data = data
        self._tile_size = tile_size
        self._samples = samples

    def __iter__(self):
        _brains = self._data.dims["groups"]
        for _ in range(self._samples):
            _brain_i = random.randint(0, _brains-1)
            _images  = self._data.dims["group"][_brain_i]
            _image_i = random.randint(0, _images-1)
            _rows, _cols = self._data.details(_brain_i, _image_i)

            if self._strategy == _c_strat_skip:
                _row_i = random.randint(0, _rows-1-self._tile_size)
                _col_i = random.randint(0, _cols-1-self._tile_size)

            elif self._strategy == _c_strat_pad:
                _row_i = random.randint(0, _rows-1)
                _col_i = random.randint(0, _cols-1)

            yield (_brain_i, _image_i, _row_i, _col_i, self._tile_size)

    def __len__(self):
        return self._samples

class __C_DataLoader__(DataLoader):
    def __init__(self, dataset, sampler, batch_size, worker_fnct:Callable[[int], None]=None):
        cpu_physical = util.cpu_count()
        cpu_logical  = util.cpu_count_logical()
        print(f"Cpu physical => {cpu_physical}")
        print(f"Cpu logical  => {cpu_logical}")
        super().__init__(dataset, batch_size, sampler=sampler, num_workers=cpu_physical,
                         worker_init_fn=worker_fnct)