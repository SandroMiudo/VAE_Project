from . import _c_types
from . import _c_const
import psutil
import torch
from typing import TypeVar
from typing import List
import random

def size(n) -> _c_types._size:
    assert isinstance(n, (int))

    return n & _c_const._c_32_t

E = TypeVar('E')
T = TypeVar('T', bound=List[E])

def flatten(l:List[T], f:T) -> T:
    for _l in l:
        f.extend(_l)

    return f

def sample_uniform(l:List[E], /):
    return l[random.randint(0, len(l)-2)]

def re_scale(x: torch.Tensor, /, *, std, mean) -> torch.Tensor:
    return (x * std) + mean

def cpu_count():
    #return len(os.sched_getaffinity(0)) # 0 -> current process, does return logical cpus somehow
    return psutil.cpu_count(logical=False)

def cpu_count_logical():
    return psutil.cpu_count()