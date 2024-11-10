from . import _c_types
from . import _c_const

from typing import TypeVar
from typing import List

def size(n) -> _c_types._size:
    assert isinstance(n, (int))

    return n & _c_const._c_32_t

E = TypeVar('E')
T = TypeVar('T', bound=List[E])

def flatten(l:List[T], f:T) -> T:
    for _l in l:
        f.extend(_l)

    return f