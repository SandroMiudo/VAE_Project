import matplotlib.pyplot as plt
import torch
from typing import Tuple
import itertools
import operator

def plot2D(x: torch.Tensor, /, *, rows:int, cols:int, shape:Tuple[int, int, int],
           show=True):
    assert x.shape() == (rows, cols)

    _, axes = plt.subplots(rows, cols, squeeze=False)

    _c = 0
    for i in range(rows):
        for j in range(cols):
            _x = x[_c].numpy()
            assert _x.size == itertools.accumulate(shape, operator.mul)[-1]
            axes[i,j].imshow(_x.reshape(shape))
            _c += 1

    if show:
        plt.show()