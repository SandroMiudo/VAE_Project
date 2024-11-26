import matplotlib.pyplot as plt
from matplotlib import colors
import torch
from typing import Tuple, Mapping
import itertools
import operator

def plot2D(x: torch.Tensor, /, *, rows:int, cols:int, shape:Tuple[int, int, int],
           show=True, title=''):
    fig, axes = plt.subplots(rows, cols, squeeze=False)
    fig.suptitle(title)

    _c = 0
    for i in range(rows):
        for j in range(cols):
            _x = x[_c].numpy()
            assert _x.size == list(itertools.accumulate(shape, operator.mul))[-1]
            axes[i,j].imshow(_x.reshape(shape), cmap='gray', vmin=0, vmax=255)
            _c += 1
    if show:
        plt.show()

def visualize_data(x: Mapping[str, torch.Tensor], t: str, /, *, clip=500_000):
    plt.figure(figsize=(15, 5))
    for tensor in x.values():
        count, bins = torch.histogram(tensor.to("cpu"))
        count, bins, _ = plt.hist(bins[:-1].numpy(), 
                                        bins=bins.numpy(), 
                                        weights=count.numpy())
    plt.title(f"{t} look up")
    plt.xlabel(f"{t} values")
    plt.ylabel("Frequency")
    plt.ylim((0, clip))
    plt.show()