from model import __C_VariatonalEncoder__, l1_regulazation_loss, l2_regulazation_loss
from data_pipe import __C_Data__, __C_DataLoader__, \
    __C_DataSampler__, __C_DataSet__
import utils._c_const as _c_const
from matplotlib import pyplot as plt
import torch.optim as opt
import torch
import os

_c_data = __C_Data__("../data/cell_data.h5", 'r')

def task_1():
    plt.plot(_c_data.keys, _c_data.group_len, '.b')
    plt.ylim((_c_data.min_len-10, _c_data.max_len+10))
    plt.xlabel('groups')
    plt.ylabel('images')
    for i in range(len(_c_data)):
        plt.plot([_c_data.key(i), _c_data.key(i)], 
                [_c_data.min_len-10, _c_data.group_len[i]], ':b')
    plt.show()

    fig = plt.figure()
    gs  = fig.add_gridspec(2, len(_c_data))

    ax1 = fig.add_subplot(gs[0, :])

    x,y = _c_data.distribution()
    x_g, y_g = _c_data.group_distribution()

    ax1.stairs(y[:-1], x, baseline=0)

    for i in range(len(_c_data)):
        if i > 0:
            ax_i = fig.add_subplot(gs[1, i], sharex=fig.get_axes()[i], sharey=fig.get_axes()[i])
        else:
            ax_i = fig.add_subplot(gs[1, i])
        ax_i.stairs(y_g[i][:-1], x_g[i], baseline=0)   

    plt.show()

    _min_images, _max_images = _c_data.data_search()

    fig, ax = plt.subplots(2, len(_c_data))

    for i in range(len(_c_data)):
        ax[0, i].imshow(_min_images[i])
        ax[1, i].imshow(_max_images[i])

    plt.show()

##########################
# choose group 4 as test #
##########################

_test_idx = 3
tilesize = 128

max, min, std, mean = _c_data.global_explore(_test_idx)

print(f"maximum => {max} , minimum => {min}\nmean => {mean:.2f} , std => {std:.2f}")

_train = _c_data.reduce(20, [_test_idx])
# _train = _c_data.extract(_test_idx)

_dataset = __C_DataSet__(_train, (mean, std))
_datasampler = __C_DataSampler__(_c_const._c_strat_skip, _dataset, tilesize, 1000)
_dataloader  = __C_DataLoader__(_dataset, _datasampler, 8)
_n_e = next(iter(_dataloader))

fig, axes = plt.subplots(2, 4)

_c = 0
for i in range(2):
    for j in range(4):
        axes[i,j].imshow(_n_e[_c].numpy().reshape(tilesize, tilesize, 1))
        _c += 1

plt.show()

x = torch.cuda.is_available()
z = torch.cuda.device_count()

print(f"Gpu available : {x} => Total devices : {z}")

_device = torch.device("cuda:0") if x else torch.device("cpu")
_type   = _device.type

print(f"Device selected => {_type}")

vae = __C_VariatonalEncoder__((1, tilesize, tilesize), 256, 2, 3, 4, 32,
                        _c_const._c_strat_max_pooling, _c_const._c_strat_relu)

def print_function(line):
    print(line)

vae.architecture(print_function)
vae.summary(print_function)
vae = vae.to(_device)

vae.c_link(opt.Adam(vae.parameters(), 1e-3), _device)
vae.downstream_link(None, _device)

vae.c_train(_dataloader, 5, alpha=2, beta=0.8, gamma=1e-5,
            loss_regulizer=l2_regulazation_loss)

_n_e = next(iter(_dataloader))

print("Recreating images")

latent_v_mu, latent_v_std, x_prime = vae.c_inference(_n_e)

fig, axes = plt.subplots(2, 4)

_c = 0
for i in range(2):
    for j in range(4):
        axes[i,j].imshow(x_prime[_c].numpy().reshape(tilesize, tilesize, 1))
        _c += 1

plt.show()

print("Sampling from model")

x_prime = vae.c_decode(8)

fig, axes = plt.subplots(2, 4)

_c = 0
for i in range(2):
    for j in range(4):
        axes[i,j].imshow(x_prime[_c].numpy().reshape(tilesize, tilesize, 1))
        _c += 1

plt.show()

_src_dir = os.path.dirname(__file__)
_tgt_dir = os.path.join(_src_dir, "..", "model", "saved_model.pt")

_model_dict = {**vae.get_config(), **vae.get_linked_config()}

print(_model_dict)

torch.save({**_model_dict, 
            "model" : vae.__class__
            }, _tgt_dir)

_config = torch.load(_tgt_dir, _device, weights_only=False)
_reconstructed_model = _config["model"].from_config(_config)
_reconstructed_model.from_linked_config(_config)

_reconstructed_model.eval()