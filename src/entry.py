from model import __C_VariatonalEncoder__, l1_regulazation_loss, l2_regulazation_loss
from data_pipe import __C_Data__, __C_DataLoader__, \
    __C_DataSampler__, __C_DataSet__
import utils._c_const as _c_const
from matplotlib import pyplot as plt
import torch.optim as opt
import torch
import os
import numpy as np
from utils import utilty
from sklearn.decomposition import PCA

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
tilesize = 156

max, min, std, mean = _c_data.global_explore([_test_idx])
max_t, min_t, std_t, mean_t = _c_data.global_explore(
    [x for x in range(len(_c_data)) if x != _test_idx])

print(f"maximum => {max} , minimum => {min}\nmean => {mean:.2f} , std => {std:.2f}")

#_train = _c_data.reduce(20, [_test_idx])
_train = _c_data.extract([_test_idx])
_test  = _c_data.extract([x for x in range(len(_c_data)) if x != _test_idx])

_dataset = __C_DataSet__(_train, (mean, std))
_datasampler = __C_DataSampler__(_c_const._c_strat_skip, _dataset, tilesize, 1000)
_dataloader  = __C_DataLoader__(_dataset, _datasampler, 64)
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

vae = __C_VariatonalEncoder__((1, tilesize, tilesize), 128, 3, 3, 4, 32,
                        _c_const._c_strat_max_pooling, _c_const._c_strat_l_relu)

def print_function(line):
    print(line)

vae.architecture(print_function)
vae.summary(print_function)
vae = vae.to(_device)

vae.c_link(opt.Adam(vae.parameters(), 1e-3), _device)
vae.downstream_link(None, _device)

vae.c_train(_dataloader, 10, alpha=0.9, beta=1, gamma=1e-6,
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

x_prime = vae.c_decode(10)

fig, axes = plt.subplots(2, 5)

_c = 0
for i in range(2):
    for j in range(5):
        axes[i,j].imshow(x_prime[_c].numpy().reshape(tilesize, tilesize, 1))
        _c += 1

plt.show()

_src_dir = os.path.dirname(__file__)
_tgt_dir = os.path.join(_src_dir, "..", "model", "saved_model.pt")

_model_dict = {**vae.get_config(), **vae.get_linked_config()}

torch.save({**_model_dict, 
            "model" : vae.__class__
            }, _tgt_dir)

_config = torch.load(_tgt_dir, _device, weights_only=False)
_reconstructed_model = _config["model"].from_config(_config)
_reconstructed_model.from_linked_config(_config)

_reconstructed_model.eval()

_dataset = __C_DataSet__(_test, (mean_t, std_t))
_datasampler = __C_DataSampler__(_c_const._c_strat_skip, _dataset, tilesize, 3000)
_dataloader  = __C_DataLoader__(_dataset, _datasampler, 8)

_reg_1 = _dataset[(0, 10, 50, 10, tilesize)]
_reg_2 = _dataset[(0, 50, 50, 10, tilesize)]
_sec_1 = _dataset[(0, 5, 15, 5, tilesize)]
_sec_2 = _dataset[(0, 5, 40, 20, tilesize)]

_reg_1 = _reg_1[None, :]
_reg_2 = _reg_2[None, :]
_sec_1 = _sec_1[None, :]
_sec_2 = _sec_2[None, :]

_interpolation_steps = 10

latent_v_mu_reg_1, latent_v_x_reg_1 = vae.c_encode(_reg_1)
latent_v_mu_reg_2, latent_v_x_reg_2 = vae.c_encode(_reg_2)
latent_v_mu_sec_1, latent_v_x_sec_1 = vae.c_encode(_sec_1)
latent_v_mu_sec_2, latent_v_x_sec_2 = vae.c_encode(_sec_2)
latent_repr_reg_1 = vae.latent_repr(latent_v_mu_reg_1, latent_v_x_reg_1)
latent_repr_reg_2 = vae.latent_repr(latent_v_mu_reg_2, latent_v_x_reg_2)
latent_repr_sec_1 = vae.latent_repr(latent_v_mu_sec_1, latent_v_x_sec_1)
latent_repr_sec_2 = vae.latent_repr(latent_v_mu_sec_2, latent_v_x_sec_2)

interpolated_reg = np.asarray([(1-alpha)*latent_repr_reg_1.numpy() + alpha*latent_repr_reg_2.numpy() 
                for alpha in np.linspace(0, 1, _interpolation_steps)])
interpolated_sec = np.asarray([(1-alpha)*latent_repr_sec_1.numpy() + alpha*latent_repr_sec_2.numpy() 
                for alpha in np.linspace(0, 1, _interpolation_steps)])

_latent_repr_interpolated_reg = torch.from_numpy(interpolated_reg)
_latent_repr_interpolated_sec = torch.from_numpy(interpolated_reg)

_latent_repr_interpolated_reg = torch.squeeze(_latent_repr_interpolated_reg)
_latent_repr_interpolated_sec = torch.squeeze(_latent_repr_interpolated_sec)

_interpolated_images_reg = vae.c_decode_from_latent(_latent_repr_interpolated_reg)
_interpolated_images_sec = vae.c_decode_from_latent(_latent_repr_interpolated_sec)

fig, axes = plt.subplots(2, 5)

_c = 0
for i in range(2):
    for j in range(5):
        axes[i,j].imshow(_interpolated_images_reg[_c].numpy().reshape(tilesize, tilesize, 1))
        _c += 1

plt.show()

fig, axes = plt.subplots(2, 5)

_c = 0
for i in range(2):
    for j in range(5):
        axes[i,j].imshow(_interpolated_images_sec[_c].numpy().reshape(tilesize, tilesize, 1))
        _c += 1

plt.show()

_latent_representations = []

for x in _dataloader:
    latent_v_mu, latent_v_x = vae.c_encode(x)
    latent_repr = vae.latent_repr(latent_v_mu, latent_v_x)
    _latent_representations.append(latent_repr.numpy())

_flatten_latent_repr = utilty.flatten(_latent_representations, [])

pca = PCA(2)
dim_redc = pca.fit_transform(_flatten_latent_repr)

plt.plot(dim_redc[:, 0], dim_redc[:, 1], '.b')
plt.show()