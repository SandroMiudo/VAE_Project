from typing import Iterator
import torch.nn as nn
from utils._c_types import Strategy
import torch.optim as opt
import utils.utilty as util
import utils._c_const as _c_const
import torch
from typing import Callable, Tuple, Mapping
from utils import plotter

def l1_regulazation_loss(scale, parameters):
        return scale * sum(torch.sum(torch.abs(wj)) for wj in parameters)

def l2_regulazation_loss(scale, parameters):
    return scale * sum(torch.sum(torch.square(wj)) for wj in parameters)

def latent_log_variance(latent_v_mu, latent_log_var):
    return latent_v_mu + (torch.exp(0.5 * latent_log_var) * 
                          torch.randn_like(latent_log_var))

def latent_std(latent_v_mu, latent_std):
    return latent_v_mu + (latent_std * torch.randn_like(latent_std))

# formula KL(y,y*) = 1/2 * (sigmaO**2 / sigma1**2 + ((mu1 - mu0) ** 2 / sigma1**2) - 1 + log2(sigma1**2/sigma0**2))
# variants => avg over L_KL or sum over L_KL

def latent_loss_log_variance(latent_v_mu, latent_log_var):
    return (1 / latent_v_mu.shape[1]) * torch.sum(0.5 * (torch.exp(latent_log_var) + torch.square(latent_v_mu) - 1 \
           - latent_log_var), 1)

def latent_loss_std(latent_v_mu, latent_std):
    return (1 / latent_v_mu.shape[1]) * torch.sum(0.5 * (torch.square(latent_std) + torch.square(latent_v_mu) - 1 \
           - torch.log(torch.square(latent_std))), 1)

class __C_Module__():
    
    def c_link(self, optimizer:opt.Optimizer, device:torch.device):
        self._optimizer = optimizer
        self._device    = device

    def c_loss(self):
        pass

    def c_train(self, x):
        pass

    def c_train_step(self, x):
        pass

    def architecture(self, f:Callable[[str], None]):
        pass

    def summary(self, f:Callable[[str], None]):
        pass

    def get_config(self):
        pass

    def from_config(cls, config):
        pass

    def get_linked_config(self):
        pass

    def from_linked_config(self, config):
        pass

    "Activates debug mode -> i.e registers hooks for activations & gradients"
    def debug_mode(self):
        pass

class __C_BlockBuilder__():
    def __init__(self, blocks, input_dim, filter, kernel,
                 x_feature_map:Strategy):
        assert blocks > 0
        self._blocks = nn.ModuleList()
        for i in range(blocks):
            if i == 0:
                _conv_i = nn.Conv2d(input_dim, filter, (kernel,kernel), (1,1), 
                                padding='valid')
            else:
                _conv_i = nn.Conv2d(filter, filter, (kernel,kernel), (1,1), 
                                padding='valid')
            self._blocks.append(_conv_i)
            self._blocks.append(nn.BatchNorm2d(filter))

            match x_feature_map:
                case _c_const._c_strat_relu:
                    _act_i = nn.ReLU()
                case _c_const._c_strat_l_relu:
                    _act_i = nn.LeakyReLU()
                case _c_const._c_strat_sigmoid:
                    _act_i = nn.Sigmoid()
                case _c_const._c_strat_tanh:
                    _act_i = nn.Tanh()
                case _:
                    raise Exception()

            self._blocks.append(_act_i)

    def fetch(self) -> nn.ModuleList:
        return self._blocks
    
    def out_shape(self, initial_dim:tuple[int, int]):
        _len = len(self._blocks) // 3 # exclude activations, batch norm
        _strides = self._blocks[0].stride
        _kernel  = self._blocks[0].kernel_size
        _red_0 = initial_dim[0] - (((initial_dim[0] - _kernel[0]) // _strides[0]) + 1)
        _red_1 = initial_dim[1] - (((initial_dim[1] - _kernel[1]) // _strides[1]) + 1)

        return (initial_dim[0] - (_red_0 * _len), initial_dim[1] - (_red_1 * _len))

    def block_entry(self, idx):
        idx = util.size(idx)
        assert idx < len(self._blocks)

        return self._blocks[idx]

class __C_VariatonalEncoder__(nn.Module, __C_Module__):
    def __init__(self, input_dim, latent_dim, blocks, depth, kernel, filters,
                 downsampling:Strategy, x_feature_map:Strategy, *,
                 latent_repr:str='log_var'):
        assert blocks > 0 and depth > 0
        super().__init__()
        _c_encoder =__C_Encoder__(input_dim, latent_dim,
                                      depth, blocks, filters, kernel,
                                      downsampling, x_feature_map)
        self._sub_modules = nn.ModuleDict({
            'encoder' : _c_encoder,
            'decoder' : __C_Decoder__(_c_encoder.modules(), 
                                      _c_encoder.state_shape[0], _c_encoder.state_shape[1:])
        })
        self._latent_dim = latent_dim
        self._input_dim  = input_dim
        self._blocks = blocks
        self._depth = depth
        self._kernel = kernel
        self._filters = filters
        self._downsampling_strat = downsampling
        self._x_feature_map_strat = x_feature_map
        self._built = False
        
        if latent_repr == 'log_var':
            self._latent_fnct = latent_log_variance
            self._latent_loss_fnct = latent_loss_log_variance
        elif latent_repr == 'std':
            self._latent_fnct = latent_std
            self._latent_loss_fnct = latent_loss_std
        else:
            raise Exception("No valid latent representation!")

    @property
    def built(self):
        return self._built
    
    @built.setter
    def built(self, status:bool):
        self._built = status

    def forward(self, x):
        self.built = True
        latent_v_mu, latent_repr = self._sub_modules["encoder"](x)
        
        x = self.latent_repr(latent_v_mu, latent_repr)

        return (latent_v_mu, latent_repr, self._sub_modules["decoder"](x))

    def c_train_step(self, x, /, *, alpha, beta, gamma,
                     loss_regulizer:Callable[[float, Iterator[nn.Parameter]], float]):
        self._optimizer.zero_grad()
        latent_v_mu, latent_v_std, x_prime = self(x)

        _loss:torch.Tensor = self.loss(x, latent_v_mu, latent_v_std, x_prime,
                                       alpha, beta, gamma, loss_regulizer)

        _loss = _loss.to(self._device)

        _loss.backward()

        self._optimizer.step()

        return _loss.item()

    # alpha : scale reconstruction loss
    # beta  : scale distribution loss
    # gamma : scale regulazation loss
    # loss_regulizer : regulazation function to use
    def c_train(self, x, /, epoch, *, alpha=1, beta=1, gamma=1, 
                loss_regulizer:Callable[[float, Iterator[nn.Parameter]], None]=l1_regulazation_loss,
                debug_mode: str | bool=False, debug_start:int=1):
        _debug_modes = ["batch", "epoch"]

        if isinstance(debug_mode, str) and debug_mode not in _debug_modes:
            raise ValueError()
        if isinstance(debug_mode, bool) and debug_mode:
            debug_mode = _debug_modes[1]
        
        if debug_mode:
            activations, gradients = self._register_hookups()

        self.train()
        for ep in range(epoch):
            _total_loss = 0
            for _batch in x:
                _batch = _batch.to(self._device)
                _total_loss += self.c_train_step(_batch, alpha=alpha, beta=beta,
                                                 gamma=gamma, loss_regulizer=loss_regulizer)
                if debug_mode and debug_mode == _debug_modes[0] and debug_start <= ep + 1:
                    plotter.visualize_data(activations, "Activation")
                    plotter.visualize_data(gradients, "Gradient")
            if debug_mode and debug_mode == _debug_modes[1] and debug_start <= ep + 1:
                plotter.visualize_data(activations, "Activation")
                plotter.visualize_data(gradients, "Gradient")
            
            print(f"Loss for ep={ep+1} => {_total_loss / len(x):.2f}")

    def latent_repr(self, latent_v_mu, latent_v_repr):
        return self._latent_fnct(latent_v_mu, latent_v_repr)

    def c_inference(self, x):
        self.eval()

        x = x.to(self._device)

        with torch.no_grad():
            latent_v_mu, latent_v_std, x_prime = self(x)

        return (latent_v_mu.to("cpu"), latent_v_std.to("cpu"), x_prime.to("cpu"))

    def c_encode(self, x):
        self.eval()

        x = x.to(self._device)

        with torch.no_grad():
            latent_v_mu, latent_v_std = self._sub_modules["encoder"](x)

        return (latent_v_mu.to("cpu"), latent_v_std.to("cpu"))

    def c_decode(self, samples):
        self.eval()

        _rands = torch.randn((samples, self._latent_dim))
        _rands_i = torch.randn((samples, *self._input_dim))

        _rands = _rands.to(self._device)
        _rands_i = _rands_i.to(self._device)

        with torch.no_grad():
                self._sub_modules["encoder"](_rands_i) # update pooling indices
                x_prime = self._sub_modules["decoder"](_rands)

        return x_prime.to("cpu")
    
    def c_decode_from_latent(self, latent_reprs:torch.Tensor):
        self.eval()

        _rands_i = torch.randn((latent_reprs.shape[0], *self._input_dim))

        _rands_i = _rands_i.to(self._device)
        _latent_reprs = latent_reprs.to(self._device)

        with torch.no_grad():
            self._sub_modules["encoder"](_rands_i) # update pooling indices
            x_prime = self._sub_modules["decoder"](_latent_reprs)

        return x_prime.to("cpu")

    def loss(self, x, latent_vector_mean, latent_vector_repr, x_prime,
             alpha, beta, gamma, regulizer_fnct):
        _mse     = nn.MSELoss(reduction='mean')
        _reconstruction_loss    = _mse(x, x_prime)
        
        _latent_loss = self._latent_loss_fnct(latent_vector_mean, latent_vector_repr)

        _regulazation_loss = regulizer_fnct(gamma, self.parameters())

        _latent_loss = _latent_loss.mean()

        _total_loss = alpha*_reconstruction_loss + _regulazation_loss \
                    + beta*_latent_loss

        return _total_loss
    
    def downstream_link(self, opt, device):
        for _sub_module in self._sub_modules.values():
            _sub_module.c_link(opt, device)

    def _sub_module_call(self, sub_mod, inner_list_name=str):
        _l = len(list(sub_mod.get_submodule(inner_list_name)))
        _info = []
        for _idx in range(_l):
            _sub_mod = sub_mod.get_submodule(f"{inner_list_name}.{_idx}")
            _sub_mod_params = sum([torch.numel(param) for param in _sub_mod.parameters()])
            _info.append((_sub_mod, _sub_mod_params))

        return _info

    def _module_call(self, _encoder, _decoder):
        _l_enc = self._sub_module_call(_encoder, "_flatten_module_list")
        _l_dec = self._sub_module_call(_decoder, "_module_list")
        
        return (_l_enc, _l_dec)

    def _iterate_over_info(self, info, *,
                           info_call:Callable[[Tuple[nn.Module, int, int]], None]):
        for idx, x in enumerate(info):
            info_call(x[0], x[1], idx)

    def architecture(self, f:Callable[[str], None]):
        _encoder = self._sub_modules["encoder"]
        _decoder = self._sub_modules["decoder"]

        _l_enc = len(list(_encoder.get_submodule("_flatten_module_list")))
        _l_dec = len(list(_decoder.get_submodule("_module_list")))
        
        _info_enc, _info_dec = self._module_call(_encoder, _decoder)
        f("\t-----")
        f(f"\t|VAE| : Total Modules => {_l_enc + _l_dec}")
        f("\t-----")

        def info_func_arch(x, y, idx):
            f(f"\t==>Module : {idx} => " \
              f"{x}")

        f("\t---------")
        f(f"\t|Encoder| : Total Modules => {_l_enc}")
        f("\t---------")
        self._iterate_over_info(_info_enc, info_call=info_func_arch)

        f("\t---------")
        f(f"\t|Decoder| : Total Modules => {_l_dec}")
        f("\t---------")
        self._iterate_over_info(_info_dec, info_call=info_func_arch)

    def summary(self, f:Callable[[str], None]):
        _encoder = self._sub_modules["encoder"]
        _decoder = self._sub_modules["decoder"]

        if not self.built:
            x = torch.randn([1, *self._input_dim]) 
            self(x)
        
        _info_enc, _info_dec = self._module_call(_encoder, _decoder)

        _total_param_enc = sum(x[1] for x in _info_enc)
        _total_param_dec = sum(x[1] for x in _info_dec)
        f("\t-----")
        f(f"\t|VAE| : Total Parameters => {_total_param_enc + _total_param_dec}")
        f("\t-----")
        def info_func_arch(x, y, idx):
            f(f"\t==> Module : {idx} => " \
             f"{x} *** Params => {y}")

        f("\t---------")
        f(f"\t|Encoder| : Total Parameters => {_total_param_enc}")
        f("\t---------")
        self._iterate_over_info(_info_enc, info_call=info_func_arch)

        f("\t---------")
        f(f"\t|Decoder| : Total Parameters => {_total_param_dec}")
        f("\t---------")
        self._iterate_over_info(_info_dec, info_call=info_func_arch)

    def get_config(self):
        return {
            "input_dim" : self._input_dim,
            "latent_dim" : self._latent_dim,
            "blocks" : self._blocks,
            "depth" : self._depth,
            "kernel" : self._kernel,
            "filters" : self._filters,
            "downsampling" : self._downsampling_strat,
            "x_feature_map" : self._x_feature_map_strat,
            "latent_repr" : 'log_var' if self._latent_fnct == latent_log_variance
                else 'std'
        }

    @classmethod
    def from_config(cls, config):
        _input_dim = config.pop("input_dim")
        _latent_dim = config.pop("latent_dim")
        _blocks = config.pop("blocks")
        _depth = config.pop("depth")
        _kernel = config.pop("kernel")
        _filters = config.pop("filters")
        _downsampling_strat = config.pop("downsampling")
        _x_feature_map_strat = config.pop("x_feature_map")
        _latent_repr = config.pop("latent_repr")

        return cls(_input_dim, _latent_dim, _blocks, _depth, _kernel,
                   _filters, _downsampling_strat, _x_feature_map_strat,
                   latent_repr=_latent_repr)
    
    def get_linked_config(self):
        return {
            "model_state_dict" : self.state_dict(),
            "opt_state_dict" : self._optimizer.state_dict(),
            "optimizer" : self._optimizer.__class__,
            "device" : self._device.type
        }

    def from_linked_config(self, config):
        self.load_state_dict(config["model_state_dict"])
        _optimizer = config["optimizer"](self.parameters())
        _device    = torch.device(config["device"])
        _optimizer.load_state_dict(config["opt_state_dict"])
        
        self.c_link(_optimizer, _device)

    def _register_hookups(self) -> Tuple[Mapping[str, torch.Tensor], Mapping[str, torch.Tensor]]:
        activations = dict()
        gradients   = dict()

        def register_activation(name):
            def hook(module, args, output):
                activations[name] = output.detach()
            return hook

        def register_gradient(name):
            def hook(_, grad_input, grad_output):
                name_i = name + "wrt in"
                name_j = name + "wrt out"
                if grad_input[0] != None:
                    gradients[name_i] = grad_input[0].detach() # passing w & b seperatly
                gradients[name_j] = grad_output[0].detach() # passing w & b seperatly
            return hook
        
        for name, module in self.named_modules():
            if hasattr(module, "weight") and module.get_parameter("weight").requires_grad or \
               hasattr(module, "bias") and module.get_parameter("bias").requires_grad:
                module.register_full_backward_hook(register_gradient(name))
            if isinstance(module, (nn.ReLU, nn.LeakyReLU, nn.Tanh, nn.Sigmoid)):
                module.register_forward_hook(register_activation(name))

        return activations, gradients

class __C_Encoder__(nn.Module, __C_Module__):
    def __init__(self, input_dim, latent_dim, depth, blocks, filter,
                    kernel, downsampling:Strategy, x_feature_map:Strategy):
        super().__init__()

        _channels = input_dim[0]
        _spatial  = input_dim[1:]
        _filter   = filter
        
        _layers = []
        _down_sampling_spatial = [] # needed to reconstruct input

        for _ in range(depth):
            _block = __C_BlockBuilder__(blocks, _channels, _filter, kernel,
                            x_feature_map)
            _layers.append(_block.fetch())
            _spatial = _block.out_shape(_spatial)
            _down_sampling_spatial.append(_spatial)
            match downsampling:
                case _c_const._c_strat_avg_pooling:
                    _pool = nn.AvgPool2d((2,2))
                case _c_const._c_strat_max_pooling:
                    _pool = nn.MaxPool2d((2,2), return_indices=True)
                case _:
                    pass
            _layers.append(nn.ModuleList([_pool]))
            _spatial = (((_spatial[0] - _pool.kernel_size[0]) // _pool.stride[0]) + 1,
                        ((_spatial[1] - _pool.kernel_size[1]) // _pool.stride[1]) + 1)
            _channels = _filter
            _filter *= 2

        self._flatten_module_list = util.flatten(_layers, nn.ModuleList())
        
        self._state_shape = (_channels, *_spatial)

        self._flatten_module_list.append(nn.Flatten())
        self._flatten_module_list.append(
            nn.Linear((_spatial[0]*_spatial[1]*_channels), latent_dim*2))
        self._flatten_module_list.append(nn.LeakyReLU())
        self._flatten_module_list.append(
            nn.Linear(latent_dim*2, latent_dim))
        # self._flatten_module_list.append(nn.ReLU())
        self._flatten_module_list.append(
            nn.Linear(latent_dim*2, latent_dim))
        #self._flatten_module_list.append(nn.ReLU())

        __C_Decoder__._down_sampling_spatial = list(reversed(_down_sampling_spatial))

    def modules(self) -> Iterator[nn.Module]:
        return self._flatten_module_list

    @property
    def state_shape(self):
        return self._state_shape

    def forward(self, x):
        _indices = []
        for i in range(len(self._flatten_module_list)-2):
            if isinstance(self._flatten_module_list[i], nn.MaxPool2d):
                x, indices = self._flatten_module_list[i](x)
                _indices.append(indices)
            else:
                x = self._flatten_module_list[i](x)
        _x_f = self._flatten_module_list[-2](x)
        _x_s = self._flatten_module_list[-1](x)
        __C_Decoder__._pooling_indices = list(reversed(_indices))
        return (_x_f, _x_s)
    
class __C_Decoder__(nn.Module, __C_Module__):
    def __init__(self, enc_module_list, filter, spatial):
        super().__init__()
        _rv_mod_list = list(reversed(enc_module_list))
        self._module_list = nn.ModuleList()
        _channels = filter
        for i in range(len(_rv_mod_list)):
            if isinstance(_rv_mod_list[i], (*_c_const._c_activations, nn.BatchNorm2d)):
                continue
            _mod = _rv_mod_list[i]

            if isinstance(_mod, nn.Linear):
                self._module_list.append(nn.Linear(_mod.out_features, _mod.in_features))
            elif isinstance(_mod, nn.Flatten):
                self._module_list.append(
                    nn.Unflatten(1, (_channels, spatial[0], spatial[1])))
            elif isinstance(_mod, (nn.AvgPool2d, nn.MaxPool2d)):
                self._module_list.append(
                    nn.MaxUnpool2d((2,2)))
                #self._module_list.append(
                #    nn.ConvTranspose2d(_channels, _channels, (2,2), (2,2)))
                _channels //= 2
            elif isinstance(_mod, nn.Conv2d):
                self._module_list.append(
                    nn.ConvTranspose2d(_mod.out_channels, _mod.in_channels,
                                       _mod.kernel_size, _mod.stride))
                
            if i == len(_rv_mod_list) - 1:
                break

            if isinstance(_rv_mod_list[i-1], nn.BatchNorm2d):
                self._module_list.append(nn.BatchNorm2d(_mod.in_channels))
            
            if isinstance(_rv_mod_list[i-1], _c_const._c_activations):
                self._module_list.append(self._apply_activation(i-1, _rv_mod_list))
            elif isinstance(_rv_mod_list[i-2], _c_const._c_activations):
                self._module_list.append(self._apply_activation(i-2, _rv_mod_list))

        self._module_list = self._module_list[1:] # cut out first layer (sub latent)

    def _apply_activation(self, idx, rev_mod_list):
        match type(rev_mod_list[idx]):
            case nn.ReLU:
                _mod_actv = nn.ReLU()
            case nn.LeakyReLU:
                _mod_actv = nn.LeakyReLU() 
            case nn.Sigmoid:
                _mod_actv = nn.Sigmoid()
            case nn.Tanh:
                _mod_actv = nn.Tanh()
            case _:
                pass

        return _mod_actv

    def forward(self, x):
        _c = 0
        for i in range(len(self._module_list)):
            if isinstance(self._module_list[i], nn.MaxUnpool2d):
                x = self._module_list[i](x, __C_Decoder__._pooling_indices[_c],
                                         __C_Decoder__._down_sampling_spatial[_c])
                _c += 1
            else:
                x = self._module_list[i](x)
        return x