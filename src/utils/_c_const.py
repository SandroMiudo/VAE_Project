import utils._c_types as _c_types

_c_strat_pad : _c_types.Strategy = 'pad'
_c_strat_skip : _c_types.Strategy = 'skip'

_c_strat_max_pooling = 'max_pool'
_c_strat_avg_pooling = 'avg_pool'

_c_strat_relu    = 'relu_feature_map'
_c_strat_l_relu  = 'l_relu_feature_map'
_c_strat_sigmoid = 'sigmoid_feature_map'
_c_strat_tanh    = 'tanh_feature_map'

_c_32_t = 0xFFFFFFFF
_c_16_t = 0xFFFF
_c_8_t  = 0xFF