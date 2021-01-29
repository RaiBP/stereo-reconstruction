import json
import numpy as np
import os

bm_param_list = None
sgbm_param_list = None

if not os.path.exists('./bm_parameters'):
    os.makedirs('./bm_parameters')


def load_param_list(param_list, params_path):
    with open(params_path) as f:
        params = json.load(f)

    if param_list is None:
        param_list = {}
        for key in params.keys():
            param_list[key] = [params[key]]
    else:
        for key in params.keys():
            param_list[key].append(params[key])
    return param_list


for img_number in range(1, 6):
    bm_param_list = load_param_list(
        bm_param_list, f'./bm_parameters/parameters_{img_number}_stereo-bm.json')
    sgbm_param_list = load_param_list(
        sgbm_param_list, f'./bm_parameters/parameters_{img_number}_stereo-sgbm.json')


default_bm_params = {
    'min_disp': 2 * int(np.mean(bm_param_list['min_disp']) / 2),
    'num_disp': 16 * int(np.mean(bm_param_list['num_disp']) / 16),
    'block_size': 2 * int(np.mean(bm_param_list['block_size']) / 2) + 1,
    'prefilter_cap': 63,
    'prefilter_size': 255,
    'disp12maxdiff': 255,
    'uniqueness_ratio': int(np.mean(bm_param_list['uniqueness_ratio'])),
    'speckle_windows_size': int(np.mean(bm_param_list['speckle_windows_size'])),
    'speckle_range': int(np.mean(bm_param_list['speckle_range'])),
    'texture_threshold': 255,
    'use_xsobel': True,
    'wls_filtering': False,
}

with open(f'./bm_parameters/parameters_default_stereo-bm.json', 'w') as outfile:
    json.dump(default_bm_params, outfile)

default_sgbm_params = {
    'min_disp': 2 * int(np.mean(sgbm_param_list['min_disp']) / 2),
    'num_disp': 16 * int(np.mean(sgbm_param_list['num_disp']) / 16),
    'block_size': 2 * int(np.mean(sgbm_param_list['block_size']) / 2) + 1,
    'p1': int(np.mean(sgbm_param_list['p1'])),
    'p2': int(np.mean(sgbm_param_list['p2'])),
    'prefilter_cap': 63,
    'disp12maxdiff': -1,
    'uniqueness_ratio': int(np.mean(sgbm_param_list['uniqueness_ratio'])),
    'speckle_windows_size': int(np.mean(sgbm_param_list['speckle_windows_size'])),
    'speckle_range': int(np.mean(sgbm_param_list['speckle_range'])),
    'use_dynamic_programming': '3way',
    'wls_filtering': False,
}

with open(f'./bm_parameters/parameters_default_stereo-sgbm.json', 'w') as outfile:
    json.dump(default_sgbm_params, outfile)
