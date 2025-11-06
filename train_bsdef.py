import os

import torch

import logging
from utils.utils import create_logger

from problems.ou import OU
from problems.bistable import Bistable
from problems.schlogel import Schlogel

from BSDEF.bsdef import BSDEF
from utils.normalizer import Normalizer

default_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.set_device(default_device)

##############################
### Problem specifications ###

problem = Bistable()

T_0 = 0
T_N = 1
n_obs = 10
n_steps = 16

bsdef_params = {
    'fixed_input_size': True,
    'reverse_obs_order': True, # Always puts the latest observation first
    'u_params': {
        'num_layers': 5,
        'hidden_dim': 128,
    },
    'v_params': {
        'num_layers': 5,
        'hidden_dim': 32
    }
}

normalizer_params = {
    'method': "Quad", # Quad, I-EKF or I-G
    'x_min': -5, # If quad
    'x_max': 5, # If quad
    'n_points': 1000, # If quad
    'n_samples': 10000, # If Importance
}

trainer_params = {
    'epochs': 1,
    'samples_per_epoch': 5*256,
    'batch_size': 256,
    'batch_size_obs': 64, # Use several observations per state value -> quicker normalization
    'copy_prev_network': True,
    'early_stopping': True,
    'early_stopping_patience': 50,
    'save_folder': "Test_bsdef"
}

optimizer_params = {
        'lr': 1e-4, 
        'weight_decay': 1e-8,
        'lr_final': 1e-6      # this is used for a CosineSchedulerer, which starts at 'lr' and 
}                             # changes over the epochs towards 'lr_final'. set 'lr_final' to 
                              # same value as 'lr' to have constant learning rate
logger_params = {
    'log_file': {
        'filename': 'run_log'
    }
}

### Specifications end ###
##########################

def main():
    logger_params['folder'] = os.path.join("BSDEF", "saved_models", trainer_params['save_folder'])
    create_logger(**logger_params)

    _print_config()

    timepoints = torch.linspace(T_0, T_N, n_obs*n_steps + 1)
    obs_indices = torch.arange(n_steps, len(timepoints), n_steps)

    normalizer_params['problem'] = problem
    normalizer_params['times'] = timepoints
    normalizer_params['obs_indices'] = obs_indices

    normalizer = Normalizer(normalizer_params)

    filter = BSDEF(bsdef_params, problem, timepoints, obs_indices, normalizer)
    filter.train(trainer_params, optimizer_params) # Automatically saves models after they are trained

def _print_config():
    logger = logging.getLogger('root')
    logger.info('CUDA_DEVICE_NUM: {}'.format(default_device))
    logger.info('Problem identifier: {}'.format(problem.identifier))
    logger.info('T0: {}'.format(T_0))
    logger.info('TN: {}'.format(T_N))
    logger.info('n_obs: {}'.format(n_obs))
    logger.info('n_steps: {}'.format(n_steps))
    logger.info('bsde_params: {}'.format(bsdef_params))
    logger.info('trainer_params: {}'.format(trainer_params))
    logger.info('optimizer_params: {}'.format(optimizer_params))

##########################################################################################

if __name__ == "__main__":
    main()
