import os

import torch

import logging
from utils.utils import create_logger

from problems.ou import OU
from problems.bistable import Bistable

from LogBSDEF.logbsdef import LogBSDEF
from utils.normalizer import Normalizer

from benchmark_filters.kalman_filter import KalmanFilter
from benchmark_filters.particle_filter import ParticleFilter
from benchmark_filters.extended_kalman_filter import ExtendedKalmanFilter
from benchmark_filters.ensemble_kalman_filter import EnsembleKalmanFilter

from utils.metric_evaluator import MetricEvaluator

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
n_steps = 32

n_steps_reference = 128
n_steps_benchmarks = 1

model_type = "FCN" # FCN or LSTM

'''
# Comment out LSTM parameters
u_encoder_params = {
    'reverse_obs_order': False, 
    'hidden_dim': 256, 
    'n_layers': 3,
    'dropout': 0.1,
    'return_cell_state': True
}

u_decoder_params = { 
    'hidden_dim': 256, 
    'n_layers': 5,
    'embed': True
}

v_encoder_params = {
    'reverse_obs_order': False, 
    'hidden_dim': 128, 
    'n_layers': 3,
    'dropout': 0.1,
    'return_cell_state': True
}

v_decoder_params = { 
    'hidden_dim': 128, 
    'n_layers': 5,
    'embed': True
}

logbsdef_params = {
    'u_encoder_params': u_encoder_params,
    'v_encoder_params': v_encoder_params,
    'u_decoder_params': u_decoder_params,
    'v_decoder_params': v_decoder_params,
}

'''

# For FCN
logbsdef_params = {
    'fixed_input_size': True,
    'reverse_obs_order': True, # Always puts the latest observation first
    'u_params': {
        'num_layers': 5,
        'hidden_dim': 256,
    },
    'v_params': {
        'num_layers': 5,
        'hidden_dim': 128
    }
}

normalizer_params = {
    'method': "I-EKF", # Quad, I-EKF or I-G
    'x_min': -5, # If quad
    'x_max': 5, # If quad
    'n_points': 1000, # If quad
    'n_samples': 1000, # If Importance
}

tester_params = {
    'model_folder': "Test_logbsdef",
    'save_folder': "Test_logbsdef", # Where to save results
    'metrics': ["MAE", "FME", "KLD", "NLL"], # MAE, FME, LinfLinf, L2Linf, KLD, NLL
    'n_samples': 1*10 ** 2, 
    "Grid-based": False, # How to calculate metrics
    "Integration range": [-5, 5], # grid-based evaluation
    "Integration points": 1 * 10 ** 3, # grid-based evaluation
    "MC samples": 1 * 10 ** 3 # grid-free evaluation
}

logger_params = {
    'log_file': {
        'filename': 'run_log'
    }
}

##########################
timepoints_reference = torch.linspace(T_0, T_N, n_obs*n_steps_reference + 1)
obs_indices_reference = torch.arange(n_steps_reference, len(timepoints_reference), n_steps_reference)
timepoints_benchmarks = torch.linspace(T_0, T_N, n_obs*n_steps_benchmarks + 1)
obs_indices_benchmarks = torch.arange(n_steps_benchmarks, len(timepoints_benchmarks), n_steps_benchmarks)
##########################

#reference = ExtendedKalmanFilter(problem, timepoints_reference, obs_indices_reference)
reference = ParticleFilter(problem, timepoints_reference, obs_indices_reference, 10000)
reference_name = "pf_10000"

# Benchmark filters
pf_100 = ParticleFilter(problem, timepoints_benchmarks, obs_indices_benchmarks, 100)
pf_1000 = ParticleFilter(problem, timepoints_benchmarks, obs_indices_benchmarks, 1000)
#pf_10000 = ParticleFilter(problem, timepoints_benchmarks, obs_indices_benchmarks, 10000)
ekf = ExtendedKalmanFilter(problem, timepoints_benchmarks, obs_indices_benchmarks)
enkf_100 = EnsembleKalmanFilter(problem, timepoints_benchmarks, obs_indices_benchmarks, 100)

benchmark_filters = {
    "pf_100": pf_100,
    #"pf_1000": pf_1000,
    #"pf_10000": pf_10000,
    "ekf": ekf,
    "enkf_100": enkf_100
}

### Specifications end ###
#########################################################################################

def main():
    logger_params['folder'] = os.path.join("results", tester_params['save_folder'])
    create_logger(**logger_params)

    _print_config()

    filters = benchmark_filters.copy()

    timepoints = torch.linspace(T_0, T_N, n_obs*n_steps + 1)
    obs_indices = torch.arange(n_steps, len(timepoints), n_steps)
    
    normalizer_params['problem'] = problem
    normalizer_params['times'] = timepoints
    normalizer_params['obs_indices'] = obs_indices

    if model_type == "LSTM":
        normalizer_params['encode_all_first'] = True

    normalizer = Normalizer(normalizer_params)

    filter = LogBSDEF(model_type, logbsdef_params, problem, timepoints, obs_indices, normalizer)

    for model_index in range(n_obs):
        filter.load_checkpoint(filter.models[model_index], model_index, tester_params['model_folder'])
            
    filters["logBSDEF"] = filter

    evaluator = MetricEvaluator(tester_params, filters, reference, problem)

    evaluator.evaluate_metrics(tester_params['n_samples'])
    evaluator.save_metric_results(tester_params['save_folder'])


def _print_config():
    logger = logging.getLogger('root')
    logger.info('CUDA_DEVICE_NUM: {}'.format(default_device))
    logger.info('Deep Filter: {}'.format("LogBSDEF"))
    logger.info('Problem identifier: {}'.format(problem.identifier))
    logger.info('T0: {}'.format(T_0))
    logger.info('TN: {}'.format(T_N))
    logger.info('n_obs: {}'.format(n_obs))
    logger.info('n_steps_reference: {}'.format(n_steps_reference))
    logger.info('n_steps_benchmarks: {}'.format(n_steps_benchmarks))
    logger.info('reference_name: {}'.format(reference_name))
    logger.info('benchmarks: {}'.format(benchmark_filters.keys()))
    logger.info('logbsdef_params: {}'.format(logbsdef_params))
    logger.info('tester_params: {}'.format(tester_params))

##########################################################################################

if __name__ == "__main__":
    main()


