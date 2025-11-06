import os
import sys

import torch

from logging import getLogger
from utils.utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from LogBSDEF.logbsdef_model_fcn import LogBSDEFModelFCN
from LogBSDEF.logbsdef_model_lstm import LogBSDEFModelLSTM
from LogBSDEF.logbsdef_trainer import LogBSDEFTrainer
from utils.problem_simulator import ProblemSimulator

from torch.optim import Adam as Optimizer

class LogBSDEF():
    def __init__(self, model_type, params, problem, times, obs_indices, normalizer):
        self.problem = problem
        self.times = times
        self.normalizer = normalizer

        self.simulator = ProblemSimulator(problem, times, obs_indices)

        self.n_obs = len(obs_indices)
        self.obs_indices = obs_indices

        self.params = params
        self.params['n_obs_max'] = self.n_obs - 1

        self.model_type = model_type

        self.models = []
        for n_obs in range(self.n_obs):
            if n_obs == 0:
                model_times = self.times[0:obs_indices[n_obs] + 1]
            else: 
                model_times = self.times[obs_indices[n_obs-1]:obs_indices[n_obs] + 1]
            if model_type == "FCN":
                self.models = self.models + [LogBSDEFModelFCN(self.params, problem, n_obs, model_times)]
            elif model_type == "LSTM":
                self.models = self.models + [LogBSDEFModelLSTM(self.params, problem, n_obs, model_times)]

    def train(self, trainer_params, optimizer_params):
        self.logger = getLogger(name='trainer')
        
        for i, model in enumerate(self.models):
            self.logger.info(f"Training model {i + 1}:")

            optimizer = Optimizer(model.parameters(), lr = optimizer_params['lr'], weight_decay = optimizer_params['weight_decay'])

            schedule_max = 80 
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                        optimizer, 
                                        T_max = schedule_max, 
                                        eta_min= optimizer_params['lr_final'])
            
            if trainer_params['copy_prev_network']:
                if i > 0:
                    model.load_state_dict(self.models[i-1].state_dict())
                    
            if i == 0:
                prev_model = model
            else:
                prev_model = self.models[i-1]
            
            model.train()
            trainer = LogBSDEFTrainer(trainer_params, self.simulator, model, prev_model, optimizer, scheduler, self.normalizer, self.problem)
            trainer.train_model()

            self.save_checkpoint(model, optimizer, i, trainer_params['save_folder'])

        self.logger.info('=================================================================')
        self.logger.info("***Training Done***")
            

    def filter(self, obs):
        pass

    def evaluate_logfilter_pdf(self, obs_idx, x, obs):
        """
        Evaluates the logarithmic filter pdf (unnormalized) in x given obs
        x: (B, d)
        obs: (n_obs, d')
        return: (B)
        """
        model = self.models[obs_idx]
        model.eval()

        obs_spec = obs[:(model.n_obs+1), :]
        obs_spec_exp = obs_spec.unsqueeze(0).expand(x.shape[0], (model.n_obs+1), -1) 

        with torch.no_grad():
            logfilter_unnorm = model.logfilter_unnorm(x, obs_spec_exp)
            return logfilter_unnorm.view(-1)

    def get_filter_pdfs(self, obs, device, x_min = -5, x_max = 5, n_points = 1000):
        """
        Returns the filter pdfs evaluated on a grid for all observation times
        obs: (n_obs, d')
        returns: (n_obs, n_points)
        """

        points = [torch.linspace(x_min, x_max, n_points) for _ in range(self.problem.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.ravel() for dimension in mesh]).T

        dx = ((x_max - x_min) / (n_points - 1))**self.problem.state_dim
        n_points_tot = n_points ** self.problem.state_dim
        density = torch.zeros((self.n_obs, n_points_tot))

        for i, model in enumerate(self.models):
            model.eval()

            with torch.no_grad():
                obs_spec = obs[:(model.n_obs+1), :]
                obs_spec_exp = obs_spec.unsqueeze(0).expand(n_points_tot, (model.n_obs+1), -1) 

                filter_unnorm = model.filter_unnorm(x, obs_spec_exp)
                filter_norm = filter_unnorm / (torch.sum(filter_unnorm) * dx)
                density[i, :] = filter_norm[:, 0]
                
        return density, x
    
    def get_filter_means(self, obs, device, x_min = -5, x_max = 5, n_points = 1000):
        """
        Returns the filter means for all observation times (evaluation via quadrature)
        obs: (n_obs, d')
        returns: (n_obs, d)
        """

        points = [torch.linspace(x_min, x_max, n_points) for _ in range(self.problem.state_dim)]
        mesh = torch.meshgrid(*points, indexing='ij')
        x = torch.vstack([dimension.ravel() for dimension in mesh]).T

        n_points_tot = n_points ** self.problem.state_dim
        means = torch.zeros((self.n_obs, self.problem.state_dim))

        for i, model in enumerate(self.models):
            model.eval()

            with torch.no_grad():
                obs_spec = obs[:(model.n_obs+1), :]
                obs_spec_exp = obs_spec.unsqueeze(0).expand(n_points_tot, (model.n_obs+1), -1) 

                filter_unnorm = model.filter_unnorm(x, obs_spec_exp)
                filter_mean = torch.sum(x * filter_unnorm, dim=0) / (torch.sum(filter_unnorm))
                means[i, :] = filter_mean
                
        return means

    def get_filter_means_and_logconsts(self, obs):
        """
        Returns the normalization constants and means for all observation times
        obs: (n_obs, d')
        returns: (n_obs, 1) normalization constants and (n_obs, d) means
        """
        lognorm_consts = torch.zeros((self.n_obs, 1))
        means = torch.zeros((self.n_obs, self.problem.state_dim))

        for i, model in enumerate(self.models):
            model.eval()

            with torch.no_grad():
                obs_spec = obs[:(model.n_obs+1), :]
                if model.model_type == "FCN":
                    lognorm_const, mean = self.normalizer.get_log_nc(model.logfilter_unnorm, obs = obs_spec.unsqueeze(0), get_fm=True)
                elif model.model_type == "LSTM":
                    lognorm_const, mean = self.normalizer.get_log_nc(model.logfilter_unnorm_dec, obs = obs_spec.unsqueeze(0), encode=model.encode, get_fm=True)
                means[i, :] = mean
                lognorm_consts[i, :] = lognorm_const
                
        return lognorm_consts, means

    def save_checkpoint(self, model, optimizer, model_index, folder_name):
        folder_path = os.path.join(current_dir, "saved_models", folder_name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        checkpoint = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }

        filepath = os.path.join(folder_path, f"Model_states_{model_index + 1}")
        torch.save(checkpoint, filepath)

    def load_checkpoint(self, model, model_index, folder_name, optimizer=None):
        folder_path = os.path.join(current_dir, "saved_models", folder_name)
        if not os.path.exists(folder_path):
            raise ValueError("The specified directory does not exist. Cannot load models")

        filepath = os.path.join(folder_path, f"Model_states_{model_index + 1}")

        checkpoint = torch.load(filepath)
        
        model.load_state_dict(checkpoint['model_state_dict'])        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])