import os
import sys

import torch

from logging import getLogger
from utils.utils import *

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from DSF.dsf_model import DSFModel
from DSF.dsf_trainer import DSFTrainer
from utils.problem_simulator import ProblemSimulator

from torch.optim import Adam as Optimizer

class DSF():
    def __init__(self, params, problem, times, obs_indices, normalizer):
        self.problem = problem
        self.times = times
        self.n_times = len(times)

        self.normalizer = normalizer

        self.simulator = ProblemSimulator(problem, times, obs_indices)

        self.n_obs = len(obs_indices)
        self.obs_indices = obs_indices

        self.N = (self.n_times - 1) // self.n_obs

        self.params = params
        if params['fixed_input_size']:
            self.params['n_obs_max'] = self.n_obs - 1

        self.models = []
        for i in range(1, self.n_times):
            n_obs = torch.sum(self.obs_indices < i)  
            self.models = self.models + [DSFModel(self.params, problem, n_obs, i)]

    def train(self, trainer_params, optimizer_params):
        self.logger = getLogger(name='trainer')
        
        for i, model in enumerate(self.models):
            obs_number = model.n_obs
            model_number = i % self.N + 1
            self.logger.info(f"Training model {obs_number}:{model_number}:")

            optimizer = Optimizer(model.parameters(), lr = optimizer_params['lr'], weight_decay = optimizer_params['weight_decay'])

            schedule_max = 50
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                                        optimizer, 
                                        T_max = schedule_max, 
                                        eta_min= optimizer_params['lr_final'], 
                                        last_epoch=-1)

            if trainer_params['copy_prev_network']:
                if i > 0:
                    model.load_state_dict(self.models[i-1].state_dict())

            model.train()
            trainer = DSFTrainer(trainer_params, self.simulator, model, optimizer, scheduler, model_number, 
                                        self.problem, self.models[i-1], self.normalizer, self.times[i + 1] - self.times[i])
            trainer.train_model()

            if (i + 1) in self.obs_indices:
                self.save_checkpoint(model, optimizer, i, trainer_params['save_folder'])

        self.logger.info('=================================================================')
        self.logger.info("***Training Done***")

    def filter(self, obs):
        pass
    
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

        obs_idx = 0

        for i, model in enumerate(self.models):
            if (i+1) in self.obs_indices:
                model.eval()

                with torch.no_grad():
                    obs_spec = obs[:model.n_obs, :]
                    obs_spec_exp = obs_spec.unsqueeze(0).expand(n_points_tot, model.n_obs, -1) 
                    obs_last = obs[model.n_obs, :].unsqueeze(0).expand(n_points_tot, -1)

                    pred = model(x, obs_spec_exp)
                    likelihood = self.problem.obs_likelihood(x, obs_last).unsqueeze(1)
                    filter_unnorm = pred * likelihood

                    filter_norm = filter_unnorm / (torch.sum(filter_unnorm) * dx)
                    density[obs_idx, :] = filter_norm[:, 0]
                    obs_idx += 1
                
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

        obs_idx = 0

        for i, model in enumerate(self.models):
            if (i+1) in self.obs_indices:
                model.eval()

                with torch.no_grad():
                    obs_spec = obs[:model.n_obs, :]
                    obs_spec_exp = obs_spec.unsqueeze(0).expand(n_points_tot, model.n_obs, -1) 
                    obs_last = obs[model.n_obs, :].unsqueeze(0).expand(n_points_tot, -1)

                    pred = model(x, obs_spec_exp)
                    likelihood = self.problem.obs_likelihood(x, obs_last).unsqueeze(1)
                    filter_unnorm = pred * likelihood

                    filter_mean = torch.sum(x * filter_unnorm, dim=0) / (torch.sum(filter_unnorm))
                    means[obs_idx, :] = filter_mean
                    obs_idx += 1
                
        return means
    
    def evaluate_filter_pdf(self, obs_idx, x, obs, normalized = True):
        """
        Evaluates the filter pdf in x given obs
        x: (B, d)
        obs: (n_obs, d')
        return: (B)
        """
        model_idx = self.obs_indices[obs_idx] - 1
        model = self.models[model_idx]
        model.eval()

        with torch.no_grad():
            obs_spec = obs[:model.n_obs + 1, :]
            obs_spec_exp = obs_spec.unsqueeze(0).expand(x.shape[0], model.n_obs + 1, -1) 

            def unnorm_density(X, obs):
                obs_last = obs[:, -1, :]
                return model(X, obs[:, :-1, :]) * self.problem.obs_likelihood(X, obs_last).view(-1, 1)

            if normalized:
                filter_norm = unnorm_density(x, obs_spec_exp) / self.normalizer.get_nc(unnorm_density, obs_spec_exp)
                return filter_norm.view(-1)
            else: 
                return unnorm_density(x, obs_spec_exp).view(-1)
            
    def get_filter_means_and_consts(self, obs):
        """
        Returns the normalization constants and means for all observation times
        obs: (n_obs, d')
        returns: (n_obs, 1) normalization constants and (n_obs, d) means
        """
        norm_consts = torch.zeros((self.n_obs, 1))
        means = torch.zeros((self.n_obs, self.problem.state_dim))

        obs_idx = 0

        for i, model in enumerate(self.models):
            if (i+1) in self.obs_indices:
                model.eval()

                with torch.no_grad():
                    obs_spec = obs[:model.n_obs + 1, :]
                    
                    def unnorm_density(X, obs):
                        obs_last = obs[:, -1, :]
                        return model(X, obs[:, :-1, :]) * self.problem.obs_likelihood(X, obs_last).view(-1, 1)
                    
                    norm_const, mean = self.normalizer.get_nc(unnorm_density, obs = obs_spec.unsqueeze(0), get_fm=True)
                    means[obs_idx, :] = mean
                    norm_consts[obs_idx, :] = norm_const
                    obs_idx += 1
                
        return norm_consts, means
    
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

        #checkpoint = torch.load(filepath)
        checkpoint = torch.load(filepath)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])