import torch
import torch.nn as nn

from logging import getLogger
from utils.utils import *

class BSDEFTrainer():
    def __init__(self, trainer_params, simulator, model, prev_model, optimizer, scheduler, normalizer, problem):
        self.simulator = simulator
        self.model = model
        self.prev_model = prev_model
        self.normalizer = normalizer
        self.problem = problem

        self.epochs = trainer_params['epochs']
        self.training_samples = trainer_params['samples_per_epoch']
        self.batch_size = trainer_params['batch_size']
        self.early_stop = trainer_params['early_stopping']

        if self.early_stop:
            if "early_stopping_patience" in trainer_params:
                patience = trainer_params['early_stopping_patience']
            else:
                patience = 5
            self.early_stopper = EarlyStopper(patience=patience)
        
        if "batch_size_obs" in trainer_params:
            self.batch_size_obs = trainer_params['batch_size_obs']
        else:
            self.batch_size_obs = 2**6

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.logger = getLogger(name='trainer')
        self.criterion = nn.MSELoss()

    def train_model(self):

        self.logger.info('=================================================================')

        schedule_counter = 0

        for epoch in range(self.epochs):
            loss_mean_epoch = self._train_one_epoch()

            if self.early_stop:
                stop = self.early_stopper.early_stop(loss_mean_epoch)

                if self.early_stopper.counter > 10:
                    schedule_counter += 1
                    if schedule_counter < self.scheduler.T_max:
                        self.scheduler.step()
                    elif schedule_counter == self.scheduler.T_max:
                        self.logger.info(f"Reach final learning rate {self.optimizer.param_groups[0]['lr']}")


                if stop: 
                    self.logger.info(f"Early stopping at epoch {epoch + 1} - Average loss: {loss_mean_epoch}")
                    break

            self.logger.info(f"Epoch: {epoch + 1} - Average Loss: {loss_mean_epoch}")

        self.logger.info(f"Final learning rate {self.optimizer.param_groups[0]['lr']}")
        
    
    def _train_one_epoch(self):

        remaining_samples = self.training_samples
        loss_mean_epoch = 0

        while remaining_samples > 0:
            batch_size = min(self.batch_size, remaining_samples)
            remaining_samples -= batch_size

            loss_mean = self._train_one_batch(batch_size)
            loss_mean_epoch += loss_mean * batch_size / self.training_samples

        return loss_mean_epoch


    def _train_one_batch(self, batch_size):

        batch_size_obs = self.batch_size_obs
        thin_out_factor = batch_size // batch_size_obs

        if (self.problem.identifier == "schlogel") or (self.problem.identifier == "L96_4"):
            q0 = self.problem.sample_q0
        else:
            q0 = self.problem.sample_p0

        X, obs, deltaW = self.simulator.simulate_state_and_obs(batch_size, self.model.n_obs + 1, q0)

        # Want X and obs independent -> Shuffle obs
        perm = torch.randperm(X.shape[0])
        obs = obs[perm[:batch_size_obs], :, :]  # Thin observations

        # Only need relevant snippet
        start_time = torch.where(self.simulator.times == self.model.times[0])[0]
        X = X[:, start_time:, :]
        deltaW = deltaW[:, start_time:, :]
        obs = obs[:, :-1, :]  

        # Expand obs to match X: repeat each obs `thin_out_factor` times
        obs = obs.repeat_interleave(thin_out_factor, dim=0)

        # Trim X and deltaW to match this shape
        X = X[:batch_size_obs * thin_out_factor]
        deltaW = deltaW[:batch_size_obs * thin_out_factor]

        # Pass through model
        Y = self.model(X, obs, deltaW)

        with torch.no_grad():
            if self.model.n_obs == 0:
                labels = self.problem.p0(X[:, -1, :])
            else:
                unique_obs = obs[::thin_out_factor]  # [batch_size_obs, ...]

                self.prev_model.eval()

                norm_consts = self.normalizer.get_nc(self.prev_model.filter_unnorm, unique_obs)
                norm_consts = (norm_consts).repeat_interleave(thin_out_factor, dim=0)
                labels = self.prev_model.filter_unnorm(X[:, -1, :], obs) / norm_consts
        
        loss = self.criterion(Y, labels)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss

class EarlyStopper:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss >= (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False