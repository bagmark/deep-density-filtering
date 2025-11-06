import torch
import torch.nn as nn

from logging import getLogger
from utils.utils import *

class LogDSFTrainer():
    def __init__(self, trainer_params, simulator, dsf_model, optimizer, scheduler, model_number,
                        problem, prev_model, normalizer, deltat):
        self.simulator = simulator
        self.model = dsf_model
        self.model_number = model_number

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
        
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.problem = problem
        self.prev_model = prev_model
        self.normalizer = normalizer
        self.deltat = deltat

        if "batch_size_obs" in trainer_params:
            self.batch_size_obs = trainer_params['batch_size_obs']
        else:
            self.batch_size_obs = 2**6


        self.logger = getLogger(name='trainer')

        self.criterion = nn.MSELoss()

    def train_model(self):
        self.train_model_fixed()

    def train_model_fixed(self):
        self.logger.info('=================================================================')
        schedule_counter = 0

        # Precompute training data
        Z_input, obs, labels = self._prepare_dataset_batched()

        for epoch in range(self.epochs):

            loss_mean_epoch = self._train_one_epoch_fixed(Z_input, obs, labels)

            if self.model_number > 1:
                if epoch > 10:   # on prediction step we are happy with ~10 epochs 
                    break
            if self.early_stop:
                stop = self.early_stopper.early_stop(loss_mean_epoch)

                if self.early_stopper.counter > 2:
                    schedule_counter += 1
                    if schedule_counter < self.scheduler.T_max:
                        self.scheduler.step()
                    elif schedule_counter == self.scheduler.T_max:
                        self.logger.info(f"Reach final learning rate {self.optimizer.param_groups[0]['lr']}")

                if stop:
                    self.logger.info(f"Early stopping at epoch {epoch + 1} - Average loss: {loss_mean_epoch}")
                    break

            self.logger.info(f"Epoch: {epoch + 1} - Average Loss: {loss_mean_epoch}")

    def _train_one_epoch_fixed(self, Z_input, obs, labels):
        N = self.training_samples
        loss_mean_epoch = 0

        for i in range(0, N, self.batch_size):
            batch_Z_input = Z_input[i:i+self.batch_size]
            batch_obs     = obs[i:i+self.batch_size]
            batch_labels  = labels[i:i+self.batch_size]

            loss_mean = self._train_one_batch_fixed(batch_Z_input, batch_obs, batch_labels)
            loss_mean_epoch += loss_mean * batch_Z_input.shape[0] / N

        return loss_mean_epoch

    def _train_one_batch_fixed(self, Z_input, obs, labels):
        device = next(self.model.parameters()).device

        Z_input = Z_input.to(device, non_blocking=True)
        obs     = obs.to(device, non_blocking=True)
        labels  = labels.to(device, non_blocking=True)

        pred = self.model(Z_input, obs)
        loss = self.criterion(pred, labels)

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def _label_function_t0(self, X, obs):
        """ Label function when using prior density as label """

        with torch.no_grad():
            val = -self.problem.log_p0(X)
            grad = -self.problem.log_p0_grad(X)
            Ff = self.problem.log_dsf_f(X, val, grad)

        # NOTE: Assume p0 is already normalized
        return val + Ff * self.deltat
    
    def _label_function(self, X, obs):
        """ Label function when using prediction density as label """
        
        self.prev_model.eval()

        X_grad = X.clone().requires_grad_(True)
        val = self.prev_model(X_grad, obs)
        grad = torch.autograd.grad(outputs=val, inputs=X_grad, grad_outputs=torch.ones_like(val), create_graph=False, retain_graph=False, only_inputs=True)[0]
        #
        X_grad = X_grad.detach()
        #
        val = val.detach() 
        grad = grad.detach()

        with torch.no_grad():
            Ff = self.problem.log_dsf_f(X, val, grad)

        # Do not normalize 
        return val + Ff * self.deltat

    def _prepare_dataset_batched(self, chunk_size: int = None):

        total = self.training_samples
        if chunk_size is None:
            chunk_size = 2**14

        device = next(self.model.parameters()).device 

        Z_input_list, obs_list, labels_list = [], [], []
        processed = 0

        while processed < total:
            n = min(chunk_size, total - processed)

            with torch.no_grad():

                q0 = self.problem.sample_p0

                X_all, obs_chunk, _ = self.simulator.simulate_state_and_obs(n, self.model.n_obs + 1, q0) 

                # auxiliary process by shuffling within the chunk
                perm = torch.randperm(X_all.shape[0], device=X_all.device)
                Z = X_all[perm, :, :]

                Z_label_chunk = Z[:, self.model.idx, :]
                Z_input_chunk = Z[:, self.model.idx - 1, :]
                obs_chunk = obs_chunk[:, :-1, :]

                Z_label_chunk_dev = Z_label_chunk.to(device, non_blocking=True)
                obs_for_labels = obs_chunk.to(device, non_blocking=True)

            if self.model_number > 1:
                labels_chunk = self._label_function(Z_label_chunk_dev, obs_for_labels).detach()

            elif self.model.n_obs == 0:
                labels_chunk = self._label_function_t0(Z_label_chunk_dev, obs_for_labels).detach()

            else:
                thin_out_factor = max(1, n // self.batch_size_obs)
                usable = self.batch_size_obs * thin_out_factor
                if usable < n:
                    Z = Z[:usable]
                    Z_label_chunk = Z_label_chunk[:usable]
                    Z_input_chunk = Z_input_chunk[:usable]
                    obs_chunk = obs_chunk[:usable]
                    Z_label_chunk_dev = Z_label_chunk_dev[:usable]
                    obs_for_labels = obs_for_labels[:usable]
                    n = usable

                perm_obs = torch.randperm(n, device=obs_chunk.device)
                obs_unique = obs_chunk[perm_obs[:self.batch_size_obs], :, :]
                obs_expanded = obs_unique.repeat_interleave(thin_out_factor, dim=0)
                obs_chunk = obs_expanded.to(device, non_blocking=True)

                self.prev_model.eval()

                X = Z_label_chunk_dev
                X_grad = X.clone().requires_grad_(True)

                pred = self.prev_model(X_grad, obs_expanded[:, :-1, :])
                pred_grad = torch.autograd.grad(
                    outputs=pred,
                    inputs=X_grad,
                    grad_outputs=torch.ones_like(pred),
                    create_graph=False,
                    retain_graph=False,
                    only_inputs=True
                )[0]

                X_grad   = X_grad.detach()
                pred     = pred.detach()
                pred_grad= pred_grad.detach()

                with torch.no_grad():
                    obs_last = obs_expanded[:, -1, :]                                      
                    loglikelihood = self.problem.obs_loglikelihood(X, obs_last).view(-1, 1)

                    expanded_obs = (obs_last - self.problem.obs_batch(X)).unsqueeze(2)
                    H = self.problem.get_obs_jacobian(X)
                    Ht = torch.transpose(H, 1, 2)                  

                    loglikelihood_grad = (
                        torch.einsum('ijk,ikm->ijm', Ht, expanded_obs).transpose(2, 0)[0]
                    ).t() / (self.problem.obs_noise_std ** 2)

                    unique_obs = obs_expanded[::thin_out_factor]

                    def unnorm_log_density(X_last_in, obs_in):
                        o_last = obs_in[:, -1, :]
                        return -self.prev_model(X_last_in, obs_in[:, :-1, :]) + \
                            self.problem.obs_loglikelihood(X_last_in, o_last).view(-1, 1)

                    log_norm_consts_unique = self.normalizer.get_log_nc(unnorm_log_density, unique_obs)  

                    log_norm_consts = log_norm_consts_unique.repeat_interleave(thin_out_factor, dim=0)

                    # Deep Splitting (log version)
                    val  = pred - loglikelihood + log_norm_consts 
                    grad = pred_grad - loglikelihood_grad + log_norm_consts

                    Ff = self.problem.log_dsf_f(X, val, grad)
                    labels_chunk = (val + Ff * self.deltat)

            # ---------- stash CPU copies ----------
            Z_input_list.append(Z_input_chunk.cpu())
            obs_list.append(obs_chunk.cpu())
            labels_list.append(labels_chunk.detach().cpu())

            # optional: free GPU
            if torch.cuda.is_available():
                for name in ["Z", "X_all", "Z_label_chunk", "Z_input_chunk",
                            "obs_for_labels", "labels_chunk", "X", "pred", "pred_grad",
                            "obs_expanded", "obs_unique", "log_norm_consts", "log_norm_consts_unique"]:
                    if name in locals():
                        del locals()[name]
                torch.cuda.empty_cache()

            processed += n

        Z_input = torch.cat(Z_input_list, dim=0)
        obs = torch.cat(obs_list, dim=0)
        labels = torch.cat(labels_list, dim=0)
        return Z_input, obs, labels

    
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