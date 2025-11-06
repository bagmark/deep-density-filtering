import torch

import numpy as np
from scipy.stats import gaussian_kde

class ParticleFilter():
    def __init__(self, problem, times, obs_indices, n_particles):
        self.device = "cpu" # Faster on CPU in most cases

        self.problem = problem
        self.state_dim = problem.state_dim
        self.obs_dim = problem.obs_dim
        self.times = times.to(self.device)
        self.n_times = len(times)
        self.obs_indices = obs_indices.to(self.device)
        self.n_particles = n_particles

    def filter(self, obs):
        """ obs: (n_obs, d') """
        self.obs = obs.to(self.device)

        self.states = torch.zeros((self.n_times, self.n_particles, self.state_dim), device=self.device) 
        self.weights = torch.zeros((self.n_times, self.n_particles), device=self.device)         

        with torch.no_grad():
            # At time t_0
            self.states[0,:,:] = self.problem.sample_p0(self.n_particles, self.device)
            self.weights[0,:] = torch.ones(self.n_particles, device=self.device) / self.n_particles

            for k in range(self.n_times - 1):
                if k in self.obs_indices:
                    x = self._resample(self.states[k,:,:], self.weights[k,:])
                else:
                    x = self.states[k,:,:]

                deltat = self.times[k+1] - self.times[k]
                deltaW = torch.randn((self.n_particles, self.state_dim), device=self.device) * torch.sqrt(deltat)
                drift_term = self.problem.drift_batch(x) * deltat
                diffusion_term = torch.einsum('ijk,ik->ij', self.problem.diffusion_batch(x), deltaW)
                self.states[k+1,:,:] = x + drift_term + diffusion_term

                if self.problem.identifier == f"L96_{self.state_dim}": # for robustness to avoid divergence
                    self.states[k+1,:,:] = torch.clamp(self.states[k+1,:,:], min=-500, max=500)

                if (k + 1) in self.obs_indices:
                    idx = torch.where(self.obs_indices == (k+1))[0][0]
                    self.weights[k+1,:] = self._calculate_weights(self.states[k+1,:,:], self.obs[idx, :])
                else:
                    self.weights[k+1,:] = torch.ones(self.n_particles, device=self.device) / self.n_particles 

        return self.states, self.weights
    
    def _resample(self, states, weights):
        sampled_indices = torch.tensor(np.random.choice(states.shape[0], size=states.shape[0], p=weights.cpu().numpy()), device=self.device)
        sampled_states = states[sampled_indices, :]
        return sampled_states
    
    def _calculate_weights(self, states, observations):
        log_likelihoods = self.problem.obs_loglikelihood_mto(states, observations)
        weights = torch.exp(log_likelihoods - torch.max(log_likelihoods))

        if self.problem.identifier == f"L96_{self.state_dim}": # for robustness to avoid divergence
            weights = torch.clamp_min(weights, 1e-10)
        
        return weights / torch.sum(weights)
    
    def build_kdes(self):
        """
        To improve performance downstream, this function prebuilds kde objects in filtering times
        """
        self.kdes = []

        for idx in self.obs_indices:
            if self.problem.identifier == f"L96_{self.state_dim}": # for robustness to avoid divergence /singularity
                x = (self.states[idx,:,:] + 1e-3 * torch.randn_like(self.states[idx,:,:])).T.cpu()
            else:
                x = self.states[idx,:,:].T.cpu()
            weight = self.weights[idx,:].cpu()
            kde = gaussian_kde(x, weights = weight)
            self.kdes = self.kdes + [kde]
    
    def get_filter_pdfs(self, observations, device, x_min=-5, x_max=5, n_points = 1000):
        """ Returns the pdfs of the filter at all observation times evaluated on a grid """

        with torch.no_grad():
            points = [torch.linspace(x_min, x_max, n_points) for _ in range(self.state_dim)]
            mesh = torch.meshgrid(*points, indexing='ij')
            x = torch.vstack([dimension.ravel() for dimension in mesh]).T

            values = torch.zeros((len(self.obs_indices), x.shape[0]), device=device)
            for i in range(len(self.obs_indices)):
                values[i,:] = self.evaluate_filter_pdf(i, x, device)

        return values, x.to(device=device)
    
    def get_filter_means(self, observations, device, x_min=None, x_max=None, n_points = None):
        """ Returns the means of the distributions at all timepoints """

        with torch.no_grad():
            states = self.states[self.obs_indices,:,:]
            weights = self.weights[self.obs_indices,:]
            weights_new = weights.unsqueeze(2).expand(-1, self.n_particles, self.state_dim)
            means = torch.sum(states * weights_new, dim=1) / torch.sum(weights_new, dim=1)
        return means.to(device=device)

    def evaluate_filter_pdf(self, obs_idx, x, device, obs=None):
        """
        Returns the pdf of the filter at chosen timepoint in the points x
        obs_idx: index in self.obs_indices
        x: (n points, state space dimension)
        """
        
        # Use pre-built kdes
        x = x.cpu().detach()
        return torch.tensor(self.kdes[obs_idx].pdf(x.T), device=device)
    
    def sample_filter_densities(self, obs, n_samples, device):
        """ Returns samples from the filter at all timepoints"""

        samples = torch.zeros((len(self.obs_indices), n_samples, self.state_dim))
        values = torch.zeros((len(self.obs_indices), n_samples), device=device)

        for i in range(len(self.obs_indices)):
            idx = self.obs_indices[i]
            states = self.states[idx,:,:]
            weights = self.weights[idx,:]

            sampled_indices = torch.multinomial(weights, num_samples=n_samples, replacement=True)
            samples[i,:,:] = states[sampled_indices, :]
            
            values[i,:] = self.evaluate_filter_pdf(i, samples[i,:,:], device)

        return samples.to(device=device), values