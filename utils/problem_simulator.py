import torch

# Class implementing methods for simulating trajectories for diffusion-type SDEs with observations

class ProblemSimulator:
    def __init__(self, problem, times, obs_indices):
        self.problem = problem
        self.state_dim = problem.state_dim
        self.meas_dim = problem.obs_dim
        self.times = times
        self.n_times = len(times)
        self.obs_indices = obs_indices
        self.n_obs = len(obs_indices)

    def _simulate_state_em(self, n_obs, q0 = None):
        if q0 == None:
            q0 = self.problem.sample_p0
        # Simulate state using Euler-Maruyama method
        obs_indices = self.obs_indices[:n_obs]
        times = self.times[:obs_indices[-1] + 1]
        n_times = len(times)

        self.X = torch.zeros((self.N_simulations, n_times, self.state_dim))
        self.deltaW = torch.zeros((self.N_simulations, n_times, self.state_dim))
        self.X[:,0,:] = q0(self.N_simulations, device=self.X.device)

        for i in range(0, n_times-1):
            deltat = times[i+1] - times[i]
            deltaW = torch.randn((self.N_simulations, self.state_dim)) * torch.sqrt(deltat)

            self.deltaW[:, i, :] = deltaW

            drift_term = self.problem.drift_batch(self.X[:,i,:]) * deltat
            diffusion_term = torch.einsum("ijk, ik->ij", self.problem.diffusion_batch(self.X[:,i,:]), deltaW)

            self.X[:,i+1,:] = self.X[:,i,:] + drift_term + diffusion_term

    def _measure(self, n_obs):
        obs_indices = self.obs_indices[:n_obs]

        self.Y = self.problem.obs_batch(self.X[:,obs_indices,:]) + torch.randn((self.N_simulations, n_obs, self.meas_dim)) * self.problem.obs_noise_std
    
    def simulate_state_and_obs(self, N_simulations = 1, n_obs = None, q0 = None):
        if q0 == None:
            q0 = self.problem.sample_p0
        # importance specifies if we should q0 instead of p0 to sample from

        if n_obs is None: # If number of observations was not specified assume we want all of them
            n_obs = self.n_obs

        self.N_simulations = N_simulations
        self._simulate_state_em(n_obs, q0)
        self._measure(n_obs)

        return self.X, self.Y, self.deltaW