import torch

from scipy.stats import gaussian_kde

class EnsembleKalmanFilter():
    def __init__(self, problem, times, obs_indices, ensemble_size=100):
        self.device = "cpu" # Often this is faster on cpu

        self.problem = problem
        self.state_dim = problem.state_dim
        self.obs_dim = problem.obs_dim
        self.times = times.to(self.device)
        self.n_times = len(times)
        self.obs_indices = obs_indices.to(self.device)
        self.ensemble_size = ensemble_size

        self.m0, self.P0 = problem.get_prior_moments(self.device)

    def filter(self, obs):
        """ obs: (n_obs, d') """
        self.obs = obs.to(self.device)
        # Initialize ensemble from prior
        X = self.problem.sample_p0(self.ensemble_size, self.device)
        self.ensemble = torch.zeros((self.n_times, self.ensemble_size, self.state_dim), device=self.device)
        self.ensemble[0] = X

        with torch.no_grad():

            for k in range(self.n_times - 1):
                deltat = self.times[k + 1] - self.times[k]
                deltaW = torch.randn((self.ensemble_size, self.state_dim), device=self.device) * torch.sqrt(deltat)

                # Prediction step: propagate each ensemble member
                drift_term = self.problem.drift_batch(X) * deltat
                diffusion_term = torch.einsum('ijk,ik->ij', self.problem.diffusion_batch(X), deltaW)
                
                X = X + drift_term + diffusion_term
                if self.problem.identifier == "schlogel": # requires positive values
                    X = torch.clamp(X, min=0.1)
                elif self.problem.identifier == f"L96_{self.state_dim}": # requires positive values
                    X = torch.clamp(X, min=-500, max=500)
            
                # Update if observation is available
                if (k + 1) in self.obs_indices:
                    idx = torch.where(self.obs_indices == k + 1)[0][0]
                    y = self.obs[idx, :]

                    # Perturb observations for each ensemble member
                    R = self.problem.obs_noise_std**2 * torch.eye(self.obs_dim, device=self.device)

                    obs_noise = torch.distributions.MultivariateNormal(torch.zeros(self.obs_dim, device=self.device), R).sample((self.ensemble_size,))
                    
                    Y = self.problem.obs_batch(X) + obs_noise

                    X_mean = X.mean(dim=0, keepdim=True)
                    Y_mean = Y.mean(dim=0, keepdim=True)

                    P_xy = ((X - X_mean).unsqueeze(2) * (Y - Y_mean).unsqueeze(1)).mean(dim=0)
                    P_yy = ((Y - Y_mean).unsqueeze(2) * (Y - Y_mean).unsqueeze(1)).mean(dim=0)

                    K = torch.linalg.solve(P_yy + 1e-6 * torch.eye(self.obs_dim, device=self.device), P_xy.T).T
                    innovation = (y - Y)
                
                    X = X + (innovation @ K.T)
                    
                if self.problem.identifier == "schlogel": # requires positive values
                    X = torch.clamp(X, min=0.1)
                elif self.problem.identifier == f"L96_{self.state_dim}": # requires positive values
                    X = torch.clamp(X, min=-500, max=500)

                self.ensemble[k + 1] = X

        self.X = X 

    def build_kdes(self):
        """
        To improve performance downstream, this function prebuilds kde objects in filtering times
        """
        self.kdes = []

        for idx in self.obs_indices:
            if self.problem.identifier == f"L96_{self.state_dim}": # for robustness to avoid divergence /singularity
                x = (self.ensemble[idx,:,:] + 1e-3 * torch.randn_like(self.ensemble[idx,:,:])).T.cpu()
            else:
                x = self.ensemble[idx,:,:].T.cpu()
            weight = torch.ones(self.ensemble_size) / self.ensemble_size  # Uniform weights for EnKF
            kde = gaussian_kde(x, weights = weight.cpu())
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
        means = self.ensemble[self.obs_indices, :, :].mean(dim=1).to(device)
        return means
    
    def evaluate_filter_pdf(self, obs_idx, x, device, obs = None):
        """
        Returns the pdf of the filter at chosen timepoint in the points x
        obs_idx: index in self.obs_indices
        x: (n points, state space dimension)
        """

        # Compute kernel density estimate using pre-built KDE
        kde = self.kdes[obs_idx]
        pdf_vals = torch.tensor(kde.evaluate(x.cpu().T), device=device)
        return pdf_vals

    def sample_filter_densities(self, obs, n_samples, device):
        """ Returns samples from the filter at all timepoints"""
    
        samples = torch.zeros((len(self.obs_indices), n_samples, self.state_dim), device=device)
        values = torch.zeros((len(self.obs_indices), n_samples), device=device)

        for i in range(len(self.obs_indices)):
            idx = self.obs_indices[i]
            states = self.ensemble[idx,:,:]
            weights = torch.ones(self.ensemble_size) / self.ensemble_size  # Uniform weights for EnKF

            sampled_indices = torch.multinomial(weights, num_samples=n_samples, replacement=True)
            samples[i,:,:] = states[sampled_indices, :]

            values[i,:] = self.evaluate_filter_pdf(i, samples[i,:,:], device)

        return samples.to(device), values