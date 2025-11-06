import torch

class KalmanFilter():
    def __init__(self, problem, times, obs_indices):
        self.device = "cpu" # Here filtering is more efficient on the cpu 

        self.problem = problem
        self.state_dim = problem.state_dim
        self.obs_dim = problem.obs_dim
        self.times = times.to(self.device)
        self.n_times = len(times)
        self.obs_indices = obs_indices.to(self.device)

        self.m0, self.P0 = problem.get_prior_moments(self.device)

    def filter(self, obs):
        """ obs: (n_obs, d') """
        self.obs = obs.to(self.device)
        self.m = torch.zeros((self.n_times, self.state_dim), device=self.device)
        self.P = torch.zeros((self.n_times, self.state_dim, self.state_dim), device=self.device)

        # At time t_0
        # Assume no update
        self.m[0,:] = self.m0
        self.P[0,:,:] = self.P0

        with torch.no_grad():
            for k in range(0, self.n_times-1):
                deltat = self.times[k + 1] - self.times[k]
                Ak, bk, Qk, R, H = self.problem.get_kf_params(deltat, device=self.device)

                # Prediction step
                mm = torch.matmul(Ak, self.m[k,:]) + bk
                Pm = torch.matmul(torch.matmul(Ak, self.P[k,:,:]), torch.transpose(Ak, 0, 1)) + Qk

                # Update step
                if (k + 1) in self.obs_indices:
                    idx = torch.where(self.obs_indices == k+1)[0][0]
                    v = self.obs[idx, :] -  torch.matmul(H, mm)
                    S = torch.matmul(torch.matmul(H, Pm), torch.transpose(H, 0, 1)) + R
                    K = torch.matmul(torch.matmul(Pm, torch.transpose(H, 0, 1)), torch.linalg.inv(S))

                    self.m[k+1, :] = mm + torch.matmul(K, v)
                    self.P[k+1, :,:] = Pm - torch.matmul(torch.matmul(K, S), torch.transpose(K, 0, 1))
                else:
                    self.m[k+1, :] = mm
                    self.P[k+1, :,:] = Pm 
    
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
    
    def get_filter_means(self, observations, device):
        """ Returns the means of the distributions at all timepoints """

        m = self.m[self.obs_indices, :].to(device=device)
        return m

    def evaluate_filter_pdf(self, obs_idx, x, device, obs = None):
        """
        Returns the pdf of the filter at chosen timepoint in the points x
        obs_idx: index in self.obs_indices
        x: (n points, state space dimension)
        """

        idx = self.obs_indices[obs_idx]

        m = self.m[idx, :]
        P = self.P[idx, :, :]

        with torch.no_grad():
            mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=m, covariance_matrix=P)
            predictions = torch.exp(mvn.log_prob(x.to(self.device)))

        return predictions.to(device=device)
    
    def sample_filter_densities(self, obs, n_samples, device):
        """ Returns samples from the filter at all timepoints"""

        samples = torch.zeros((len(self.obs_indices), n_samples, self.state_dim), device=self.device)
        values = torch.zeros((len(self.obs_indices), n_samples), device=self.device)

        for i in range(len(self.obs_indices)):
            idx = self.obs_indices[i]

            m = self.m[idx, :]
            P = self.P[idx, :, :]

            mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=m, covariance_matrix=P)
            samples[i,:,:] = mvn.sample((n_samples,))
            values[i,:] = torch.exp(mvn.log_prob(samples[i, :, :]))

        return samples.to(device=device), values.to(device=device)