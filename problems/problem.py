import numpy as np

import torch

# General parent class for problems implementing generic methods

class Problem():
    def __init__(self):
        self.identifier = "Pr"

        self.obs_noise_std = 1
        self.state_dim = 1
        self.obs_dim = 1
    
    def drift_batch(self, X):
        pass 

    def diffusion_batch(self, X):
        pass
    
    def obs_batch(self, X):
        pass

    def get_deep_filter_params(self, X):
        """
        Returns values and gradients needed for the deep filters
        X: (B, d) a collection of states
        Let a = sigma*sigma^T
        returns 
        a1: da_ij/dx_i matrix, (B, d, d)
        a2: d^2 a_ij/dx_i*dx_j matrix, (B, d, d)
        a3: dmu_i/dx_i vector, (B, d)
        a4: mu vector, (B, d)
        """
        pass
    
    def get_obs_jacobian(self, X):
        """ 
        Returns observation function Jacobian 
        X: (B, d)
        returns: (B, d', d)
        """
        pass
    
    def sample_p0(self, N_simulations, device):
        pass
    
    def p0(self, x):
        """
        Returns the likelihood of a given starting state (pdf of the prior)
        Input:
        x: (B, d)
        Returns: 
        likelihoods: (B, 1)
        """
        pass
    
    def log_p0(self, x):
        """
        Returns the loglikelihood of a given starting state (pdf of the prior)
        Input:
        x: (B, d)
        Returns: 
        likelihoods: (B, 1)
        """
        pass
    
    def p0_grad(self, x):
        """
        Returns gradient of the prior
        """
        pass
    
    def log_p0_grad(self, x):
        """
        Returns gradient of the log-prior
        """
        pass

    def get_prior_moments(self, device):
        """ 
        Returns mean and covariance for the prior distribution
        Needed for all kalman filters
        """

        pass

    def get_kf_params(self, deltat, device):
        """
        Returns parameters needed for the kalman filter
        Based on model: 
        x_k = A_(k-1) x_(k-1) + q_(k-1) + b_(k-1)
        y_k = H_k x_k + r_k
        q_(k-1) ~ N(0, Q_(k-1))
        r_k ~ R_k 
        """
        pass
    
    def get_ekf_update_params(self, x, r):
        """
        Returns parameters needed for the extended kalman filter update step
        Based on model: 
        y_k = h(x_k, r)
        r ~ N(0, R)
        Hx jacobian of h wrt x
        Hr jacobian of h wrt r
        """
        pass
    
    def get_ekf_pred_params(self, x, q, deltat):
        """
        Returns parameters needed for the extended kalman filter prediction step
        Based on model: 
        x_k = f(x_k-1, q_k-1)
        q_(k-1) ~ N(0, Q)
        Fx jacobian of f wrt x
        Fq jacobian of f wrt q
        """
        pass
    
    def bsdef_f(self, x, u, v):
        """
        x: (B, d)
        u: (B, 1)
        v: (B, d)
        returns: f(x, u, v): (B, 1)
        """

        a1, a2, a3, a4 = self.get_deep_filter_params(x)
        # a1: (B, d, d), a2: (B, d, d), a3: (B, d), a4: (B, d)

        term1 = torch.sum(torch.matmul(a1, v.unsqueeze(2)), dim=1)
        term2 = torch.sum(torch.sum(a2, dim=2), dim=1, keepdim=True) * u / 2
        term3 = -torch.sum(a3, dim=1, keepdim=True) * u
        term4 = -2 * torch.sum(a4 * v, dim=1, keepdim=True)

        return term1 + term2 + term3 + term4
    
    def dsf_F(self, x, val, grad):
        """
        val: (B, 1)
        grad: (B, d)
        returns: (Ff)(x), the operator F acting on the function f whose value is val and gradient grad in x
        """
        return self.bsdef_f(x, val, grad)

    def log_bsdef_f(self, x, u, v):
        """
        x: (B, d)
        u: (B, 1)
        v: (B, d)
        returns: f(x, u, v): (B, 1)
        """

        sigma = self.diffusion_batch(x)
        sigma_t = torch.transpose(sigma,1,2)

        transformed = torch.bmm(sigma_t, v.unsqueeze(-1)).squeeze(-1)

        squared_norms = torch.sum(transformed**2, dim=1)

        term1 = -0.5 * squared_norms.unsqueeze(-1)

        term2 = -self.bsdef_f(x, torch.ones_like(u), -v)

        return term1 + term2
    
    def log_dsf_f(self, x, val, grad):
        """
        x: (B, d)
        u: (B, 1)
        v: (B, d)
        returns: f(x, u, v): (B, 1)
        """

        return self.log_bsdef_f(x, val, grad)
    
    def obs_likelihood(self, x, y):
        """
        Evaluates the multivariate normal pdf with mean h(x) in y
        one-to-one version -> one x point for one y point
        Input:
        x: (B, d)
        y: (B, d')
        Output:
        likelihoods (B)
        """
        variance = self.obs_noise_std**2
        #R = np.identity(self.obs_dim) * variance
        means = self.obs_batch(x)
        sqrt_det_cov = variance**(self.obs_dim/2)
        inv_cov = torch.eye(self.obs_dim) / variance
        diff = y - means
        exponent = -0.5 * torch.sum(torch.matmul(diff, inv_cov) * diff, dim=1)
        prefactor = 1 / ((2 * np.pi) ** (self.obs_dim / 2) * sqrt_det_cov)
        likelihoods = prefactor * torch.exp(exponent)

        return likelihoods

    def obs_loglikelihood(self, x, y):
        """
        Evaluates the log of the multivariate normal pdf with mean h(x) at y
        one-to-one version -> one x point for one y point
        Input:
        x: (B, d)
        y: (B, d')
        Output:
        log_likelihoods (B)
        """
        variance = self.obs_noise_std**2
        means = self.obs_batch(x)
        inv_cov = torch.eye(self.obs_dim) / variance
        diff = y - means
        exponent = -0.5 * torch.sum(torch.matmul(diff, inv_cov) * diff, dim=1)
        
        log_prefactor = -0.5 * self.obs_dim * np.log(2 * np.pi) - 0.5 * self.obs_dim * torch.log(torch.tensor(variance))
        log_likelihoods = log_prefactor + exponent

        return log_likelihoods
    
    def obs_loglikelihood_mto(self, x, y):
        """
        Evaluates the multivariate normal pdf with mean h(x) in y
        Many-to-one version -> one y point many x points
        Input:
        x: (B, d)
        y: (d')
        Output:
        likelihoods (num_means)
        """
        device = x.device

        R = torch.eye(self.obs_dim, device=device) * self.obs_noise_std**2

        # Use symmetry of normal distribution to evaluate one pdf in many points instead of many pdfs in one point
        points = self.obs_batch(x)
        mean = y
        mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, covariance_matrix=R)
        log_likelihoods = mvn.log_prob(points)

        return log_likelihoods