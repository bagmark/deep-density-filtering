import torch

from problems.problem import Problem

# Defines a 1-D SDE with two modes in its unconditional distribution. Observations are linear. 
# dXt = b(aXt - Xt^3) dt + sigma * dWt
# h(x) = x 

class Bistable(Problem):
    def __init__(self, a = 5., b = 2/5, obs_noise_std = 1, sigma = 1.):
        self.identifier = "bistable"

        self.obs_noise_std = obs_noise_std
        self.state_dim = 1
        self.obs_dim = 1
        self.a = a
        self.b = b
        self.sigma = sigma
    
    def drift_batch(self, X):
        # Maybe needs more adaptation with 'device' I simply added an optional argument to align with metric-evaluator / filter code.
        # X: (B, d)
        return self.b * (self.a * X - torch.pow(X, 3))

    def diffusion_batch(self, X):
        # X: (B, d)
        device = X.device

        sigma = torch.tensor(self.sigma, device=device)
        B = X.shape[0]
        sigma = sigma.unsqueeze(0).expand(B, self.state_dim, self.state_dim)
        return sigma
    
    def obs_batch(self, X):
        return X

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
        B = X.shape[0]

        a1 = torch.zeros((self.state_dim, self.state_dim))
        a1 = a1.unsqueeze(0).repeat(B, 1, 1)

        a2 = torch.zeros((self.state_dim, self.state_dim))
        a2 = a2.unsqueeze(0).repeat(B, 1, 1)

        a3 = self.b  * (self.a - 3 * torch.pow(X, 2))

        a4 = self.drift_batch(X)

        return a1, a2, a3,  a4
    
    def get_obs_jacobian(self, X):
        """ 
        Returns observation function Jacobian 
        X: (B, d)
        returns: (B, d', d)
        """
        B, _ = X.shape
        H = torch.eye(self.state_dim)
        H = H.unsqueeze(0).repeat(B, 1, 1)
        return H
    
    def sample_p0(self, N_simulations, device):
        X_0 = torch.randn((N_simulations, self.state_dim), device=device)
        return X_0
    
    def p0(self, x):
        """
        Returns the likelihood of a given starting state (pdf of the prior)
        Input:
        x: (B, d)
        Returns: 
        likelihoods: (B, 1)
        """
        P0 = torch.eye(self.state_dim, dtype=x.dtype)
        m0 = torch.zeros(self.state_dim, dtype=x.dtype)

        mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=m0, scale_tril = P0)
        likelihood = torch.exp(mvn.log_prob(x))

        return likelihood.unsqueeze(1)

    def log_p0(self, x):
        """
        Returns the likelihood of a given starting state (pdf of the prior)
        Input:
        x: (B, d)
        Returns: 
        likelihoods: (B, 1)
        """
        P0 = torch.eye(self.state_dim, dtype=x.dtype)
        m0 = torch.zeros(self.state_dim, dtype=x.dtype)

        mvn = torch.distributions.multivariate_normal.MultivariateNormal(loc=m0, scale_tril = P0)
        loglikelihood = mvn.log_prob(x)

        return loglikelihood.unsqueeze(1)
    
    def p0_grad(self, x):
        """
        Returns gradient of the prior
        """
        P0 = torch.eye(self.state_dim)
        m0 = torch.zeros(self.state_dim)
        m0 = m0.unsqueeze(0).repeat(x.shape[0], 1)

        likelihood = self.p0(x)
        gradient = (-torch.matmul(torch.linalg.inv(P0), (x - m0).t()) * likelihood.t()).t()
        return gradient
    
    def log_p0_grad(self, x):
        """
        Returns gradient of the log-prior
        """
        P0 = torch.eye(self.state_dim)
        m0 = torch.zeros(self.state_dim)
        m0 = m0.unsqueeze(0).repeat(x.shape[0], 1)

        gradient = (-torch.matmul(torch.linalg.inv(P0), (x - m0).t())).t()
        return gradient

    def get_prior_moments(self, device):
        """ 
        Returns mean and covariance for the prior distribution
        Needed for all kalman filters
        """

        P0 = torch.eye(self.state_dim, device=device)
        m0 = torch.zeros(self.state_dim, device=device)
        return m0, P0

    def get_kf_params(self, deltat, device):
        """
        Returns parameters needed for the kalman filter
        Based on model: 
        x_k = A_(k-1) x_(k-1) + q_(k-1) + b_(k-1)
        y_k = H_k x_k + r_k
        q_(k-1) ~ N(0, Q_(k-1))
        r_k ~ R_k 
        """
        raise ValueError("The specified SDE is not linear. Cannot use Kalman Filter")
            
        return A, b, Q, R, H
    
    def get_ekf_update_params(self, x, r):
        """
        Returns parameters needed for the extended kalman filter update step
        Based on model: 
        y_k = h(x_k, r)
        r ~ N(0, R)
        Hx jacobian of h wrt x
        Hr jacobian of h wrt r
        """
        device = x.device

        R = torch.eye(self.obs_dim, device=device) * self.obs_noise_std**2
        Hx = torch.eye(self.state_dim, device=device)
        Hr = torch.eye(self.obs_dim, device=device)
            
        return R, Hx, Hr
    
    def get_ekf_pred_params(self, x, q, deltat):
        """
        Returns parameters needed for the extended kalman filter prediction step
        Based on model: 
        x_k = f(x_k-1, q_k-1)
        q_(k-1) ~ N(0, Q)
        Fx jacobian of f wrt x
        Fq jacobian of f wrt q
        """
        device = x.device

        f = x + self.b * (self.a * x - torch.pow(x, 3)) * deltat + self.sigma * q
        Fx = (1 + self.b * (self.a - 3 * torch.pow(x, 2)) * deltat).view(-1, 1)
        Fq = torch.tensor(self.sigma, device=device).view(-1, 1)
        Q = (deltat).view(-1, 1)
            
        return f, Q, Fx, Fq