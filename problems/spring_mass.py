import numpy as np

import torch

from problems.problem import Problem

# Defines a spring-mass system in d dimensions with linear measurements
# dXt = A Xt dt + sigma dWt
# h(x) = x
# A = [0, I; A21, A22]

class SM(Problem):
    def __init__(self, masses = 1, observed_masses = 1, obs_noise_std = 1):
        self.identifier = "SM_" + str(masses)

        self.obs_noise_std = obs_noise_std

        self.state_dim = masses * 2  # 2 for position and velocity
        self.obs_dim = observed_masses

        self.M = masses
        self.sigma = torch.eye(self.state_dim)

        self.A = torch.zeros((self.state_dim, self.state_dim))
        self.A[:self.M ,self.M:] = torch.eye(self.M)

        np.random.seed(42) # Set seed to use the same A matrix for all runs
        self.m = torch.tensor(np.random.uniform(0.8, 1.2, self.M))
        self.k = torch.tensor(np.random.uniform(0.8, 1.2, self.M + 1))
        self.c = torch.tensor(np.random.uniform(0.15, 0.25, self.M + 1))

        # A21
        a21diag = -(self.k[:-1] + self.k[1:]) / self.m
        a21diagu = self.k[1:-1] / self.m[:-1]
        a21diagl = self.k[1:-1] / self.m[1:]
        A21 = torch.zeros((self.M, self.M))
        A21.diagonal().copy_(a21diag)
        for i in range(self.M-1):
            A21[i, i+1] = a21diagu[i]
        for i in range(1, self.M):
            A21[i, i-1] = a21diagl[i-1]
        self.A[self.M:, :self.M] = A21

        # A22
        a22diag = -(self.c[:-1] + self.c[1:]) / self.m
        A22 = torch.zeros((self.M, self.M))
        A22.diagonal().copy_(a22diag)
        self.A[self.M:, self.M:] = A22

        # Measurement matrix
        self.H = torch.zeros((self.obs_dim, self.state_dim))
        for i in range(observed_masses):
            self.H[i, i] = 1
    
    def drift_batch(self, X):
        # X: (B, d)
        device = X.device

        if device == torch.device('cpu'):
            A = self.A.cpu()
        else:
            A = self.A
        
        mu = torch.matmul(A, X.t()).t()
        return mu

    def diffusion_batch(self, X):
        # X: (B, d)
        device = X.device

        if device == torch.device('cpu'):
            sigma = self.sigma.cpu()
        else:
            sigma = self.sigma

        B = X.shape[0]
        sigma = sigma.unsqueeze(0).expand(B, self.state_dim, self.state_dim)
        return sigma
    
    def obs_batch(self, X):
        # X: (B, d)
        device = X.device

        if device == torch.device('cpu'):
            H = self.H.cpu()
        else:
            H = self.H

        h = torch.matmul(X, H.T)
        return h

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

        a3 = torch.diag(self.A)
        a3 = a3.unsqueeze(0).repeat(B, 1)

        a4 = self.drift_batch(X)

        return a1, a2, a3,  a4
    
    def get_obs_jacobian(self, X):
        """ 
        Returns observation function Jacobian 
        X: (B, d)
        returns: (B, d', d)
        """
        B, _ = X.shape
        H = self.H.unsqueeze(0).repeat(B, 1, 1)
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
        if torch.device(device) == torch.device('cpu'):
            sigma = self.sigma.cpu()
            A_mat = self.A.cpu()
        else:
            sigma = self.sigma
            A_mat = self.A

        A = torch.eye(self.state_dim, device=device)
        A += A_mat*deltat
        b = torch.zeros(self.state_dim, device=device)
        Q = torch.matmul(sigma, torch.transpose(sigma, 0, 1)) * deltat
        R = torch.eye(self.obs_dim, device=device) * self.obs_noise_std**2

        if torch.device(device) == torch.device('cpu'):
            H = self.H.cpu()
        else:
            H = self.H
            
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

        if device == torch.device('cpu'):
            Hx = self.H.cpu()
        else:
            Hx = self.H

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

        if device == torch.device('cpu'):
            sigma = self.sigma.cpu()
            A = self.A.cpu()
        else:
            sigma = self.sigma
            A = self.A

        f = torch.matmul(torch.eye(self.state_dim, device=device), x)
        f += torch.matmul(A, x)*deltat
        f += torch.matmul(sigma, q)
        Fx = torch.eye(self.state_dim, device=device)
        Fx += A*deltat
        Fq = sigma
        Q = torch.eye(self.state_dim, device=device) * deltat
            
        return f, Q, Fx, Fq