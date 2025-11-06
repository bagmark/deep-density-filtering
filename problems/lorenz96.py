import torch

from problems.problem import Problem

# Lorenz–96 SDE with linear partial observations
# dX_i = [(X_{i+1} - X_{i-2}) X_{i-1} - X_i + F] dt + (sigma dW)_i
# y_k = H X_{t_k} + r_k,  r_k ~ N(0, obs_noise_std^2 I)

class Lorenz96(Problem):
    def __init__(
        self,
        dim,
        sigma,
        F=8.0,
        obs_noise_std=2.0,
        obs_every=1,           # observe every r-th component
        obs_offset=0,          # start index offset for observations
    ):
        
        self.identifier = f"L96_{dim}"
        self.state_dim = dim

        self.F = torch.tensor(float(F))
        self.F_cpu = torch.tensor(float(self.F), device='cpu')

        self.obs_noise_std = float(obs_noise_std)
        self.sigma = sigma
        self.sigma_cpu = self.sigma.clone().to('cpu')

        idxs = torch.arange(obs_offset, obs_offset + dim, obs_every) % dim
        self.obs_indices = idxs.to(torch.long)
        self.obs_dim = len(self.obs_indices)

        H = torch.zeros((self.obs_dim, self.state_dim))
        H[torch.arange(self.obs_dim), self.obs_indices] = 1.0
        self.H = H
        self.H_cpu = H.clone().to('cpu')

    def _roll(self, X, k):
        # cyclic shift along state dimension (last dim)
        return torch.roll(X, shifts=k, dims=-1)

    def drift_batch(self, X):
        """
        X: (B, d)
        """
        device = X.device

        F = self.F if device != torch.device('cpu') else self.F_cpu
        x_ip1 = self._roll(X, -1)
        x_im1 = self._roll(X, +1)
        x_im2 = self._roll(X, +2)
        mu = (x_ip1 - x_im2) * x_im1 - X + F
        return mu

    def diffusion_batch(self, X):
        """
        X: (B, d)
        """
        device = X.device

        if device == torch.device('cpu'):
            sigma = self.sigma_cpu
        else:
            sigma = self.sigma
        B = X.shape[0]
        # expand to (B, d, m)
        return sigma.unsqueeze(0).expand(B, sigma.shape[0], sigma.shape[1])

    def obs_batch(self, X):
        """
        X: (B, d) -> (B, p)
        """
        device = X.device

        H = self.H_cpu if device == torch.device('cpu') else self.H
        return X @ H.t()

    def get_obs_jacobian(self, X):
        """
        X: (B, d) -> (B, p, d)
        """
        device = X.device

        B = X.shape[0]
        H = self.H_cpu if device == torch.device('cpu') else self.H
        return H.unsqueeze(0).expand(B, *H.shape)

    def sample_p0(self, N_simulations, device):
        return torch.randn((N_simulations, self.state_dim), device=device) + self.F

    def sample_q0(self, N_simulations, device):
        samples = 1*torch.randn((N_simulations, self.state_dim), device=device) + self.F
        return samples

    def p0(self, x):
        """
        x: (B, d)
        """
        P0 = torch.eye(self.state_dim, dtype=x.dtype, device=x.device)
        m0 = torch.zeros(self.state_dim, dtype=x.dtype, device=x.device) + self.F
        mvn = torch.distributions.MultivariateNormal(loc=m0, covariance_matrix=P0)
        likelihood = torch.exp(mvn.log_prob(x))
        return likelihood.unsqueeze(1)

    def log_p0(self, x):
        """
        x: (B, d)
        """
        P0 = torch.eye(self.state_dim, dtype=x.dtype, device=x.device)
        m0 = torch.zeros(self.state_dim, dtype=x.dtype, device=x.device) + self.F
        mvn = torch.distributions.MultivariateNormal(loc=m0, covariance_matrix=P0)
        loglikelihood = mvn.log_prob(x)

        return loglikelihood.unsqueeze(1)

    def p0_grad(self, x):

        P0_inv = torch.eye(self.state_dim, device=x.device)
        m0 = torch.zeros((x.shape[0], self.state_dim), device=x.device) + self.F
        likelihood = self.p0(x)
        gradient = (-torch.matmul(P0_inv, (x - m0).t()) * likelihood.t()).t()
        return gradient
    
    def log_p0_grad(self, x):

        P0 = torch.eye(self.state_dim)
        m0 = torch.zeros(self.state_dim)
        m0 = m0.unsqueeze(0).repeat(x.shape[0], 1)

        gradient = (-torch.matmul(torch.linalg.inv(P0), (x - m0).t())).t()
        return gradient

    def get_prior_moments(self, device):

        P0 = torch.eye(self.state_dim, device=device)
        m0 = torch.zeros(self.state_dim, device=device) + self.F.to(device)
        return m0, P0

    def get_kf_params(self, deltat, device):

        raise NotImplementedError("Kalman filter parameters are not defined for nonlinear Lorenz–96. Use EKF/EnKF.")

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

        R = torch.eye(self.obs_dim, device=device) * (self.obs_noise_std ** 2)
        Hx = self.H_cpu if device == torch.device('cpu') else self.H
        Hr = torch.eye(self.obs_dim, device=device)
        return R, Hx, Hr

    def _jacobian_drift(self, x):
        """
        Jacobian J of mu at x.
        Accepts x of shape (d,) or (B,d).
        Returns (d,d) if x is (d,), else (B,d,d).
        """
        squeeze_out = False
        if x.dim() == 1:
            x = x.unsqueeze(0)           # (1,d)
            squeeze_out = True
        B, d = x.shape

        J = torch.zeros((B, d, d), device=x.device, dtype=x.dtype)
        x_ip1 = self._roll(x, -1)
        x_im1 = self._roll(x, +1)
        x_im2 = self._roll(x, +2)

        idx = torch.arange(d, device=x.device)
        im2 = (idx - 2) % d
        im1 = (idx - 1) % d
        ip1 = (idx + 1) % d

        J[:, idx, im2] += -x_im1
        J[:, idx, im1] += (x_ip1 - x_im2)
        J[:, idx, idx] += -1.0
        J[:, idx, ip1] += x_im1

        return J[0] if squeeze_out else J

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

        x = x.unsqueeze(0)  # (1,d)
        B, d = x.shape

        sigma = self.sigma_cpu if device == torch.device('cpu') else self.sigma
        sigma = sigma.to(x.device, x.dtype)
        m = sigma.shape[1]

        dt = torch.full((B, 1), float(deltat), device=x.device, dtype=x.dtype)

        q = q.to(x.device, x.dtype)
        qB = q.view(1, -1).expand(B, -1)

        mu = self.drift_batch(x)
        J  = self._jacobian_drift(x)

        f = x + mu * dt + qB.matmul(sigma.t())

        I = torch.eye(d, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, d, d)
        Fx = I + J * dt.view(B, 1, 1)
        Fq = sigma.unsqueeze(0).expand(B, d, m)

        Q = torch.eye(m, device=x.device, dtype=x.dtype).unsqueeze(0).expand(B, m, m) \
            * dt.view(B, 1, 1)

        f  = f[0]          # (d,)
        Fx = Fx[0]         # (d,d)
        Fq = Fq[0]         # (d,m)
        Q  = Q[0]          # (m,m)

        return f, Q, Fx, Fq
    
    def get_deep_filter_params(self, X):

        B, d = X.shape
        a1 = torch.zeros((B, d, d), device=X.device, dtype=X.dtype)
        a2 = torch.zeros((B, d, d), device=X.device, dtype=X.dtype)
        a3 = -torch.ones((B, d), device=X.device, dtype=X.dtype)
        a4 = self.drift_batch(X)
        return a1, a2, a3, a4