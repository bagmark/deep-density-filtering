import torch

from problems.problem import Problem

class Schlogel(Problem):
    def __init__(self, theta=(3e-7, 1e-4, 1e-3, 3.5), A=1e5, B=2e5, obs_noise_std=0.5, prior_std=15.):
        self.identifier = "schlogel"
        self.obs_noise_std = obs_noise_std
        self.prior_std = prior_std
        self.state_dim = 1
        self.obs_dim = 1
        self.theta = theta
        self.A = A
        self.B = B
        self.smallest_val = torch.tensor(1e-200)
        self.std_1 = 10
        self.std_2 = 10

    def drift_batch(self, X):
        θ1, θ2, θ3, θ4 = self.theta
        x = X.squeeze(-1)
        term1 = θ1 * self.A * x * (x - 1) / 2
        term2 = -θ2 * x * (x - 1) * (x - 2) / 6
        term3 = θ3 * self.B
        term4 = -θ4 * x
        return (term1 + term2 + term3 + term4).unsqueeze(-1)

    def diffusion_batch(self, X):
        θ1, θ2, θ3, θ4 = self.theta
        x = X.squeeze(-1)

        s1 = θ1 * self.A * x * (x - 1) / 2
        s2 = θ2 * x * (x - 1) * (x - 2) / 6
        s3 = θ3 * self.B
        s4 = θ4 * x

        sigma_sq = s1 + s2 + s3 + s4
        sigma = torch.sqrt(sigma_sq).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        return sigma
    
    def obs_batch(self, X):
        return torch.log1p(X)

    def get_deep_filter_params(self, X):
        B = X.shape[0]
        θ1, θ2, θ3, θ4 = self.theta
        A, B_ = self.A, self.B
        x = X.squeeze(-1)

        mu = self.drift_batch(X)

        dmu_dx = (
            θ1 * A * (2 * x - 1) / 2
            - θ2 * (3 * x**2 - 6 * x + 2) / 6
            - θ4
        ).unsqueeze(-1)

        ds1 = θ1 * A * (2 * x - 1) / 2
        ds2 = θ2 * (3 * x**2 - 6 * x + 2) / 6
        ds4 = θ4 * torch.ones_like(x)

        da_dx = ds1 + ds2 + ds4
        d2a_dx2 = θ1 * A + θ2 * (x - 1)

        a1 = da_dx.view(B, 1, 1)
        a2 = d2a_dx2.view(B, 1, 1)
        a3 = dmu_dx
        a4 = mu

        return a1, a2, a3, a4

    def get_obs_jacobian(self, X):

        x = X.squeeze(-1)
        jac = (1.0 / (1.0 + x)).unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)

        return jac
    
    def sample_q0(self, N_simulations, device):
        # Sample from a mixture: 50% from N(150, std_1^2), 50% from N(350, std_2^2)
        mixture_mask = torch.rand(N_simulations, device=device) < 0.5  # (B,)
        base_samples = torch.randn((N_simulations, self.state_dim), device=device)

        samples = torch.where(
            mixture_mask.unsqueeze(1),
            150. + 2.5*self.std_1 * base_samples,
            375. + 6*self.std_2 * base_samples
        )
        return samples

    def sample_p0(self, N_simulations, device):
        # Sample from a mixture: 50% from N(150, std_1^2), 50% from N(350, std_2^2)
        mixture_mask = torch.rand(N_simulations, device=device) < 0.5  # (B,)
        base_samples = torch.randn((N_simulations, self.state_dim), device=device)

        samples = torch.where(
            mixture_mask.unsqueeze(1),
            150. + self.std_1 * base_samples,
            350. + self.std_2 * base_samples
        )
        return samples

    def p0(self, x):
        # Evaluate the PDF of the Gaussian mixture
        m1 = torch.tensor([150.], device=x.device)
        P1 = torch.eye(self.state_dim, device=x.device) * self.std_1 ** 2
        mvn1 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m1, covariance_matrix=P1)

        m2 = torch.tensor([350.], device=x.device)
        P2 = torch.eye(self.state_dim, device=x.device) * self.std_2 ** 2
        mvn2 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m2, covariance_matrix=P2)

        pdf1 = torch.exp(mvn1.log_prob(x))
        pdf2 = torch.exp(mvn2.log_prob(x))

        return 0.5 * (pdf1 + pdf2).unsqueeze(1)

    def log_p0(self, x):
        # Use log-sum-exp trick for numerical stability
        m1 = torch.tensor([150.], device=x.device)
        P1 = torch.eye(self.state_dim, device=x.device) * self.std_1 ** 2
        mvn1 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m1, covariance_matrix=P1)

        m2 = torch.tensor([350.], device=x.device)
        P2 = torch.eye(self.state_dim, device=x.device) * self.std_2 ** 2
        mvn2 = torch.distributions.multivariate_normal.MultivariateNormal(loc=m2, covariance_matrix=P2)

        logp1 = mvn1.log_prob(x)
        logp2 = mvn2.log_prob(x)

        stacked = torch.stack([logp1, logp2], dim=1)  # (B, 2)
        log_mix = torch.logsumexp(stacked, dim=1) - torch.log(torch.tensor(2.0, device=x.device))
        return log_mix.unsqueeze(1)

    def p0_grad(self, x):
        # Gradient of mixture: weighted gradient of components
        m1 = torch.tensor([150.], device=x.device).expand_as(x)
        P1inv = torch.eye(self.state_dim, device=x.device) / self.std_1**2
        grad1 = -torch.matmul(x - m1, P1inv)

        m2 = torch.tensor([350.], device=x.device).expand_as(x)
        P2inv = torch.eye(self.state_dim, device=x.device) / self.std_2**2
        grad2 = -torch.matmul(x - m2, P2inv)

        # Compute weights for each component
        logp1 = -0.5 * torch.sum((x - m1) @ P1inv * (x - m1), dim=1)
        logp2 = -0.5 * torch.sum((x - m2) @ P2inv * (x - m2), dim=1)

        stacked = torch.stack([logp1, logp2], dim=1)  # (B, 2)
        log_weights = stacked - torch.logsumexp(stacked, dim=1, keepdim=True)  # normalized log weights
        weights = torch.exp(log_weights)  # (B, 2)

        return weights[:, 0:1] * grad1 + weights[:, 1:2] * grad2  # weighted sum
    
    
    def log_p0_grad(self, x):
        """
        Gradient of log p0(x) for a 1D two-component Gaussian mixture with equal mixing weights (0.5, 0.5).
        x: (B, 1) or (B,)
        returns: (B, 1)
        """
        device, dtype = x.device, x.dtype
        x_flat = x.squeeze(-1)  # (B,)

        m1 = torch.tensor(150.0, device=device, dtype=dtype)
        m2 = torch.tensor(350.0, device=device, dtype=dtype)
        s1 = torch.as_tensor(self.std_1, device=device, dtype=dtype)
        s2 = torch.as_tensor(self.std_2, device=device, dtype=dtype)

        # Component scores: d/dx log N(x|m, s^2) = -(x - m)/s^2
        grad1 = -(x_flat - m1) / (s1 * s1)   # (B,)
        grad2 = -(x_flat - m2) / (s2 * s2)   # (B,)

        # Responsibilities γ_k(x) using full log-densities (constants included)
        # log N(x|m, s^2) = -0.5*((x-m)^2 / s^2) - 0.5*log(2π s^2)
        logp1 = -0.5 * ((x_flat - m1)**2) / (s1 * s1) - 0.5 * torch.log(2 * torch.pi * (s1 * s1))
        logp2 = -0.5 * ((x_flat - m2)**2) / (s2 * s2) - 0.5 * torch.log(2 * torch.pi * (s2 * s2))

        stacked = torch.stack([logp1, logp2], dim=1)  # (B,2)
        # Equal mixing weights => adding log(0.5) to each cancels in normalization, so omit
        log_resp = stacked - torch.logsumexp(stacked, dim=1, keepdim=True)  # (B,2)
        resp = torch.exp(log_resp)  # (B,2)

        # Mixture score = sum_k γ_k * component score_k
        grad = resp[:, 0] * grad1 + resp[:, 1] * grad2  # (B,)
        return grad.unsqueeze(1)  # (B,1)

    def get_prior_moments(self, device):
        # Approximate moments of the mixture
        m0 = 0.5 * (150. + 350.)
        v0 = 0.5 * (self.std_1 ** 2 + self.std_2 ** 2 + (150. - m0) ** 2 + (350. - m0) ** 2)
        mean = torch.tensor([m0], device=device)
        cov = torch.eye(self.state_dim, device=device) * v0
        return mean, cov

    def get_kf_params(self, deltat, device):
        raise ValueError("The specified SDE is not linear. Cannot use Kalman Filter")

    def get_ekf_update_params(self, x, r):

        device = x.device

        R = torch.eye(self.obs_dim, device=device) * self.obs_noise_std**2
        Hx = torch.tensor([[1.0 / (1.0 + x.item())]], device=device)
        Hr = torch.eye(self.obs_dim, device=device)

        return R, Hx, Hr

    def get_ekf_pred_params(self, x, q, deltat):
        θ1, θ2, θ3, θ4 = self.theta
        x0 = x.squeeze(-1)

        drift = θ1 * self.A * x0 * (x0 - 1) / 2 \
              - θ2 * x0 * (x0 - 1) * (x0 - 2) / 6 \
              + θ3 * self.B - θ4 * x0

        Fx = 1 + deltat * (
            θ1 * self.A * (2 * x0 - 1) / 2
            - θ2 * (3 * x0**2 - 6 * x0 + 2) / 6
            - θ4
        )
        Fx = Fx.view(-1, 1)

        sigma = self.diffusion_batch(x).squeeze()  # (B,)
        f = x + drift.view(-1, 1) * deltat + sigma.view(-1, 1) * q

        Fq = sigma.view(-1, 1)
        Q = deltat.view(-1, 1)

        return f, Q, Fx, Fq