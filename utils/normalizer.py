import torch

from utils.utils import *

from benchmark_filters.extended_kalman_filter import ExtendedKalmanFilter

# Class with methods for normalizing densities

class Normalizer():
    def __init__(self, params):

        self.params = params
        self.method = params['method']

        self.problem = params['problem']
        self.state_dim = self.problem.state_dim
        self.times = params['times']
        self.obs_indices = params['obs_indices']
        self.n_obs = len(self.obs_indices)

        self.smallest_log = torch.tensor(1e-200)

        self.encode_all_first = params.get('encode_all_first', False)

        if (self.method == "I-EKF") or (self.method == "I-G"):
            self.import_variance = params.get("importance_variance", 5)
            self.n_samples = self.params['n_samples']

        if self.method == "I-EKF":
            self.importance_ekf_timestep = params.get("importance_ekf_timestep", 2)
            self.times_ekf = torch.linspace(self.times[0], self.times[-1], self.n_obs*self.importance_ekf_timestep + 1)
            self.obs_ind_ekf = torch.arange(self.importance_ekf_timestep, len(self.times), self.importance_ekf_timestep)

        elif self.method == "I-G":
            self.importance_mean = params.get("importance_mean", torch.zeros((self.n_obs, self.problem.state_dim)))

        elif self.method == "Quad":
            self.x_min = self.params['x_min']
            self.x_max = self.params['x_max']
            self.n_points = self.params['n_points']

            points = [torch.linspace(self.x_min, self.x_max, self.n_points) for _ in range(self.state_dim)]
            mesh = torch.meshgrid(*points, indexing='ij')
            self.x = torch.vstack([dimension.ravel() for dimension in mesh]).T
            self.dx = ((self.x_max - self.x_min) / (self.n_points - 1))**self.state_dim
            self.n_points_tot = self.n_points ** self.state_dim

    def _get_nc_quad(self, density, obs, encode=None):

        batch_size = obs.shape[0]
        norm_consts = torch.zeros((batch_size, 1))

        if self.encode_all_first: # Use that encoder can be run once per observation irrespective of state value x
            emb = encode(obs)

        for i in range(batch_size):
            if self.encode_all_first:
                emb_spec = emb[i]
                emb_spec_exp = emb_spec.unsqueeze(0).expand(self.n_points_tot, -1)

                obs_last = obs[i, -1, :]
                obs_last_exp = obs_last.unsqueeze(0).expand(self.n_points_tot, -1)

                dens_values = density(self.x, emb_spec_exp, obs_last_exp)
            else:
                obs_spec = obs[i, :, :]
                obs_spec_exp = obs_spec.unsqueeze(0).expand(self.n_points_tot, -1, -1)
                
                dens_values = density(self.x, obs_spec_exp)

            norm_consts[i] = torch.sum(dens_values) * self.dx

        return norm_consts
    
    def _get_nc_i_ekf(self, density, obs, encode=None, get_fm=False):
        """
        Importance sampling from ekf filter distribution
        Optional calculation of first moment
        """
        
        batch_size = obs.shape[0]
        norm_consts = torch.zeros((batch_size, 1))
        if get_fm:
            means = torch.zeros((batch_size, self.problem.state_dim))

        ekf = ExtendedKalmanFilter(self.problem, self.times_ekf, self.obs_ind_ekf)

        if self.encode_all_first: # Use that encoder can be run once per observation irrespective of state value x
            emb = encode(obs)

        for i in range(batch_size):
            # Filter to get importance density
            ekf.filter(obs[i], n_obs=obs.shape[1])
            idx = ekf.obs_indices[obs.shape[1]-1]
            m = ekf.m[idx, :].to(obs.device)
            P = self.import_variance*ekf.P[idx, :, :].to(obs.device)
        
            samples, importance_correction = _gaussian_importance(m, P, self.n_samples)

            if self.encode_all_first:
                emb_spec = emb[i]
                emb_spec_exp = emb_spec.unsqueeze(0).expand(self.n_samples, -1)

                obs_last = obs[i, -1, :]
                obs_last_exp = obs_last.unsqueeze(0).expand(self.n_samples, -1)

                log_density = torch.log(density(samples, emb_spec_exp, obs_last_exp))
            else:
                obs_spec_exp = obs[i].unsqueeze(0).expand(self.n_samples, -1, -1)
                log_density = torch.log(density(samples, obs_spec_exp))

            predictions = torch.exp(log_density - importance_correction)

            norm_consts[i] = torch.mean(predictions)
            if get_fm:
                means[i, :] = torch.mean(predictions * samples, dim=0) / norm_consts[i]

        if get_fm:
            return norm_consts, means
        else:
            return norm_consts

    def _get_nc_i_g(self, density, obs, encode=None, get_fm=False):
                
        batch_size = obs.shape[0]
        obs_index = obs.shape[1]
        norm_consts = torch.zeros((batch_size, 1))
        if get_fm:
            means = torch.zeros((batch_size, self.problem.state_dim))

        m = self.importance_mean[obs_index-1].to(obs.device) 
        P = self.import_variance*torch.eye(self.state_dim).to(obs.device)

        samples, importance_correction = _gaussian_importance(m, P, self.n_samples)

        if self.encode_all_first:
            emb = encode(obs)

        for i in range(batch_size):
            if self.encode_all_first:
                emb_spec = emb[i]
                emb_spec_exp = emb_spec.unsqueeze(0).expand(self.n_samples, -1)

                obs_last = obs[i, -1, :]
                obs_last_exp = obs_last.unsqueeze(0).expand(self.n_samples, -1)

                log_density = torch.log(density(samples, emb_spec_exp, obs_last_exp))
            else:
                obs_spec_exp = obs[i].unsqueeze(0).expand(self.n_samples, -1, -1)
                log_density = torch.log(density(samples, obs_spec_exp))

            predictions = torch.exp(log_density - importance_correction)
            norm_consts[i] = torch.mean(predictions)
            if get_fm:
                means[i, :] = torch.mean(predictions * samples, dim=0) / norm_consts[i]

        if get_fm:
            return norm_consts, means
        else:
            return norm_consts

    def _get_log_nc_i_ekf(self, log_density, obs, encode=None, get_fm=False):
        
        batch_size = obs.shape[0]
        lognorm_consts = torch.zeros((batch_size, 1))
        if get_fm:
            means = torch.zeros((batch_size, self.problem.state_dim))

        ekf = ExtendedKalmanFilter(self.problem, self.times_ekf, self.obs_ind_ekf)

        if self.encode_all_first:
            emb = encode(obs)

        for i in range(batch_size):
            ekf.filter(obs[i], n_obs=obs.shape[1])
            idx = ekf.obs_indices[obs.shape[1]-1]
            m = ekf.m[idx, :].to(obs.device)
            P = self.import_variance * ekf.P[idx, :, :].to(obs.device)

            samples, importance_correction = _gaussian_importance(m, P, self.n_samples)

            if self.problem.identifier == "schlogel":
                samples = torch.clamp(samples, min=1e-6)
                mvn = torch.distributions.MultivariateNormal(m, P)
                importance_correction = mvn.log_prob(samples).view(-1, 1)

            if self.encode_all_first:
                emb_spec = emb[i]
                emb_spec_exp = emb_spec.unsqueeze(0).expand(self.n_samples, -1)

                obs_last = obs[i, -1, :]
                obs_last_exp = obs_last.unsqueeze(0).expand(self.n_samples, -1)

                logfiltervalue = log_density(samples, emb_spec_exp, obs_last_exp)
            else:
                obs_spec_exp = obs[i].unsqueeze(0).expand(self.n_samples, -1, -1)
                logfiltervalue = log_density(samples,obs_spec_exp)

            weights = logfiltervalue - importance_correction
            max_weight = torch.max(weights)

            log_NC = -torch.log(torch.tensor(self.n_samples)) + max_weight + torch.log(torch.sum(torch.exp(weights - max_weight)))
            lognorm_consts[i] = log_NC

            if get_fm:
                new_weights = torch.exp(weights - max_weight)
                new_weights = new_weights/torch.sum(new_weights,dim=0)
                means[i, :] = torch.sum(new_weights * samples, dim=0)

        if get_fm:
            return lognorm_consts, means
        else:
            return lognorm_consts

    def _get_log_nc_i_g(self, log_density, obs, encode=None, get_fm=False):

        batch_size = obs.shape[0]
        obs_index = obs.shape[1]
        lognorm_consts = torch.zeros((batch_size, 1))
        if get_fm:
            means = torch.zeros((batch_size, self.problem.state_dim))

        if self.encode_all_first:
            emb = encode(obs)

        m = self.importance_mean[obs_index-1].to(obs.device) 
        P = self.import_variance*torch.eye(self.state_dim).to(obs.device)
        
        samples, importance_correction = _gaussian_importance(m, P, self.n_samples)

        for i in range(batch_size):
            if self.encode_all_first:
                emb_spec = emb[i]
                emb_spec_exp = emb_spec.unsqueeze(0).expand(self.n_samples, -1)

                obs_last = obs[i, -1, :]
                obs_last_exp = obs_last.unsqueeze(0).expand(self.n_samples, -1)

                logfiltervalue = log_density(samples, emb_spec_exp, obs_last_exp)
            else:
                obs_spec_exp = obs[i].unsqueeze(0).expand(self.n_samples, -1, -1)
                logfiltervalue = log_density(samples,obs_spec_exp)

            weights = logfiltervalue - importance_correction
            max_weight = torch.max(weights)

            log_NC = -torch.log(torch.tensor(self.n_samples)) + max_weight + torch.log(torch.sum(torch.exp(weights - max_weight)))
            lognorm_consts[i] = log_NC

            if get_fm:
                new_weights = torch.exp(weights - max_weight)
                new_weights = new_weights/torch.sum(new_weights,dim=0)
                means[i, :] = torch.sum(new_weights * samples, dim=0)

        if get_fm:
            return lognorm_consts, means
        else:
            return lognorm_consts

    def get_nc(self, density, obs, encode=None, get_fm=False):
        """
        Calculates normalization constant and optionally first moment
        density: callable function representing unnormalized density
        obs: (B, l, d')
        returns: (B, 1) normalization constants
        """

        if self.method == "Quad":
            return self._get_nc_quad(density, obs, encode) # No fm calculation for quad
        elif self.method == "I-EKF":
            return self._get_nc_i_ekf(density, obs, encode, get_fm=get_fm)
        elif self.method == "I-G":
            return self._get_nc_i_g(density, obs, encode, get_fm=get_fm)

    def get_log_nc(self, logdensity, obs, encode=None, get_fm=False):
        """
        Calculates normalization constant in log-version and optionally first moment
        Optimized for high-dimensional problems - used in Log-versions of the filters
        density: callable function representing unnormalized log-density
        obs: (B, l, d')
        returns: (B, 1) normalization constants
        """

        if self.method == "Quad":
            if self.encode_all_first:
                def exp_density(state, emb, obs_last):
                    return torch.exp(logdensity(state, emb, obs_last))
            else:
                def exp_density(x,obs):
                    return torch.exp(logdensity(x,obs))
            return torch.log(self._get_nc_quad(exp_density, obs, encode)) # No fm calculation for quad
        elif self.method == "I-EKF":
            return self._get_log_nc_i_ekf(logdensity, obs, encode, get_fm)
        elif self.method == "I-G":
            return self._get_log_nc_i_g(logdensity, obs, encode, get_fm)

def _gaussian_importance(mean, cov, n_samples):
    """
    Utility function to sample from a Gaussian importance distribution
    """
    mean = mean
    cov = cov
    mvn = torch.distributions.MultivariateNormal(mean, cov)
    samples = mvn.sample((n_samples,))

    importance_correction = mvn.log_prob(samples).view(-1, 1)

    return samples, importance_correction