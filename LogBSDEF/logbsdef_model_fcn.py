import torch
import torch.nn as nn

class LogBSDEFModelFCN(nn.Module):
    def __init__(self, model_params, problem, n_obs, times):
        super().__init__()
        self.n_obs = n_obs
        self.times = times
        self.n_times = len(times)

        self.model_type = 'FCN'

        self.model_params = model_params

        self.u_params = model_params['u_params']
        self.v_params = model_params['v_params']
        self.fixed_size = model_params['fixed_input_size']
        self.reverse_obs_order = model_params['reverse_obs_order']
        self.u_params['fixed_input_size'] = self.fixed_size
        self.v_params['fixed_input_size'] = self.fixed_size
        self.u_params['reverse_obs_order'] = self.reverse_obs_order
        self.v_params['reverse_obs_order'] = self.reverse_obs_order

        if self.fixed_size:
            self.u_params['n_obs_max'] = model_params['n_obs_max']
            self.v_params['n_obs_max'] = model_params['n_obs_max']

        self.problem = problem
        state_dim = problem.state_dim
        obs_dim = problem.obs_dim

        self.u = UModel(self.u_params, n_obs, state_dim, obs_dim)
        self.v_models = []
        for time in self.times[:-1]:
            self.v_models = self.v_models + [VModel(self.v_params, n_obs, state_dim, obs_dim)]
        self.v_models = nn.ModuleList(self.v_models)

    def predict(self, state, obs):
        """ Prediction density """

        with torch.no_grad():
            out = torch.exp(-self.u(state, obs))

        return out
    
    def filter_unnorm(self, state, obs):
        """ Unnormalized filtering density """
        # state: (B, d), obs: (B, l, d')
        obs_prev = obs[:, :-1, :]
        obs_last = obs[:, -1, :]
        
        pred = self.predict(state, obs_prev)
        likelihood = self.problem.obs_likelihood(state, obs_last).unsqueeze(1)
        return pred * likelihood
    
    def logfilter_unnorm(self, state, obs):
        """ Unnormalized logfiltering density """
        obs_prev = obs[:, :-1, :]
        obs_last = obs[:, -1, :]
        
        logpred = -self.u(state, obs_prev)
        loglikelihood = self.problem.obs_loglikelihood(state, obs_last).unsqueeze(1)
        return logpred + loglikelihood
    
    def forward(self, state, obs, deltaW):
        # state: (B, n_times, d), obs: (B, (l-1), d'), deltaW: (B, n_times, d)
        X = state[:, 0, :]
        Y = self.u(X, obs)

        vs = torch.zeros((X.shape[0], self.n_times - 1, X.shape[1]))
        for i in range(self.n_times - 1):
            vs[:, i, :] = self.v_models[i](state[:, i, :], obs)

        for i in range(self.n_times - 1):
            deltat = self.times[i+1] - self.times[i]
            deltaw = deltaW[:, i, :]
            X = state[:, i, :]

            v = vs[:, i, :]
            Z = torch.matmul(self.problem.diffusion_batch(X).transpose(1, 2), v.unsqueeze(2))

            f = self.problem.log_bsdef_f(X, Y, v)

            Y = Y  - deltat * f + torch.matmul(Z.transpose(1,2), deltaw.unsqueeze(2))[:, :, 0]

        # Y: (B, 1)
        return Y
    
class UModel(nn.Module):
    def __init__(self, params, n_obs, state_dim, obs_dim):
        super(UModel, self).__init__()
        self.n_obs = n_obs

        self.hidden_dim = params['hidden_dim']
        self.num_hidden = params['num_layers'] - 2
        self.fixed_size = params['fixed_input_size']
        self.reverse_obs_order = params['reverse_obs_order']
        
        if self.fixed_size:
            self.dim_in = state_dim + obs_dim * params['n_obs_max']
        else:
            self.dim_in = state_dim + obs_dim * n_obs

        self.dim_out = 1

        self.params = params

        self.linear_in = nn.Linear(self.dim_in, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.dim_out)
        self.hidden_layers = [
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                    for _ in range(self.num_hidden)
                ]
        
        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)

    def forward(self, state, obs):
        # state: (B, d), obs: (B, (l-1), d')
        B, _ = state.shape

        if self.reverse_obs_order:
            obs = torch.flip(obs, dims=(1,))

        if self.n_obs > 0:
            obs_flattened = obs.reshape(B, -1)
            x = torch.cat((state, obs_flattened), dim=1)
        else:
            x = state

        if self.fixed_size:
            _, in_size = x.shape
            zeros = torch.zeros((B, self.dim_in - in_size), device=x.device)
            x = torch.cat((x, zeros), dim=1)

        x = self.linear_in(x)
        x = torch.relu(x)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = torch.relu(x)
        
        x = self.linear_out(x)

        return x

class VModel(nn.Module):
    def __init__(self, params, n_obs, state_dim, obs_dim):
        super(VModel, self).__init__()
        self.n_obs = n_obs

        self.hidden_dim = params['hidden_dim']
        self.num_hidden = params['num_layers'] - 2
        self.fixed_size = params['fixed_input_size']
        self.reverse_obs_order = params['reverse_obs_order']
        
        if self.fixed_size:
            self.dim_in = state_dim + obs_dim * params['n_obs_max']
        else:
            self.dim_in = state_dim + obs_dim * n_obs
        self.dim_out = state_dim

        self.params = params

        self.linear_in = nn.Linear(self.dim_in, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, self.dim_out)
        self.hidden_layers = [
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                    for _ in range(self.num_hidden)
                ]
        
        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)

    def forward(self, state, obs):
        # state: (B, d), obs: (B, (l-1), d')
        B, _ = state.shape

        if self.reverse_obs_order:
            obs = torch.flip(obs, dims=(1,))

        if self.n_obs > 0:
            obs_flattened = obs.reshape(B, -1)
            x = torch.cat((state, obs_flattened), dim=1)
        else:
            x = state

        if self.fixed_size:
            _, in_size = x.shape
            zeros = torch.zeros((B, self.dim_in - in_size))
            x = torch.cat((x, zeros), dim=1)

        x = self.linear_in(x)
        x = torch.relu(x)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = torch.relu(x)
        
        x = self.linear_out(x)

        return x