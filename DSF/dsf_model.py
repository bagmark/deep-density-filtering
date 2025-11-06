import torch
import torch.nn as nn

class DSFModel(nn.Module):
    def __init__(self, params, problem, n_obs, idx):
        super(DSFModel, self).__init__()
        self.n_obs = n_obs
        self.idx = idx 

        self.params = params
        self.fixed_size = params['fixed_input_size']
        self.reverse_obs_order = params['reverse_obs_order']

        self.problem = problem
        state_dim = problem.state_dim
        obs_dim = problem.obs_dim

        self.hidden_dim = params['hidden_dim']
        self.num_hidden = params['num_layers'] - 2
        self.fixed_size = params['fixed_input_size']
        
        if self.fixed_size:
            self.dim_in = state_dim + obs_dim * params['n_obs_max']
        else:
            self.dim_in = state_dim + obs_dim * n_obs

        self.dim_out = 1

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

        return torch.exp(x)
