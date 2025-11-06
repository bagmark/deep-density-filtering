import torch
import torch.nn as nn

class LogBSDEFModelLSTM(nn.Module):
    def __init__(self, model_params, problem, n_obs, times):
        super().__init__()
        self.n_obs = n_obs
        self.times = times
        self.n_times = len(times)

        self.model_type = 'LSTM'

        self.model_params = model_params

        self.problem = problem

        if self.model_params['u_encoder_params']['return_cell_state']:
            self.model_params['u_decoder_params']['input_dim'] = self.model_params['u_encoder_params']['hidden_dim'] * 2
        else: 
            self.model_params['u_decoder_params']['input_dim'] = self.model_params['u_encoder_params']['hidden_dim']
        if self.model_params['v_encoder_params']['return_cell_state']:
            self.model_params['v_decoder_params']['input_dim'] = self.model_params['v_encoder_params']['hidden_dim'] * 2
        else: 
            self.model_params['v_decoder_params']['input_dim'] = self.model_params['v_encoder_params']['hidden_dim']                

        self.u_encoder = LSTMEncoder(model_params['u_encoder_params'], problem, n_obs)
        self.v_encoder = LSTMEncoder(model_params['v_encoder_params'], problem, n_obs)

        self.u_decoder = FCNDecoder(model_params['u_decoder_params'], problem, n_obs)
        self.v_decoders = []
        for time in self.times[:-1]:
            self.v_decoders = self.v_decoders + [FCNDecoder(model_params['v_decoder_params'], problem, n_obs, problem.state_dim)]
        self.v_decoders = nn.ModuleList(self.v_decoders)
    
    def predict(self, state, obs):
        """ Prediction density """
        
        with torch.no_grad():
            emb = self.u_encoder(obs)
            out = torch.exp(-self.u_decoder(state, emb))

        return out
        
    def logpredict(self, state, obs):
        """ Log prediction density """

        with torch.no_grad():
            emb = self.u_encoder(obs)
            out = -self.u_decoder(state, emb)

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
        """ Unnormalized filtering density """
        # state: (B, d), obs: (B, l, d')

        obs_prev = obs[:, :-1, :]
        obs_last = obs[:, -1, :]
        
        logpred = self.logpredict(state, obs_prev)
        loglikelihood = self.problem.obs_loglikelihood(state, obs_last).unsqueeze(1)
        return logpred + loglikelihood

    def forward(self, state, obs, deltaW):
        # state: (B, n_times, d), obs: (B, (l-1), d'), deltaW: (B, n_times, d)

        X = state[:, 0, :]

        emb = self.u_encoder(obs)
        Y = self.u_decoder(X, emb)

        vs = torch.zeros((X.shape[0], self.n_times - 1, X.shape[1]))

        emb = self.v_encoder(obs)
        for i in range(self.n_times - 1):
            vs[:, i, :] = self.v_decoders[i](state[:, i, :], emb)

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
    
    def encode(self, obs):
        obs_prev = obs[:, :-1, :]
        with torch.no_grad():
            emb = self.u_encoder(obs_prev)

        return emb
    
    def logfilter_unnorm_dec(self, state, emb, obs_last):
        """ Unnormalized filtering density given embeddings from encoder"""
        
        logpred = -self.u_decoder(state, emb)
        loglikelihood = self.problem.obs_loglikelihood(state, obs_last).unsqueeze(1)

        return logpred + loglikelihood
    
    def filter_unnorm_dec(self, state, emb, obs_last):
        """ Unnormalized filtering density given embeddings from encoder"""
        
        logpred = -self.u_decoder(state, emb)
        loglikelihood = self.problem.obs_loglikelihood(state, obs_last).unsqueeze(1)

        return torch.exp(logpred + loglikelihood)
        

class LSTMEncoder(nn.Module):
    def __init__(self, model_params, problem, n_obs):
        super().__init__()
        self.n_obs = n_obs

        self.model_params = model_params

        if n_obs > 0:
            self.n_inputs = n_obs
        else: 
            self.n_inputs = 1
        
        self.input_dim = problem.obs_dim
        self.hidden_dim = model_params['hidden_dim']
        self.n_layers = model_params['n_layers']
        self.dropout = model_params['dropout']
        self.reverse_obs_order = model_params['reverse_obs_order']

        self.return_cell = model_params['return_cell_state']

        self.lstm = nn.LSTM(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=self.n_layers, dropout=self.dropout, batch_first=True)

    def forward(self, obs):
        # obs: (B, (l-1), d')
        B, _, _ = obs.shape

        if self.reverse_obs_order:
            obs = torch.flip(obs, dims=(1,))

        if self.n_obs > 0:
            x = obs

            start_token = torch.zeros((B, 1, self.input_dim))
            x = torch.cat((start_token, x), dim=1)
        else:
            x = torch.zeros((B, self.n_inputs, self.input_dim))

        output, (hidden, cell) = self.lstm(x)

        if not self.return_cell:
            return hidden[-1]
        else:
            return torch.cat((hidden[-1], cell[-1]), dim=1)
        
class FCNDecoder(nn.Module):
    def __init__(self, model_params, problem, n_obs, output_dim=1):
        super().__init__()
        self.n_obs = n_obs

        self.model_params = model_params
        
        self.input_dim = problem.state_dim + model_params['input_dim']

        self.hidden_dim = model_params['hidden_dim']
        self.n_hidden = model_params['n_layers'] - 2
        self.embed = model_params['embed']
        self.output_dim = output_dim

        if self.embed:
            self.embedder = nn.Linear(problem.state_dim, self.hidden_dim)
            self.input_dim -= problem.state_dim
            self.input_dim += self.hidden_dim

        self.linear_in = nn.Linear(self.input_dim, self.hidden_dim)
        self.linear_out = nn.Linear(self.hidden_dim, output_dim)
        self.hidden_layers = [
                    nn.Linear(self.hidden_dim, self.hidden_dim)
                    for _ in range(self.n_hidden)
                ]
        
        self.hidden_layers = torch.nn.ModuleList(self.hidden_layers)

    def forward(self, x, emb):
        # x: (B, d)
        # emb: (B, emb_dim)

        if self.embed:
            x = self.embedder(x)

        x = torch.cat((x, emb), dim=1)

        x = self.linear_in(x)
        x = torch.relu(x)

        for i, layer in enumerate(self.hidden_layers):
            x = layer(x)
            x = torch.relu(x)
        
        x = self.linear_out(x)

        return x