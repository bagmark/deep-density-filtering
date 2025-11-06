import os

import pandas as pd

import torch

from scipy.stats import entropy

from logging import getLogger
from utils.utils import *

from utils.problem_simulator import ProblemSimulator

from benchmark_filters.particle_filter import ParticleFilter
from benchmark_filters.ensemble_kalman_filter import EnsembleKalmanFilter
from BSDEF.bsdef import BSDEF
from DSF.dsf import DSF
from LogBSDEF.logbsdef import LogBSDEF
from LogDSF.logdsf import LogDSF

class MetricEvaluator:
    def __init__(self, params, filters, reference_filter, problem):
        self.problem = problem

        self.state_dim = problem.state_dim
        self.obs_dim = problem.obs_dim

        self.reference_times = reference_filter.times
        self.n_times_reference = len(self.reference_times)
        self.reference_obs_indices = reference_filter.obs_indices
        self.obs_times = self.reference_times[self.reference_obs_indices]
        self.n_obs = len(self.obs_times)

        self.filters = filters
        self.reference_filter = reference_filter

        # Metrics to evaluate
        self.metrics_to_calculate = params['metrics']

        self.params = params
        self.grid_based = params["Grid-based"]
        
        self.min_value_integration = params["Integration range"][0]
        self.max_value_integration = params["Integration range"][1]
        self.points_per_dim = params["Integration points"]
        self.mc_samples = params["MC samples"]
        self.element_size = ((self.max_value_integration - self.min_value_integration)/(self.points_per_dim-1))**self.state_dim

        self.metrics = {}
        for metric in self.metrics_to_calculate:
            self.metrics[metric] = {}
            for filter_name, filter in filters.items():
                self.metrics[metric][filter_name] = torch.zeros(self.n_obs)

        if "LinfLinf" in self.metrics_to_calculate or "L2Linf" in self.metrics_to_calculate: 
            if not self.grid_based:
                raise ValueError("LinfLinf and L2Linf are only implemented using grid-based evaluation.")
            
        if "NLL" in self.metrics_to_calculate:
            if self.grid_based:
                raise ValueError("NLL is only implemented using grid-free evaluation.")

        if "MAE" in self.metrics_to_calculate: # Reference only for MAE and NLL
            self.metrics["MAE"]['reference'] = torch.zeros(self.n_obs)
        if "NLL" in self.metrics_to_calculate:
            self.metrics["NLL"]['reference'] = torch.zeros(self.n_obs)

        self.smallest_log = torch.tensor(1e-200)

        self.logger = getLogger(name='trainer')

    def evaluate_metrics(self, n_samples): 
        if self.grid_based:
            return self.evaluate_metrics_grid_based(n_samples)
        else:
            return self.evaluate_metrics_grid_free_log(n_samples)

    def evaluate_metrics_grid_based(self, n_samples):
        """ Evaluate metrics using grid-based approach """

        simulator = ProblemSimulator(self.problem, self.reference_times, self.reference_obs_indices)
        X, Y, _ = simulator.simulate_state_and_obs(N_simulations=n_samples)

        self.logger.info('=================================================================')

        for i in range(n_samples):
            Yi = Y[i, :, :] 

            if (i + 1) == 1 or ((i + 1) % 10 == 0):
                self.logger.info(f"Evaluating metrics on sample {i + 1}/{n_samples}")

            for filter_name, filter in self.filters.items():
                filter.filter(Yi)
                if isinstance(filter, ParticleFilter) or isinstance(filter, EnsembleKalmanFilter):
                    filter.build_kdes()

            self.reference_filter.filter(Yi)
            if isinstance(self.reference_filter, ParticleFilter) or isinstance(filter, EnsembleKalmanFilter):
                self.reference_filter.build_kdes()

            if "MAE" in self.metrics_to_calculate or "FME" in self.metrics_to_calculate:
                reference_filter_means = self.reference_filter.get_filter_means(Yi, device=X.device)

            states = X[i, self.reference_obs_indices, :]
            if "MAE" in self.metrics_to_calculate:
                self.metrics["MAE"]['reference'] += torch.norm(reference_filter_means - states, dim=1, p=2) / n_samples

            if "KLD" in self.metrics_to_calculate or "LinfLinf" in self.metrics_to_calculate or "L2Linf" in self.metrics_to_calculate:
                pdfs_reference, _ = self.reference_filter.get_filter_pdfs(Yi, X.device, self.min_value_integration, self.max_value_integration, self.points_per_dim)

            for filter_name, filter in self.filters.items():
 
                filter_means = filter.get_filter_means(Yi, device=X.device)
                
                if "MAE" in self.metrics_to_calculate:
                    self.metrics["MAE"][filter_name] += torch.norm(filter_means - states, dim=1, p=2) / n_samples

                if "FME" in self.metrics_to_calculate:
                    self.metrics["FME"][filter_name] += torch.norm(filter_means - reference_filter_means, dim=1, p=2) / n_samples

                if "KLD" in self.metrics_to_calculate or "LinfLinf" in self.metrics_to_calculate or "L2Linf" in self.metrics_to_calculate:
                    pdfs_filter, _ = filter.get_filter_pdfs(Yi, X.device, self.min_value_integration, self.max_value_integration, self.points_per_dim)

                if "KLD" in self.metrics_to_calculate:
                    pdfs_filter += 10 ** (-30) # Add a very small number to prevent pdf from being 0
                    self.metrics["KLD"][filter_name] += torch.tensor(entropy(pdfs_reference.detach().cpu().numpy(), pdfs_filter.detach().cpu().numpy(), axis=1)) / n_samples
                    
                if "L2Linf" in self.metrics_to_calculate:
                    pdfs_diff = pdfs_filter - pdfs_reference
                    self.metrics["L2Linf"][filter_name] += torch.max(torch.pow(pdfs_diff,2), dim=1)[0] / n_samples

                if "LinfLinf" in self.metrics_to_calculate:
                    pdfs_diff = pdfs_filter - pdfs_reference
                    Linf = torch.max(torch.abs(pdfs_diff), dim=1)[0]
                    for jx in range(self.n_obs):
                        self.metrics["LinfLinf"][filter_name][jx] = torch.max(self.metrics["LinfLinf"][filter_name][jx],Linf[jx])

        if "L2Linf" in self.metrics_to_calculate:
            for filter_name, filter in self.filters.items():
                self.metrics["L2Linf"][filter_name] = torch.sqrt(self.metrics["L2Linf"][filter_name])


        self.logger.info('***Evaluation Done***')
        self.logger.info('=================================================================')

    def evaluate_metrics_grid_free_log(self, n_samples):
        """ Evaluate metrics using grid-free approach with log densities"""

        simulator = ProblemSimulator(self.problem, self.reference_times, self.reference_obs_indices)
        X, Y, _ = simulator.simulate_state_and_obs(N_simulations=n_samples)

        self.logger.info('=================================================================')
        
        for i in range(n_samples):
            Yi = Y[i, :, :] 

            if ((i + 1) % 10 == 0) or (i + 1) == 1:
                self.logger.info(f"Evaluating metrics on sample {i + 1}/{n_samples}")

            for filter_name, filter in self.filters.items():
                filter.filter(Yi)
                if isinstance(filter, ParticleFilter) or isinstance(filter, EnsembleKalmanFilter):
                    filter.build_kdes()

            self.reference_filter.filter(Yi)
            if isinstance(self.reference_filter, ParticleFilter) or isinstance(self.reference_filter, EnsembleKalmanFilter):
                self.reference_filter.build_kdes()

            if "MAE" in self.metrics_to_calculate or "FME" in self.metrics_to_calculate:
                reference_filter_means = self.reference_filter.get_filter_means(Yi, X.device)

            states = X[i, self.reference_obs_indices, :]
            if "MAE" in self.metrics_to_calculate:
                self.metrics["MAE"]['reference'] += torch.norm(reference_filter_means - states, dim=1, p=2) / n_samples

            if "NLL" in self.metrics_to_calculate:
                NLLs_reference = torch.zeros(self.n_obs)
                for j in range(self.n_obs):
                    x = states[j].unsqueeze(0)
                    filter_values = self.reference_filter.evaluate_filter_pdf(j, x, X.device)
                    logfilter_values = torch.log(torch.maximum(filter_values, self.smallest_log))

                    NLLs_reference[j] = - logfilter_values

                self.metrics["NLL"]['reference'] += NLLs_reference / n_samples


            if "KLD" in self.metrics_to_calculate:
                reference_samples, reference_values = self.reference_filter.sample_filter_densities(Yi, self.mc_samples, X.device)
                logreference_values = torch.log(torch.maximum(reference_values, self.smallest_log))

            for filter_name, filter in self.filters.items():
                if isinstance(filter, BSDEF) or isinstance(filter, DSF):
                    filter_consts, filter_means = filter.get_filter_means_and_consts(Yi)
                elif isinstance(filter, LogBSDEF) or isinstance(filter, LogDSF):
                    logfilter_consts, filter_means = filter.get_filter_means_and_logconsts(Yi)
                else: 
                    filter_means = filter.get_filter_means(Yi, X.device)

                if "MAE" in self.metrics_to_calculate:
                    self.metrics["MAE"][filter_name] += torch.norm(filter_means - states, dim=1, p=2) / n_samples

                if "FME" in self.metrics_to_calculate:
                    self.metrics["FME"][filter_name] += torch.norm(filter_means - reference_filter_means, dim=1, p=2) / n_samples

                if "KLD" in self.metrics_to_calculate:
                    KLDs = torch.zeros(self.n_obs)
                if "NLL" in self.metrics_to_calculate:
                    NLLs = torch.zeros(self.n_obs)

                for j in range(self.n_obs):

                    if "KLD" in self.metrics_to_calculate:
                        x = reference_samples[j,:,:]
                        if isinstance(filter, BSDEF) or isinstance(filter, DSF):
                            filter_values = filter.evaluate_filter_pdf(j, x, Yi, normalized=False) / filter_consts[j]
                            logfilter_values = torch.log(torch.maximum(filter_values, self.smallest_log))
                        elif isinstance(filter, LogBSDEF) or isinstance(filter, LogDSF):
                            logfilter_values = filter.evaluate_logfilter_pdf(j, x, Yi) 
                            logfilter_values = logfilter_values - logfilter_consts[j]
                        else: 
                            filter_values = filter.evaluate_filter_pdf(j, x, X.device)
                            logfilter_values = torch.log(torch.maximum(filter_values, self.smallest_log))


                        normalised_ref_values = torch.maximum(reference_values[j,:], self.smallest_log)
                        normalised_ref_values = normalised_ref_values/torch.sum(normalised_ref_values)
                        
                        KLDs[j] = torch.maximum(torch.mean(logreference_values[j,:] -  logfilter_values) , self.smallest_log)

                    if "NLL" in self.metrics_to_calculate:
                        x = states[j].unsqueeze(0)

                        if isinstance(filter, BSDEF) or isinstance(filter, DSF):
                            filter_values = filter.evaluate_filter_pdf(j, x, Yi, normalized=False) / filter_consts[j]
                            logfilter_values = torch.log(torch.maximum(filter_values, self.smallest_log))
                        elif isinstance(filter, LogBSDEF) or isinstance(filter, LogDSF):
                            logfilter_values = filter.evaluate_logfilter_pdf(j, x, Yi) 
                            logfilter_values = logfilter_values - logfilter_consts[j]
                        else: 
                            filter_values = filter.evaluate_filter_pdf(j, x, X.device)
                            logfilter_values = torch.log(torch.maximum(filter_values, self.smallest_log))

                        NLLs[j] = - logfilter_values

                if "KLD" in self.metrics_to_calculate:
                    self.metrics["KLD"][filter_name] += KLDs / n_samples

                if "NLL" in self.metrics_to_calculate:
                    self.metrics["NLL"][filter_name] += NLLs / n_samples

        self.logger.info('***Evaluation Done***')
        self.logger.info('=================================================================')

    def save_metric_results(self, folder_name):
        self.logger.info('***Saving Results***')

        current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_name = os.path.join(current_dir, "results", folder_name)
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

        columns = ["Timepoint"]
        for filter_name, filter in self.filters.items():
            columns.append(filter_name)
        columns_mae_nll = columns.copy()
        columns_mae_nll.append("reference")

        for metric in self.metrics_to_calculate:
            if metric == "MAE":
                columns_ = columns_mae_nll
            elif metric == "NLL":
                columns_ = columns_mae_nll
            else:
                columns_ = columns
            df = pd.DataFrame(columns = columns_)

            row = {}
            for i in range(self.n_obs):

                row["Timepoint"] = np.round(self.obs_times[i].detach().cpu().item(), 4)

                for filter_name, filter in self.filters.items():
                    row[filter_name] = self.metrics[metric][filter_name][i].detach().cpu().item()
                
                if metric == "MAE":
                    row["reference"] = self.metrics["MAE"]['reference'][i].detach().cpu().item()

                if metric == "NLL":
                    row["reference"] = self.metrics["NLL"]['reference'][i].detach().cpu().item()

                df.loc[len(df)] = row

            path = os.path.join(folder_name, metric)
            df.to_csv(path, index=False)

        self.logger.info(f'Results saved in folder: {folder_name}')