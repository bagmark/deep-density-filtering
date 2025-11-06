# deep-density-filtering

Reference implementation for the Deep Splitting Filter (DSF), Backward Stochastic Differential Equation Filter (BSDEF) and their logarithmic variants, described in the articles: <br>

> "A convergent scheme for the Bayesian filtering problem based on the Fokker--Planck equation and deep splitting."
Kasper Bågmark, Adam Andersson, Stig Larsson, Filip Rydin.
https://arxiv.org/abs/2409.14585 <br>

> "Nonlinear filtering based on density approximation and deep BSDE prediction."
Kasper Bågmark, Adam Andersson, Stig Larsson.
https://arxiv.org/abs/2508.10630 <br>

> FILL IN

The code is written in Python using Pytorch. 

## Basic Usage

The respective folders contain implementations of the four methods.

#### Training
Train a model * using,
   ```
   python train_*.py
   ```
modify method parameters and problem setup in the .py file. The trained model is saved in the folder */saved_models

#### Evaluation

Evaluate a model * using,
   ```
   python test_*.py
   ```
modify method parameters, problem setup and evaluation in the .py file. The chosen metrics are saved in the results folder. 

#### Problem Specification

Problems, e.g., OU, Bistable, Lorenz-96 are specified in the problems folder. Each problem is a child of the parent class in problems/problem.py. New problems should implement the same methods and overwrite empty ones. 

#### Benchmarking

Implementations of the Ensemble Kalman Filter (EnKF), bootstrap Particle Filter (PF), Extended Kalman Filter (EKF) and standard Kalman Filter (KF) can be found in the benchmark_filters folder. See reference usage of these in the test_*.py files. 

## Used Libraries:

python 3.12.10

numpy 2.1.2

scipy 1.15.3

torch 2.7.1+cu118

pandas 2.2.3

## Acknowledgements

If you find this repository useful, please cite our article: <br>

> FILL IN



