bayesiantesting
==============================
[//]: # (Badges)
[![Test Status](https://github.com/SimonBoothroyd/bayesiantesting/workflows/tests/badge.svg)](https://github.com/SimonBoothroyd/bayesiantesting/actions)
[![codecov](https://codecov.io/gh/SimonBoothroyd/bayesiantesting/branch/master/graph/badge.svg)](https://codecov.io/gh/SimonBoothroyd/bayesiantesting/branch/master)

A space for testing out projects based on Bayesian methods.

## Bayesian sampling of 2CLJQ models paper

For the code/files used in the Bayesian sampling of 2CLJQ models paper, navigate to `studies/combined/`

In order to reproduce the study from the paper, follow the following steps:

1. Clone the repository with `git clone https://github.com/SimonBoothroyd/bayesiantesting.git`
2. Switch to the `combined_run` branch using `git checkout -b combined_run`
3. Navigate to the top-level directory and create a conda environment for the project
   using `conda env create -f environment.yml`
4. Navigate to the `studies/combined/` directory, and do `. run_all_All` to run all of the 3-criteria calculations (
   rho_l, P_sat, Surface Tension) or `. run_all_rhol+Psat` to run all of the 2-criteria calculations. NOTE: These will
   run several MCMC simulations each for 6 or 7 compounds. This process will take a while.


### Copyright

Copyright (c) 2019, Simon Boothroyd


#### Acknowledgements

This repository is heavily based on Owen Madins [RJMC_LJ_Ethane](https://github.com/ocmadin/RJMC_LJ_Ethane) repository.

Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.1.
