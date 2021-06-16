# Indeitifiability Scores for Differentially Private Deep Learning
[![REUSE status](https://api.reuse.software/badge/github.com/SAP-samples/security-research-identifiability-in-dpdl)](https://api.reuse.software/info/github.com/SAP-samples/security-research-identifiability-in-dpdl)

## Description
SAP Security Research sample code to reproduce the research done in our paper "Quantifying identifiability to choose and audit \eps in differentially private deep learning"[1].

## Requirements
- [Python](https://www.python.org/) 3.6
- [h5py](https://www.h5py.org/)
- [numpy](https://numpy.org/)
- [scikit-learn](https://scikit-learn.org/)
- [scipy](https://scipy.org/)
- [Tensorflow](https://github.com/tensorflow)
- [Tensorflow Privacy](https://github.com/tensorflow/privacy)
- [matplotlib](https://matplotlib.org/)
- [pytest](https://pytest.org) 

## Download and Installation
### DPAttack Framework

Implementation of a Differential Privacy adversary to quantify identifiability in neural networks.

### Install

Running `make install` in the DPAttack folder should be enough for most usecases.

It creates the basic project directory structure and installs dpa as well as other requirements.
You can use pip as your package manager and install the `dpa` package via `python -m pip install -e ./`
For other package managers you need to install dpa using `setup.py`.

### Directory Structure

After having run `make install`, the following directory structure should be created in your local 
file system. Note: Everything that must not be tracked by git is already in `.gitignore`.

```
DPAttack/
     |-- Makefile
     |-- setup.py
     |-- requirements.txt
     |-- data/                # data files
     |-- experiments/         # experiment results
     |-- logs/		          # log files
     |-- notebooks/           # evaluation notebooks
     |-- dpa/			     # source root
          |--projects/	     # project implementations using dpa

```

For every dpa-project, a subdirectory should be created in `./DPAttack/dpa/projects` and the function `nn_attack_graph` in `./DPAttack/dpa` can be extended to load other datasets. For the evaluated datasets, we provided scripts to generate datasets based on local sensitivity in `./DPAttack/dpa/projects/mnist/heuristics` and `./DPAttack/dpa/projects/purchases/heuristics`, for which the resulting datasets are stored in `./data`. Running `python ./run_experiment.py -it=1 -ep=30 -l2=3.0 -ls -dlt=0.001 -lr=0.01 -bnd` in either dataset folder will train a single model and run the attack, returning the beliefs and other values for the attack analysis. The number of iterations and epochs, the clipping norm, delta, learning rate, and scenario (local/global sensitivity and bounded/unbounded differential privacy) can be specified here.

### Using the Framework

A sample visualization of an dp attack is provided in `./DPAttack/notebooks/synthetic_data_evaluation.ipynb`. Alternatively, the visualizations from the paper can be found in `./DPAttack/notebooks/mnist_eval.ipynb` and `./DPAttack/notebooks/purchases_eval.ipynb`.

In the following, we explain how the different tables and diagrams of the paper can be reproduced. Running the `./DPAttack/dpa/projects/mnist/run_experiment.sh` and `./DPAttack/dpa/projects/purchases/run_experiment.sh` script will run the experiments 250 times, and the scripts parameters can be adapted to execute all possible scenarios and repeat the experiment a variable number of times. The results dictionary consist of beliefs, sensitivities, and model accuracies (train and test) and are stored in the `./experiments` folder. We chose to further subdivide the results folder based on the evaluated scenario. The results can then be retrieved and plotted in a notebook, as we have done in `./DPAttack/notebooks/mnist_eval.ipynb` and `./DPAttack/notebooks/purchases_eval.ipynb`.


### Contributors

 - Daniel Bernau (corresponding author)
 - Hannah Keller
 - Philip-William Grassal
 - Jonas Robl

## Citations
If you use this code in your research, please cite:

```
@article{BKG+21,
  author    = {Daniel Bernau and
  			   Hannah Keller and
               Philip{-}William Grassal and
               Guenther Eibl and
               Florian Kerschbaum},
  title     = {Quantifying identifiability to choose and audit epsilon in differentially private deep learning},
  journal   = {CoRR},
  volume    = {abs/2103.02913},
  year      = {2021},
  url       = {http://arxiv.org/abs/2103.02913},
  archivePrefix = {arXiv},
  eprint    = {2103.02913},
}
```

## References
[1] Daniel Bernau, Guenther Eibl, Philip Grassal, Hannah Keller, Florian Kerschbaum:
Quantifying identifiability to choose and audit epsilon in differentially private deep learning.
arXiv:2103.02913
https://arxiv.org/abs/2103.02913

## License
Copyright (c) 2021 SAP SE or an SAP affiliate company. All rights reserved. This project is licensed under the Apache Software License, version 2.0 except as noted otherwise in the [LICENSE](LICENSES/Apache-2.0.txt) file.
