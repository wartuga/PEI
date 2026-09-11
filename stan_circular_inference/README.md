# Stan Circular Inference
## Author: Diogo Almeida - fc64854@alunos.ciencias.ulisboa.pt

## Introduction

This work is motivated by the gap between circular statistical theory and its practical application in the context of probabilistic programming. The project focuses on the specification and estimation of circular models using Bayesian inference, and on building a modular and intuitive Python library. 

The library will provide algorithms to describe various models based on circular distributions (including the following distributions: Von Mises, Cardioid, Wrapped Cauchy), and will use an existing probabilistic programming tool from the literature, Stan. This library is a public service, organized, tested, validated, and well-documented, making these models accessible to a wider audience.

The project is essentially divided into two modules: the Inference Pipeline and the Probabilistic Models.

## Step by Step Setup

To obtain a stable environment for using the tool, the user is recommended to follow the steps below.

On Windows environments, execute the following commands:
```bash
wsl --install (requires to restart)
wsl -d Ubuntu
```

To obtain a stable version of Anaconda:
```bash
sudo wget https://repo.continuum.io/archive/Anaconda3-2022.10-Linux-x86_64.sh
bash Anaconda3-2022.10-Linux-x86_64.sh (requires to restart)
# Do you wish the installer to initialise Anaconda3? yes
```

Close the command line and open it again.

On Windows systems, to enter the Linux subsystem:
```bash
wsl -d Ubuntu
```

To install the Ubuntu dependencies:
```bash
sudo apt update
sudo apt install build-essential
sudo apt-get install manpages-dev
```

To create an environment in Anaconda:
```bash
conda create -n environment_name python=3.10
```

To activate that environment:
```bash
conda activate environment_name
```

To install the Anaconda dependencies:
```bash
conda install -c conda-forge gcc libstdcxx-ng
```

Finally, to install the tool's dependency:
```bash
pip install stan-circular-inference
```

Latest version in: https://pypi.org/project/stan-circular-inference/0.1.2/

## Inference Pipeline

The specifics of the `service` module are documented separately in [service](src/stan_circular_inference/service/README.md).

## Probabilistic Models

A comprehensive description of the `xpto` module is provided in [xpto](path/to/module).