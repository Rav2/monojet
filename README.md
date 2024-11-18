# Machine Learning Electroweakino Production

Rafał Masełek, Mihoko M. Nojiri, Kazuki Sakurai

[![arXiv](https://img.shields.io/badge/arXiv-1234.56789-b31b1b.svg)](https://arxiv.org/abs/2411.00093)



This repository contains the code for the results presented in the paper: 
[Machine Learning Electroweakino Production](https://arxiv.org/pdf/2411.00093)

**Abstract:**

The system of light electroweakinos and heavy squarks gives rise to one of the most
challenging signatures to detect at the LHC. It consists of missing transverse energy recoiled against a few hadronic jets originating either from QCD radiation or squark decays.
The analysis generally suffers from the large irreducible Z + jets (Z → νν¯) background.
In this study, we explore Machine Learning (ML) methods for efficient signal/background
discrimination. Our best attempt uses both reconstructed (jets, missing transverse energy,
etc.) and low-level (particle-flow) objects. We find that the discrimination performance
improves as the pT threshold for soft particles is lowered from 10 GeV to 1 GeV, at the
expense of larger systematic uncertainty. In many cases, the ML method provides a factor
two enhancement in S/√(S + B) from a simple kinematical selection. The sensitivity on
the squark-elecroweakino mass plane is derived with this method, assuming the Run-3
and HL-LHC luminosities. Moreover, we investigate the relations between input features
and the network’s classification performance to reveal the physical information used in
the background/signal discrimination process.

## Dataset

Monte Carlo data is available in the CNRS Research Data repository under this [link]().

## Installation

The code requires python3. The recommended version is python 3.8.10. You can create a conda environment with:
`conda env create --name ml-ewkprod python==3.8.10`

Once you have created the new environment, activate it with:
`conda activate ml-ewkprod`

If you use Mac with M1 processor or newer, you need to install tensorflow-macos and tensorflow-metal packages:
`pip install tensorflow-macos==2.7 tensorflow-metal==0.3`

If you use different system, proceed to the next step. Install the requirement packages with:
`pip install -r requirements.txt`

## Usage

Example code showing the training of a GNN model is in the `train_ensemble.ipynb` file. How to load a trained model and use it to make predictions is described in `evaluate.ipynb`.

## Citation

If you use this code, please cite our paper:

`@article{Maselek:2024qyp,
    author = "Mase\l{}ek, Rafa\l{} and Nojiri, Mihoko M. and Sakurai, Kazuki",
    title = "{Machine Learning Electroweakino Production}",
    eprint = "2411.00093",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "10",
    year = "2024"
}
`
