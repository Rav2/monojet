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

Monte Carlo data is available in the CNRS Research Data repository under this [link](https://doi.org/10.57745/RVC6WQ).

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

## Folder structure

- **data:** this is a placeholder for data set which has to be downloaded from [link](https://doi.org/10.57745/RVC6WQ).
- **models:** this directory contains all trained NN models (60 in total).
- **preselection_cutflows:** this directory contains information about efficiencies of preselection cuts.
- **simulation:** this directory contains examples of MG5 cards and SLHA files.

## License
<p xmlns:cc="http://creativecommons.org/ns#" >This work is licensed under <a href="https://creativecommons.org/licenses/by/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""></a> Details are in the license.txt file.</p>



## Citation

If you use this code, please cite our paper:

```
@article{Maselek:2024qyp,
    author = "Mase\l{}ek, Rafa\l{} and Nojiri, Mihoko M. and Sakurai, Kazuki",
    title = "{Machine Learning Electroweakino Production}",
    eprint = "2411.00093",
    archivePrefix = "arXiv",
    primaryClass = "hep-ph",
    month = "10",
    year = "2024"
}
```

If you use the data set, please additionally cite:
```
@data{RVC6WQ_2024,
author = {MASELEK, Rafal},
publisher = {Recherche Data Gouv},
title = {{Simulated data for searches for electroweakino dark matter in the monojet channel}},
year = {2024},
version = {V1},
doi = {10.57745/RVC6WQ},
url = {https://doi.org/10.57745/RVC6WQ}
}
```
