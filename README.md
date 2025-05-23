# CaloHadronic: a diffusion model for the generation of hadronic pion showers

<div style="text-align: center;">
Thorsten Buss, Frank Gaede, Gregor Kasieczka, Anatolii Korol, Katja Kruger, Peter McKeown and Martina Mozzanica 

[![arXiv](https://img.shields.io/badge/arXiv-2501.05534-red)](https://arxiv.org/abs/2501.05534)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)

</div>

This repository contains the code for the results presented in the paper [`CaloHadronic: a diffusion model for the generation of hadronic pion showers`](link)
<img src=model.pdf width=900 style="border-radius:10px">

**Abstract:**

```
We show the first use of generative transformers for generating calorimeter showers as point clouds
in a high-granularity calorimeter. Using the tokenizer and generative part of the OmniJet-α model,
we represent the hits in the detector as sequences of integers. This model allows variable-length
sequences, which means that it supports realistic shower development and does not need to be
conditioned on the number of hits. Since the tokenization represents the showers as point clouds,
the model learns the geometry of the showers without being restricted to any particular voxel grid.
```

## Table of Contents

- [How to run the code](#how-to-run-the-code)
- [Dataset](#dataset)
- [Installation](#installation)
- [Tokenization](#tokenization)
- [Generative training](#generative-training)
- [Transfer learning / Classifier training](#transfer-learning--classifier-training)
- [Citation](#citation)


## Setup

Create the environment using:

```bash
# Using pip
pip install -r requirements.txt

# Or using conda
conda env create -f environment.yml
```

## Dataset 
You can get the data of CaloHadronic from: 
```
wget zenodo-link
```
note: there is one file containing ~20k showers 

You can get the raw root file with all Geant4 steps: 
```
wget zenodo-link
```

You can get the data needed to train the PointCountFM by running:
```
python pion-clouds/scripts/SF_arrays.py
```

## Training
You can run the training of PointCountFM:
```
python pion-clouds/scripts/shower_flow_train.py
```
Note: the architecture of PointCountFM is taken from ```https://github.com/FLC-QU-hep/PointCountFM```

You can run the training of ecal and hcal edm-diffusion:
```
python pion-clouds/scripts/training.py --type ecal 
python pion-clouds/scripts/training.py --type hcal
```

## Evaluation
You can run the evaluation of CaloHadronic: 
```
python pion-clouds/scripts/evaluation/eval_checkpoints.py
```

## Plots
You can run the plotting scripts of CaloHadronic: 
```
python pion-clouds/scripts/evaluation/occ-scale.py --load_dataset_and_project 1 --compute_features 1 --load_array_and_plot 1
python pion-clouds/scripts/evaluation/occ-scale_MultipleEnergies.py
```
