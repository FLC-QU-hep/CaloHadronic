# CaloHadronic: a diffusion model for the generation of hadronic pion showers

<div style="text-align: center;">
Thorsten Buss, Frank Gaede, Gregor Kasieczka, Anatolii Korol, Katja Kruger, Peter McKeown and Martina Mozzanica 
contact: martina.mozzanica@uni-hamburg.de
[![arXiv](https://img.shields.io/badge/arXiv-2501.05534-red)](https://arxiv.org/abs/2501.05534)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2-orange)](https://pytorch.org)

</div>

This repository contains the code for the results presented in the paper [`CaloHadronic: a diffusion model for the generation of hadronic pion showers`](link)

<img src=model-1.png width=900 style="border-radius:10px">

**Abstract:**

```
Simulating showers of particles in highly-granular calorimeters is a key frontier in the application of
machine learning to particle physics. Achieving high accuracy and speed with generative machine learning
models can enable them to augment traditional simulations and alleviate a major computing constraint. 
Recent developments have shown how diffusion based generative shower simulation approaches that do not
rely on a fixed structure, but instead generate geometry-independent point clouds,care very efficient.
We present a transformer-based extension to previous architectures which were developed for simulating
electromagnetic showers in the highly granular electromagnetic calorimeter of ILD. 
The attention mechanism now allows us to generate complex hadronic showers with more pronounced
substructure across both the electromagnetic and hadronic calorimeters. This is the first time that ML methods
are used to holistically generate showers across ECal and HCal in highly granular imaging calorimeters.
```

## Table of Contents

- [Setup](#setup)
- [Dataset](#dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Plots](#plots)


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
### Top contributors:

<a href="https://github.com/FLC-QU-hep/CaloHadronic/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=othneildrew/Best-README-Template" alt="contrib.rocks image" />
</a>

<p align="right">(<a href="#readme-top">back to top</a>)</p>


## Contact

Martina Mozzanica - [@linkedin](https://www.linkedin.com/in/martina-mozzanica-20017b202/) - martina.mozzanica@uni-hamburg.de

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- LICENSE -->
## License

Distributed under the Unlicense License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

