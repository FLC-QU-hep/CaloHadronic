# CaloHadronic

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
