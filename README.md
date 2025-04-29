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
You can get the data from: 
```
wget zenodo-link
```
note: there is one file containing ~20k showers 

You can get the data needed to train the #PointsFM by running:
```
python pion-clouds/scripts/SF_arrays.py
```

## Training
You can run the training of #PointsFM:
```
python pion-clouds/scripts/shower_flow_train.py
```

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
