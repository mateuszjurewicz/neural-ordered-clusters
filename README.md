![Python](https://img.shields.io/badge/python-v3.6.5-green.svg)
![PyTorch](https://img.shields.io/badge/pytorch-v1.10.0-blue.svg)

# Neural Ordered Clusters

This code repository contains all code and links to datasets required for repeated experiments for a scientific publication introducing Neural Ordered Clusters (NOC), an adaptive, input-dependent clustering and cluster ordering method capable of making predictions for sets of varying cardinality. What follows is a description of the steps necessary to set up the environment and run experiments locally.

## Setup

The following section explains how to obtain the freely available datasets and prepare the virtual environment.

### Data
The datasets needed to be downloaded prior to running the experiments are available under the following links:

- [Gauss2D Ordered By Distance from Origin](https://github.com/anonymous-paper-submissions/neural-ordered-clusters/blob/main/run_gauss2D.py) is generated based on the default configuration provided in the linked file's call to `main()`.
- [Synthetic Catalogs](https://github.com/anonymous-paper-submissions/neural-ordered-clusters/blob/main/run_synthetic.py) are generated based on the default configuration file already provided within this repository [here](https://github.com/anonymous-paper-submissions/neural-ordered-clusters/blob/main/run_configs/synthetic_rulesets.json). 
- [PROCAT](https://doi.org/10.6084/m9.figshare.14709507) is also freely available and contains human-made product catalogs.

The PROCAT dataset requires additional preprocessing as provided in the `procat_preprocess_for_bert.py` file, which produces 3 csv files. The final directory tree should look like this:

```
|-- README.md
|-- danish-bert-botxo
|-- data
|   |-- PROCAT
|   |   |-- PROCAT.test.csv
|   |   |-- PROCAT.train.csv
|   |   |-- PROCAT.validation.csv
|   |   |-- procat_dataset_loader.py
|   |   `-- procat_preprocess_for_bert.py
├── data_gauss2D.py
├── data_procat.py
├── data_synthetic.py
├── figures
│   ├── Gauss2D
│   ├── PROCAT
│   └── SyntheticStructures
├── inspect_gauss2D.py
├── inspect_synthetic.py
├── logs
├── model_noc.py
├── requirements.txt
├── run_configs
│   └── synthetic_rulesets.json
├── run_gauss2D.py
├── run_procat.py
├── run_synthetic.py
├── saved_models
│   ├── Gauss2D
│   ├── PROCAT
│   └── SyntheticStructures
├── show.py
├── utils.py
└── visual
```

The PROCAT experiment employs a language specific version for the BERT model. The PROCAT model is available for download [here](https://github.com/certainlyio/nordic_bert),
and should be placed in its `danish-bert-botxo` directory.

### Environment

The code requires **python 3.6.5**. All requirements are listed in the `requirements.txt` and can be installed in the following way (on a Linux system):

```
python -m venv /path/to/new/virtual/environment
source <ven_pathv>/bin/activate
python -m pip install -r requirements.txt
```

If you wish to store & view repeated experimental results somewhere other than in the log files saved in the `/logs` directory, you will also need to set up a Mongo database (instructions [here](https://docs.mongodb.com/manual/installation/#std-label-tutorial-installation)) and have a local omniboard instance running (more information [here](https://github.com/vivekratnavel/omniboard)).

## Usage

After finishing the preceding steps, you should be able to run each experiment through its corresponding `.py` file. The majority of the model architecture is defined in the `NOC()` class from `model_noc.py`. For example:

```
# Run 2D Gauss Experiment, with the default model configuration:
python run_gauss2D.py
```
The synthetic catalogs and PROCAT experiments generate the preprocessed datasets using parameters specified in their corresponding scripts. When a training script is running, comprehensive progress logs will be visible in the console and saved to a log file in the `./logs` directory, e.g.:

```
2022-09-27 14:24:40,575 | INFO | EXPERIMENT run started, id: 1340156480
2022-09-27 14:24:40,686 | INFO | 1340156480 | Time started (UTC): 2022-09-27 14:24:40,686 
2022-09-27 14:24:40,689 | INFO | 1340156480 | Seeding for reproducibility with: 1234
2022-09-27 14:24:40,690 | INFO | 1340156480 | Arguments: (...)
2022-09-27 14:24:40,692 | INFO | 1340156480 | Device: gpu
2022-09-27 14:24:40,801 | INFO | 1340156480 | The model has 12,117,655 trainable parameters
2022-09-27 14:25:38,906 | INFO | 1340156480 | 1  N:83  K:5  Clustering Loss: 115.910  Ordering Loss: 6.862  Cardinality Loss: 5.815 Total Loss: 128.586  Mean Time/Iteration: 0.6
(...)
```
