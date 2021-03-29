# Benchmarking Representation Learning for Natural World Collections

## Dataset Preparation
You need to download the following datasets:

## Pretrained Model Preparation
You need to download the following pretrained models:

## Create user_configs.py
You need to create a user_configs.py in the `benchmark/` directory that specifies paths to the various dataset directoried and pretrained model directories:
```
################
# Adjust the following paths for your local setup

# Datasets
NEWT_DATASET_DIR = "/data/datasets/newt"
FG_DATASETS = {
    'CUB' : '/data/datasets/CUB_200_2011/CUB_200_2011/',
    'CUBExpert' : '/data/datasets/CUB_200_2011/CUB_200_2011/',
    'NABirds' : '/data/datasets/nabirds/',
    'OxfordFlowers' : '/data/datasets/oxford_flowers/',
    'StanfordDogs' : '/data/datasets/stanford_dogs/',
    'StanfordCars' : '/data/datasets/stanford_cars/',
}

# Weight Directories
PYTORCH_PRETRAINED_MODELS_DIR = '/data/models/cvpr21_pretrained_models/pt/'
TENSORFLOW_PRETRAINED_MODELS_DIR = '/data/models/cvpr21_pretrained_models/tf/'
```

## Reproduce the experiments in the paper

Extract features from the datasets using tensorflow models (estimated time ~10 hours):
```
$ CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 python tf_extract_features.py \
--newt_feature_dir newt_features \
--fg_feature_dir fg_features \
--batch_size 64 \
--x4_batch_size 16 
```


Extract features from the datasets using pytorch models (estimated time ~3 hours):
```
$ CUDA_VISIBLE_DEVICES=0 python pt_extract_features.py \
--newt_feature_dir newt_features \
--fg_feature_dir fg_features \
--batch_size 64
```


Evaluate linear models on the pre-extracted NeWT features (estimated time ~3 hours (02:48:31)):
```
$ python evaluate_linear_models.py \
--feature_dir newt_features \
--result_dir newt_results_linearsvc_1000_standardize_noramlize_grid_search \
--model linearsvc \
--max_iter 1000 \
--standardize \
--normalize \
--grid_search
```

Evaluate linear models on the pre-extracted FG Datasets features (estimated time ~10 hours (09:36:28)):
```
$ python evaluate_linear_models.py \
--feature_dir fg_features \
--result_dir fg_results_sgd_3000_standardize_noramlize_grid_search \
--model sgd \
--max_iter 3000 \
--standardize \
--normalize \
--grid_search
```


Create the FG datasets stem plot and latex table:
```
$ python make_fg_plots.py \
--result_dir fg_results_sgd_3000_standardize_noramlize_grid_search \
--output_dir figures_v2
```


Create the NeWT tasks stem plot and latex table:
```
$ python make_newt_plots.py \
--result_dir newt_results_linearsvc_1000_standardize_noramlize_grid_search \
--output_dir figures_v2
```