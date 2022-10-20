import user_configs

NEWT_DATASET_DIR = user_configs.NEWT_DATASET_DIR
FG_DATASETS = user_configs.FG_DATASETS
PYTORCH_PRETRAINED_MODELS_DIR = user_configs.PYTORCH_PRETRAINED_MODELS_DIR
TENSORFLOW_PRETRAINED_MODELS_DIR = user_configs.TENSORFLOW_PRETRAINED_MODELS_DIR

# Libraries
TENSORFLOW = "tensorflow"
PYTORCH = "pytorch"

# Pretraining Datasets
IMAGENET = "ImageNet"
INAT2021 = "iNat2021"
INAT2021_MINI = "iNat2021 Mini"
INAT2018 = "iNat2018"

# Training Objectives
SUPERVISED = "Supervised"
MOCO_V2 = "MOCO v2"
SWAV = "SwAV"
SIMCLR = "SimCLR"
SIMCLR_V2 = "SimCLR v2"

# Models
RESNET50 = "ResNet50"
RESNET50_X4 = "ResNet50 x4"

model_specs = [
    {
        "name" : "random",
        "display_name" : "Random Init",
        "color" : "fuchsia",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : None,
        "training_dataset" : None,
        "train_objective" : "random",
        "pretrained_weights" : None
    },
    {
        "name" : "imagenet_supervised",
        "display_name" : "ImageNet Supervised (pytorch)",
        "color" : "black",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : None,
        "training_dataset" : IMAGENET,
        "train_objective" : SUPERVISED,
        "pretrained_weights" : None
    },
    {
        "name" : "imagenet_swav",
        "display_name" : "ImageNet SwAV",
        "color" : "C3",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'imagenet_swav_800ep_pretrain.pth.tar',
        "training_dataset" : IMAGENET,
        "train_objective" : SWAV,
        "pretrained_weights" : None
    },
    {
        "name" : "imagenet_moco_v2",
        "display_name" : "ImageNet MoCo v2",
        "color" : "C4",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'imagenet_moco_v2_800ep_pretrain.pth.tar',
        "training_dataset" : IMAGENET,
        "train_objective" : MOCO_V2,
        "pretrained_weights" : None
    },
    {
        "name" : "inat2021_supervised",
        "display_name" : "iNat2021 Supervised",
        "color" : "C9",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_supervised_large.pth.tar',
        "training_dataset" : INAT2021,
        "train_objective" : SUPERVISED,
        "pretrained_weights" : IMAGENET
    },
    {
        "name" : "inat2021_supervised_from_scratch",
        "display_name" : "iNat2021 Supervised",
        "color" : "purple",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_supervised_large_from_scratch.pth.tar',
        "training_dataset" : INAT2021,
        "train_objective" : SUPERVISED,
        "pretrained_weights" : None
    },
    {
        "name" : "inat2021_mini_supervised",
        "display_name" : "iNat2021 Mini Supervised",
        "color" : "C6",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_supervised_mini.pth.tar',
        "training_dataset" : INAT2021_MINI,
        "train_objective" : SUPERVISED,
        "pretrained_weights" : IMAGENET
    },
    {
        "name" : "inat2021_mini_supervised_from_scratch",
        "display_name" : "iNat2021 Mini Supervised",
        "color" : "pink",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_supervised_mini_from_scratch.pth.tar',
        "training_dataset" : INAT2021_MINI,
        "train_objective" : SUPERVISED,
        "pretrained_weights" : None
    },
    {
        "name" : "inat2021_mini_swav",
        "display_name" : "iNat2021 Mini SwAV",
        "color" : "tab:red",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_mini_swav_ckp-199.pth',
        "training_dataset" : INAT2021_MINI,
        "train_objective" : SWAV,
        "pretrained_weights" : None
    },
    {
        "name" : "inat2021_mini_swav_1k",
        "display_name" : "iNat2021 Mini SwAV 1k",
        "color" : "orangered",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_swav_mini_1000_ep.pth',
        "training_dataset" : INAT2021_MINI,
        "train_objective" : SWAV,
        "pretrained_weights" : None
    },
    {
        "name" : "inat2021_mini_moco_v2",
        "display_name" : "iNat2021 Mini MoCo v2",
        "color" : "tab:purple",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2021_moco_v2_mini_1000_ep.pth.tar',
        "training_dataset" : INAT2021_MINI,
        "train_objective" : MOCO_V2,
        "pretrained_weights" : None
    },
    {
        "name" : "inat2018_supervised",
        "display_name" : "iNat2018 Supervised",
        "color" : "C7",
        "format" : PYTORCH,
        "backbone" : RESNET50,
        "weights" : PYTORCH_PRETRAINED_MODELS_DIR + 'inat2018_supervised.pth.tar',
        "training_dataset" : INAT2018,
        "train_objective" : SUPERVISED,
        "pretrained_weights" : IMAGENET
    },
]