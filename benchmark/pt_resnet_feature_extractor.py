
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

def loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


class list_loader(Dataset):

    def __init__(self, im_paths, transform=None):
        # im_paths is a list
        self.im_paths = im_paths
        self.transform = transform

    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, index):
        path = self.im_paths[index]
        sample = loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, index



class PTResNet50FeatureExtractor():

    def __init__(self, model, device, im_size=256, im_size_crop=224, feature_shape=2048):

        self.model = model
        self.device = device
        self.feature_shape = feature_shape

        self.transform = transforms.Compose([
            transforms.Resize(im_size),
            transforms.CenterCrop(im_size_crop),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def extract_features_batch(self, image_fp_list, batch_size=32, num_workers=6):

        pin_mem = True
        if self.device == 'cpu':
            pin_mem = False

        loader = torch.utils.data.DataLoader(
            list_loader(image_fp_list, transform=self.transform),
            batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_mem)


        features = np.empty([len(image_fp_list), self.feature_shape])
        feature_index = 0
        ids = []
        with torch.no_grad():
            for ii, (images, indices) in enumerate(loader):
                images = images.to(self.device)

                op = self.model(images)
                op = op.squeeze(2).squeeze(2)
                feats = op.cpu().data.numpy()
                num_in_batch = feats.shape[0]

                features[feature_index:feature_index+num_in_batch] = feats
                ids.append(indices.cpu().data.numpy())
                feature_index += num_in_batch

        # double check that the image order is as expected i.e. sequential
        ids = np.hstack(ids)
        assert np.all(ids == np.arange(ids.shape[0])), "The image features were extracted out of order"

        return features


def load_feature_extractor(model_spec, device):
    """ The pytorch models have not been standardized, so we need to do some custom surgery for different checkpoint files.

    For ImageNet MOCOv2 you need to download the pretrained model from here:
    https://dl.fbaipublicfiles.com/moco/moco_checkpoints/moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar
    rename: imagenet_moco_v2_800ep_pretrain.pth.tar
    For ImageNet SWAV download:
    https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar
    rename: imagenet_swav_800ep_pretrain.pth.tar
    """

    model_type = model_spec['name']
    model_weights_fp = model_spec['weights']

    if model_type == 'imagenet_swav':
        # or could load from hub model
        # model = torch.hub.load('facebookresearch/swav', 'resnet50')

        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Identity()
        state_dict = torch.load(model_weights_fp,  map_location="cpu")

        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        for k in list(state_dict.keys()):
            if 'projection' in k or 'prototypes' in k:
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)

    elif model_type == 'imagenet_moco_v2':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Identity()
        checkpoint = torch.load(model_weights_fp, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)

    elif model_type == 'imagenet_supervised':
        model = models.resnet50(pretrained=True)

    elif model_type == 'random':
        model = models.resnet50(pretrained=False)

    elif model_type == 'inat2018_supervised':
        model = models.resnet50(pretrained=False)
        # This model was actually trained with 10000 classes for the fc layer
        # but only 8142 (the number in inat2018) were actually updated
        model.fc = torch.nn.Linear(model.fc.in_features, 10000)
        checkpoint = torch.load(model_weights_fp, map_location="cpu")
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)

    elif model_type == 'inat2021_mini_supervised':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 10000)
        checkpoint = torch.load(model_weights_fp, map_location="cpu")
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)

    elif model_type == 'inat2021_supervised':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 10000)
        checkpoint = torch.load(model_weights_fp, map_location="cpu")
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)

    elif model_type == 'inat2021_mini_supervised_from_scratch':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 10000)
        checkpoint = torch.load(model_weights_fp, map_location="cpu")
        state_dict = {k.replace("module.", ""): v for k, v in checkpoint['state_dict'].items()}
        msg = model.load_state_dict(state_dict, strict=True)

    elif model_type == 'inat2021_supervised_from_scratch':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 10000)
        checkpoint = torch.load(model_weights_fp, map_location="cpu")
        msg = model.load_state_dict(checkpoint['state_dict'], strict=True)

    elif model_type == 'inat2021_mini_moco_v2':
        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Identity()
        checkpoint = torch.load(model_weights_fp, map_location="cpu")

        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)

    elif model_type == 'inat2021_mini_swav' or model_type == 'inat2021_mini_swav_1k':
        # or could load from hub model
        # model = torch.hub.load('facebookresearch/swav', 'resnet50')

        model = models.resnet50(pretrained=False)
        model.fc = torch.nn.Identity()
        state_dict = torch.load(model_weights_fp,  map_location="cpu")

        state_dict = {k.replace("module.", ""): v for k, v in state_dict['state_dict'].items()}
        for k in list(state_dict.keys()):
            if 'projection' in k or 'prototypes' in k:
                del state_dict[k]

        msg = model.load_state_dict(state_dict, strict=True)

    else:
        raise ValueError("Unknown pytorch model: %s" % model_type)


    # remove the final fully connected layer so the model only operates with post average pool features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))
    model.to(device)
    model.eval()

    feature_extractor = PTResNet50FeatureExtractor(model, device)

    return feature_extractor