import json
import os

import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_newt_task(task_dir):
    """ Load a NeWT binary task
    """

    with open(os.path.join(task_dir, "train.json")) as f:
        train_dataset = json.load(f)
    with open(os.path.join(task_dir, "test.json")) as f:
        test_dataset = json.load(f)

    image_id_to_fp = {image['id'] : image['filename'] for image in train_dataset['images'] + test_dataset['images']}
    assert len(image_id_to_fp) == len(train_dataset['images']) + len(test_dataset['images']), "overlapping images in %s ?" % task_dir

    train_image_fps = []
    train_labels = []
    for anno in train_dataset['annotations']:

        image_fp = os.path.join(task_dir, image_id_to_fp[anno['image_id']])
        image_label = anno['category_id']
        assert image_label in [0, 1], "unexpected category id, assumed binary?"

        train_image_fps.append(image_fp)
        train_labels.append(image_label)

    test_image_fps = []
    test_labels = []
    for anno in test_dataset['annotations']:

        image_fp = os.path.join(task_dir, image_id_to_fp[anno['image_id']])
        image_label = anno['category_id']
        assert image_label in [0, 1], "unexpected category id, assumed binary?"

        test_image_fps.append(image_fp)
        test_labels.append(image_label)

    return train_image_fps, train_labels, test_image_fps, test_labels

def load_cub(dataset_path, label_file_name='image_class_labels.txt'):
    """ Load the CUB 200 dataset
    """

    # load data
    data = pd.read_csv(os.path.join(dataset_path, label_file_name), sep=' ', names=['id', 'class_label'])
    ids = data['id'].values
    labels = data.set_index('id').loc[ids].reset_index(inplace=False)['class_label'].values.astype(np.int)
    _, labels = np.unique(labels, return_inverse=True)

    files = pd.read_csv(os.path.join(dataset_path, 'images.txt'), sep=' ', names=['id', 'file'])
    files = files.set_index('id').loc[ids].reset_index(inplace=False)['file'].values
    files = [os.path.join(dataset_path, 'images', ff) for ff in files]

    is_train = pd.read_csv(os.path.join(dataset_path, 'train_test_split.txt'), sep=' ', names=['id', 'is_train'])
    is_train = is_train.set_index('id').loc[ids].reset_index(inplace=False)['is_train'].values.astype(np.int)

    train_paths = []
    train_classes = []
    test_paths = []
    test_classes = []

    for ii in range(len(files)):
        if is_train[ii] == 1:
            train_paths.append(files[ii])
            train_classes.append(labels[ii])
        else:
            test_paths.append(files[ii])
            test_classes.append(labels[ii])

    return train_paths, train_classes, test_paths, test_classes


def load_oxford_flowers(dataset_path):

    classes = loadmat(os.path.join(dataset_path, 'imagelabels.mat'))['labels'][0, :]-1
    train_ids = loadmat(os.path.join(dataset_path, 'setid.mat'))['trnid'][0, :]
    test_ids = loadmat(os.path.join(dataset_path, 'setid.mat'))['tstid'][0, :]
    train_paths = ['image_' + str(jj).zfill(5) + '.jpg' for jj in train_ids]
    test_paths = ['image_' + str(jj).zfill(5) + '.jpg' for jj in test_ids]
    train_paths = [os.path.join(dataset_path, 'jpg', jj)  for jj in train_paths]
    test_paths = [os.path.join(dataset_path, 'jpg', jj)  for jj in test_paths]
    train_classes = classes[train_ids-1].tolist()
    test_classes = classes[test_ids-1].tolist()

    return train_paths, train_classes, test_paths, test_classes


def load_stanford_dogs(dataset_path):

    train_paths = [jj[0][0] for jj in loadmat(os.path.join(dataset_path, 'train_list.mat'))['file_list']]
    test_paths = [jj[0][0] for jj in loadmat(os.path.join(dataset_path, 'test_list.mat'))['file_list']]
    train_paths = [os.path.join(dataset_path, 'Images', jj)  for jj in train_paths]
    test_paths = [os.path.join(dataset_path, 'Images', jj)  for jj in test_paths]
    train_classes = (loadmat(os.path.join(dataset_path, 'train_list.mat'))['labels'][:, 0]-1).tolist()
    test_classes = (loadmat(os.path.join(dataset_path, 'test_list.mat'))['labels'][:, 0]-1).tolist()

    return train_paths, train_classes, test_paths, test_classes

def load_stanford_cars(dataset_path):

    anns = loadmat(os.path.join(dataset_path, 'cars_annos.mat'))['annotations'][0]
    im_paths = [str(aa[0][0]) for aa in anns]
    im_paths = [os.path.join(dataset_path, aa) for aa in im_paths]
    classes = [int(aa[5][0][0])-1 for aa in anns]
    is_test = [int(aa[6][0][0]) for aa in anns]

    train_paths = []
    train_classes = []
    test_paths = []
    test_classes = []

    for ii in range(len(im_paths)):
        if is_test[ii] == 1:
            test_paths.append(im_paths[ii])
            test_classes.append(classes[ii])
        else:
            train_paths.append(im_paths[ii])
            train_classes.append(classes[ii])

    return train_paths, train_classes, test_paths, test_classes


def load_dataset(dataset_name, dataset_path):

    if dataset_name == 'CUB':
        return load_cub(dataset_path)
    elif dataset_name == 'CUBExpert':
        return load_cub(dataset_path, label_file_name='image_class_labels_expert.txt')
    elif dataset_name == 'NABirds':
        return load_cub(dataset_path)
    elif dataset_name == 'OxfordFlowers':
        return load_oxford_flowers(dataset_path)
    elif dataset_name == 'StanfordDogs':
        return load_stanford_dogs(dataset_path)
    elif dataset_name == 'StanfordCars':
        return load_stanford_cars(dataset_path)
    else:
        raise ValueError("Unknown dataset name: %s" % dataset_name)
