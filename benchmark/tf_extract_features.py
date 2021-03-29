import argparse
from functools import partial
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import tqdm

import configs
import dataset_utils
import tf_resnet_feature_extractor


def extract_fg_features(
    model_spec,
    datasets,
    feature_extractor_batch_size=32,
    device=None):

    feature_extractor = tf_resnet_feature_extractor.load_feature_extractor(model_spec, device)

    task_results = []

    pbar = tqdm.tqdm(datasets.items())
    for dataset_name, dataset_dir in pbar:

        pbar.set_description("Processing %s" % dataset_name)

        train_paths, train_classes, test_paths, test_classes =  dataset_utils.load_dataset(dataset_name, dataset_dir)

        X_train = feature_extractor.extract_features_batch(train_paths, batch_size=feature_extractor_batch_size, use_pbar=True)
        assert X_train.shape[0] == len(train_paths), "Feature extractor did not extract features for all train images?"
        y_train = np.array(train_classes)
        assert X_train.shape[0] == y_train.shape[0], "Mismatch between the number of train examples and the number of train labels"

        X_test = feature_extractor.extract_features_batch(test_paths, batch_size=feature_extractor_batch_size, use_pbar=True)
        assert X_test.shape[0] == len(test_paths), "Feature extractor did not extract features for all test images?"
        y_test = np.array(test_classes)
        assert X_test.shape[0] == y_test.shape[0], "Mismatch between the number of test examples and the number of test labels"


        results = {}
        results['name'] = dataset_name
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train
        results['y_test'] = y_test
        results['train_paths'] = np.array(train_paths, dtype=object)
        results['test_paths'] = np.array(test_paths, dtype=object)

        task_results.append(results)
    pbar.close()

    return task_results



def extract_newt_features(
    model_spec,
    newt_dataset_dir,
    feature_extractor_batch_size=32,
    device=None):

    feature_extractor = tf_resnet_feature_extractor.load_feature_extractor(model_spec, device)

    newt_task_dirs = glob.glob(os.path.join(newt_dataset_dir, "*"))

    task_results = []

    pbar = tqdm.tqdm(newt_task_dirs)
    for newt_task_dir in pbar:

        task_name = os.path.basename(newt_task_dir)
        pbar.set_description("Processing %s" % task_name)

        train_paths, train_classes, test_paths, test_classes = dataset_utils.load_newt_task(newt_task_dir)

        X_train = feature_extractor.extract_features_batch(train_paths, batch_size=feature_extractor_batch_size, use_pbar=True)
        assert X_train.shape[0] == len(train_paths), "Feature extractor did not extract features for all train images?"
        y_train = np.array(train_classes)
        assert X_train.shape[0] == y_train.shape[0], "Mismatch between the number of train examples and the number of train labels"

        X_test = feature_extractor.extract_features_batch(test_paths, batch_size=feature_extractor_batch_size, use_pbar=True)
        assert X_test.shape[0] == len(test_paths), "Feature extractor did not extract features for all test images?"
        y_test = np.array(test_classes)
        assert X_test.shape[0] == y_test.shape[0], "Mismatch between the number of test examples and the number of test labels"


        results = {}

        results['name'] = task_name
        results['X_train'] = X_train
        results['X_test'] = X_test
        results['y_train'] = y_train
        results['y_test'] = y_test
        results['train_paths'] = np.array(train_paths, dtype=object)
        results['test_paths'] = np.array(test_paths, dtype=object)

        task_results.append(results)
    pbar.close()

    return task_results



def run_tf_feature_extractor(newt_dataset_dir, fg_datasets, newt_features_dir, fg_features_dir, feature_extractor_batch_size=64, overwrite=False, x4_batch_size=16):
    """ Run the experiments from the paper for the tensorflow models.
    """

    for model_spec in configs.model_specs:

        if model_spec['format'] == configs.TENSORFLOW:

            print("Current feature extractor: %s" % model_spec['display_name'])
            st = time.time()

            # The resnet50x4 models often don't fit in a "standard" gpu
            if model_spec['backbone'] == configs.RESNET50_X4:
                device = '/device:CPU:0'
                batch_size=x4_batch_size
            else:
                device = None
                batch_size=feature_extractor_batch_size

            if newt_dataset_dir is not None:
                print("Extracting features across NeWT tasks")

                newt_features_fp = os.path.join(newt_features_dir, "%s.pkl" % model_spec['name'])
                if not os.path.exists(newt_features_fp) or overwrite:

                    newt_features  = extract_newt_features(
                        model_spec,
                        newt_dataset_dir,
                        feature_extractor_batch_size=batch_size,
                        device=device
                    )
                    newt_features_df = pd.DataFrame(newt_features)
                    newt_features_df['model_name'] = model_spec['name']
                    newt_features_df.to_pickle(newt_features_fp)
                else:
                    print("Found existing features for NeWT tasks")
            else:
                print("Skipping NeWT tasks")


            if fg_datasets is not None:
                print("Extracting features across FG datasets")
                fg_features_fp = os.path.join(fg_features_dir, "%s.pkl" % model_spec['name'])
                if not os.path.exists(fg_features_fp) or overwrite:

                    fg_features = extract_fg_features(
                        model_spec,
                        fg_datasets,
                        feature_extractor_batch_size=batch_size,
                        device=device
                    )
                    fg_features_df = pd.DataFrame(fg_features)
                    fg_features_df['model_name'] = model_spec['name']
                    fg_features_df.to_pickle(fg_features_fp)
                else:
                    print("Found existing features for FG Datasets")
            else:
                print("Skipping FG datasets")

            # Print the total time the experiment took
            et = time.time()
            total_time = et - st
            hours, remainder = divmod(total_time, 3600)
            minutes, seconds = divmod(remainder, 60)
            print('Feature Extraction Time: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
            print()



def parse_args():

    parser = argparse.ArgumentParser(description='Extract features using tensorflow models.')

    parser.add_argument('--newt_feature_dir', dest='newt_feature_dir',
                        help='Path to the directory to store the newt features.', type=str,
                        required=True)

    parser.add_argument('--fg_feature_dir', dest='fg_feature_dir',
                        help='Path to the directory to store FG dataset features.', type=str,
                        required=True)

    parser.add_argument('--batch_size', dest='batch_size',
                        help='Feature extractor batch size.', type=int,
                        required=False, default=64)

    parser.add_argument('--overwrite', dest='overwrite',
                        help='Overwrite existing saved features.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--x4_batch_size', dest='x4_batch_size',
                        help='Feature extractor batch size for the ResNet x4 models.', type=int,
                        required=False, default=16)

    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == '__main__':

    args = parse_args()

    assert args.newt_feature_dir != args.fg_feature_dir, "The NeWT and FG feature directories need to be distinct (the feature filenames will clash...)"

    if not os.path.exists(args.newt_feature_dir):
        print("Creating %s to store NeWT features" % args.newt_feature_dir)
        os.makedirs(args.newt_feature_dir)

    if not os.path.exists(args.fg_feature_dir):
        print("Creating %s to store FG Dataset features" % args.fg_feature_dir)
        os.makedirs(args.fg_feature_dir)

    run_tf_feature_extractor(
        newt_dataset_dir=configs.NEWT_DATASET_DIR,
        fg_datasets=configs.FG_DATASETS,
        newt_features_dir=args.newt_feature_dir,
        fg_features_dir=args.fg_feature_dir,
        feature_extractor_batch_size=args.batch_size,
        overwrite=args.overwrite,
        x4_batch_size=args.x4_batch_size
    )

