
import argparse
from functools import partial
import os
import time

import numpy as np
import pandas as pd
import tqdm

import configs
import linear_evaluation


def evalute_features(features_df, evaluator_fn):

    task_results = []

    pbar = tqdm.tqdm(features_df.iterrows())
    for i, row in pbar:

        dataset_name = row['name']
        pbar.set_description("Evaluating %s" % dataset_name)

        X_train = row['X_train']
        X_test = row['X_test']
        y_train = row['y_train']
        y_test = row['y_test']

        results = evaluator_fn(X_train, y_train, X_test, y_test)

        results['name'] = dataset_name

        task_results.append(results)
    pbar.close()

    return task_results


def analyze_features(feature_dir, results_dir, evaluator_fn, overwrite=False):

    for model_spec in configs.model_specs:

        model_name = model_spec['name']
        print("Evaluating features from %s" % model_name)
        st = time.time()

        results_fp = os.path.join(results_dir, model_name + ".pkl")

        # Have we already run evaluation?
        if os.path.exists(results_fp) and not overwrite:
            print("Found existing results file for model %s at %s" % (model_name, results_fp))
            continue

        # Make sure the features exists for this model
        feature_fp = os.path.join(feature_dir, model_name + ".pkl")
        if not os.path.exists(feature_fp):
            print("WARNING: did not find features for model %s at location %s" % (model_name, feature_fp))
            continue

        # Load in the features extracted by this model
        feature_df = pd.read_pickle(feature_fp)

        # Evaluate the features
        results = evalute_features(feature_df, evaluator_fn)

        # Save off the results
        results_df = pd.DataFrame(results)
        results_df['model_name'] = model_name
        results_df.to_pickle(results_fp)

        # Print the total time the experiment took
        et = time.time()
        total_time = et - st
        hours, remainder = divmod(total_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print('Evaluation Time: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
        print()




def parse_args():

    parser = argparse.ArgumentParser(description='Train and evaluate linear models.')

    parser.add_argument('--feature_dir', dest='feature_dir',
                        help='Path to the directory containing extracted features.', type=str,
                        required=True)

    parser.add_argument('--result_dir', dest='result_dir',
                        help='Path to the directory to store results.', type=str,
                        required=True)

    parser.add_argument('--model', dest='model',
                        help='Model type', type=str,
                        choices=['logreg', 'sgd', 'linearsvc'],
                        required=True)

    parser.add_argument('--overwrite', dest='overwrite',
                        help='Overwrite existing saved features.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--max_iter', dest='max_iter',
                        help='Maximum number of iterations taken for the solvers to converge.', type=int,
                        required=False, default=100)

    parser.add_argument('--standardize', dest='standardize',
                        help='Standardize features by removing the mean and scaling to unit variance.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--normalize', dest='normalize',
                        help='Scale feature vectors individually to unit norm.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--grid_search', dest='grid_search',
                        help='Search for optimal regularization terms.',
                        required=False, action='store_true', default=False)

    parser.add_argument('--dual', dest='dual',
                        help='Use the dual formulation of the SVM (only relevant for `model = linearsvc`)',
                        required=False, action='store_true', default=False)

    parsed_args = parser.parse_args()

    return parsed_args

if __name__ == '__main__':


    args = parse_args()


    if not os.path.exists(args.result_dir):
        print("Creating %s to store results" % args.result_dir)
        os.makedirs(args.result_dir)


    if args.model == 'logreg':
        evaluator_fn = partial(linear_evaluation.logreg, max_iter=args.max_iter, grid_search=args.grid_search, standardize=args.standardize, normalize=args.normalize)
    elif args.model == 'sgd':
        evaluator_fn = partial(linear_evaluation.sgd, max_iter=args.max_iter, grid_search=args.grid_search, standardize=args.standardize, normalize=args.normalize)
    elif args.model == 'linearsvc':
        evaluator_fn = partial(linear_evaluation.linearsvc, max_iter=args.max_iter, grid_search=args.grid_search, standardize=args.standardize, normalize=args.normalize, dual=args.dual)
    else:
        raise ValueError("Unknown model type")

    st = time.time()
    analyze_features(args.feature_dir, args.result_dir, evaluator_fn, args.overwrite)
    et = time.time()
    total_time = et - st
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print('Total Evaluation Time: {:02}:{:02}:{:02}'.format(int(hours), int(minutes), int(seconds)))
    print()