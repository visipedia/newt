import argparse
from contextlib import redirect_stdout
import os

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

import configs
import plot_utils

# Set matplotlib font size
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 18
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def create_fg_plots(fg_results_dir, output_dir):

    fg_results_df = pd.DataFrame(None, columns=['model_name', 'name', 'acc'])

    for model_spec in configs.model_specs:

        model_name = model_spec['name']
        results_fp = os.path.join(fg_results_dir, model_name + ".pkl")

        if not os.path.exists(results_fp):
            print("WARNING: did not find results for model %s at %s" % (model_name, results_fp))
            continue

        results_df = pd.read_pickle(results_fp)

        fg_results_df = pd.concat([fg_results_df, results_df], axis=0, ignore_index=True)

    # models are the rows
    fg_model_results_df = fg_results_df.pivot(index='model_name', columns='name', values='acc')
    fg_model_results_df = fg_model_results_df.loc[[model_spec['name'] for model_spec in configs.model_specs]]
    fg_model_results_df = fg_model_results_df[['OxfordFlowers', 'CUB', 'CUBExpert', 'NABirds', 'StandfordDogs', 'StandfordCars']]

    # datasets are the rows
    fg_dataset_results_df = fg_results_df.pivot(index='name', columns='model_name', values='acc')
    fg_dataset_results_df = fg_dataset_results_df.loc[['OxfordFlowers', 'CUB', 'CUBExpert', 'NABirds', 'StandfordDogs', 'StandfordCars']]
    fg_dataset_results_df = fg_dataset_results_df[[model_spec['name'] for model_spec in configs.model_specs]]

    #####################
    # Stem plot for FG tasks
    datasets = ['OxfordFlowers', 'CUB', 'CUBExpert', 'NABirds', 'StandfordDogs', 'StandfordCars']

    result_names=[
        'imagenet_simclr',
        'imagenet_simclr_x4',
        'imagenet_simclr_v2',
        'imagenet_swav',
        'imagenet_moco_v2',
        'inat2021_supervised',
        'inat2021_mini_supervised',
        'inat2018_supervised',
        'inat2021_simclr',
        'inat2021_mini_simclr',
        'inat2021_mini_simclr_x4',
        'inat2021_mini_simclr_v2',
        'inat2021_mini_swav',
        'inat2021_mini_moco_v2'
    ]

    baseline_scores = fg_model_results_df.loc['imagenet_supervised'][datasets].values
    exp_results = []
    for model_name in result_names:

        model_spec = next(model_spec for model_spec in configs.model_specs if model_spec['name'] == model_name)

        if model_spec['name'] == 'imagenet_supervised':
            continue

        task_scores = fg_model_results_df.loc[ model_spec['name']][datasets].values - baseline_scores

        r = {
            'name' : model_spec['name'],
            'scores' :  task_scores,
            'color' : model_spec['color'],
            'display_name' : model_spec['display_name'],
        }

        if model_spec['name'] == 'random':
            r['line_style'] = ':'
        elif model_spec['train_objective'] == configs.SUPERVISED:
            r['line_style'] = '-'
        else:
            r['line_style'] = '--'

        if model_spec['name'] == 'random':
            r['marker_format'] = '>'
        elif model_spec['training_dataset'] == configs.IMAGENET:
            r['marker_format'] = '^'
        elif model_spec['train_objective'] == configs.SUPERVISED:
            r['marker_format'] = 'o'
        elif model_spec['train_objective'] == configs.SIMCLR or model_spec['train_objective'] == configs.SIMCLR_V2:
            r['marker_format'] = '*'
        elif model_spec['train_objective'] == configs.MOCO_V2:
            r['marker_format'] = '*'
        elif model_spec['train_objective'] == configs.SWAV:
            r['marker_format'] = '*'
        else:
            raise ValueError("Unknown train objective: %s" % model_spec['train_objective'])

        exp_results.append(r)


    result_df = pd.DataFrame(exp_results)

    task_labels = ['Flowers102', 'CUB', 'CUBExpert', 'NABirds', 'StanfordDogs', 'StanfordCars']

    plot_utils.task_stem_plot(
        result_df,
        task_labels,
        task_space=4,
        task_offset=5,
        title='Change in Mean Accuracy from Imagenet Supervised Features',
        xlabel='',
        ylabel='$\Delta$ ACC',
        figsize=(15, 5),
        rotate_x_tick_labels=False,
        task_baseline_scores=baseline_scores,
        task_baseline_scores_x_offset=-.5,
        task_baseline_scores_y_pos=-.55
    )

    output_fp = os.path.join(output_dir, "fg_stem_plot.pdf")
    plt.savefig(output_fp, bbox_inches='tight')

    ##############
    # Latex Table of results

    result_names=[
        'imagenet_supervised',
        'imagenet_simclr',
        'imagenet_simclr_x4',
        'imagenet_simclr_v2',
        'imagenet_swav',
        'imagenet_moco_v2',
        'inat2021_supervised',
        'inat2021_supervised_from_scratch',
        'inat2021_mini_supervised',
        'inat2021_mini_supervised_from_scratch',
        'inat2018_supervised',
        'inat2021_simclr',
        'inat2021_mini_simclr',
        'inat2021_mini_simclr_x4',
        'inat2021_mini_simclr_v2',
        'inat2021_mini_swav',
        'inat2021_mini_moco_v2'
    ]


    num_cols = len(datasets)
    num_rows = len(result_names)

    table_fp = os.path.join(output_dir, "fg_latex_table.txt")

    with open(table_fp, "w") as f:
        with redirect_stdout(f):
            print("\\begin{table*}[t]")
            print("\\small")
            print("\\centering")
            print("\\begin{tabular}{|l | l | %s |}" % (" ".join(["c"] * (num_cols + 1))))
            print("\\hline")

            header = ["Source Dataset", "Train Loss"] + datasets + ["Mean ACC"]
            print("  &  \t".join(header) + "\\\\")
            print("\hline\hline")

            for model_name in result_names:

                model_spec = next(model_spec for model_spec in configs.model_specs if model_spec['name'] == model_name)

                model_scores = fg_model_results_df.loc[model_spec['name']]

                ys = []
                ry = []
                for i, label in enumerate(datasets):

                    v = model_scores[label]
                    ys.append(
                        "%0.3f" % v
                    )
                    ry.append(v)


                td = model_spec['training_dataset'] if model_spec['training_dataset'] is not None else ""
                to = model_spec['train_objective']
                if model_spec['backbone'] == configs.RESNET50_X4:
                    to += " x4"
                if model_spec['pretrained_weights'] is not None:
                    to += " (from %s)" % model_spec['pretrained_weights']

                row = [td, to] + ys + ["%0.3f" % np.mean(ry)]

                print("  &  \t".join(row) + " \\\\")

            print("\\hline")
            print("\\end{tabular}")
            print("\\caption{}")
            print("\\label{table:}")
            print("\\end{table*}")



def parse_args():

    parser = argparse.ArgumentParser(description='Create the stem plot figure and latex table of results for the FG datasets.')

    parser.add_argument('--result_dir', dest='result_dir',
                        help='Path to the directory containing the FG results.', type=str,
                        required=True)

    parser.add_argument('--output_dir', dest='output_dir',
                        help='Path to the directory to save figures and tables.', type=str,
                        required=True)

    parsed_args = parser.parse_args()

    return parsed_args

if __name__ == '__main__':


    args = parse_args()

    if not os.path.exists(args.result_dir):
        raise ValueError("FG results directory %s does not exist" % fg_result_dir)

    if not os.path.exists(args.output_dir):
        print("Creating %s to store plots and tables" % args.output_dir)
        os.makedirs(args.output_dir)

    create_fg_plots(args.result_dir, args.output_dir)