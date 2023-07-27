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


def create_newt_plots(newt_results_dir, output_dir):
    
    newt_results_df = pd.DataFrame(None, columns=['model_name', 'name', 'acc'])

    for model_spec in configs.model_specs:

        model_name = model_spec['name']
        results_fp = os.path.join(newt_results_dir, model_name + ".pkl")

        if not os.path.exists(results_fp):
            print("WARNING: did not find results for model %s at %s" % (model_name, results_fp))
            continue

        results_df = pd.read_pickle(results_fp)

        newt_results_df = pd.concat([newt_results_df, results_df], axis=0, ignore_index=True)

    # models are the rows
    newt_model_results_df = newt_results_df.pivot(index='model_name', columns='name', values='acc')
    newt_model_results_df = newt_model_results_df.loc[[model_spec['name'] for model_spec in configs.model_specs]]

    # datasets are the rows
    newt_dataset_results_df = newt_results_df.pivot(index='name', columns='model_name', values='acc')
    newt_dataset_results_df = newt_dataset_results_df[[model_spec['name'] for model_spec in configs.model_specs]]
    
    # Load in the cluster info
    # GVH: this needs to be factored out
    newt_clusters_df = pd.read_csv("newt_task_clusters.csv")
    
    # The order in which we want to render the task clusters
    task_order=[
        {
            "name" : "Appearance\nAge",
            "cluster" : "appearance",
            "sub_cluster" : "age"
        },
        {
            "name" : "Appearance\nAttribute",
            "cluster" : "appearance",
            "sub_cluster" : "attribute"
        },
        {
            "name" : "Appearance\nHealth",
            "cluster" : "appearance",
            "sub_cluster" : "health"
        },
        {
            "name" : "Appearance\nSpecies",
            "cluster" : "appearance",
            "sub_cluster" : "species"
        },
        {
            "name" : "Behavior",
            "cluster" : "behavior",
            "sub_cluster" : None
        },
        {
            "name" : "Context",
            "cluster" : "context",
            "sub_cluster" : None
        },
        {
            "name" : "Counting",
            "cluster" : "counting",
            "sub_cluster" : None
        },
        {
            "name" : "Gestalt",
            "cluster" : "gestalt",
            "sub_cluster" : None
        }
    ]

    # Go through and add task counts 
    for task_info in task_order:

        # Get the dataset names that belong to this cluster (and subcluster)
        if task_info['sub_cluster'] is not None:
            cluster_dataset_names = newt_clusters_df[(newt_clusters_df['cluster'] == task_info['cluster']) & (newt_clusters_df['sub_cluster'] == task_info['sub_cluster'])]['name']
        else:
            cluster_dataset_names = newt_clusters_df[newt_clusters_df['cluster'] == task_info['cluster']]['name']

        # Get the model results on these datasets
        cluster_dataset_results = newt_dataset_results_df[newt_dataset_results_df.index.isin(cluster_dataset_names)]

        num_datasets = cluster_dataset_results.shape[0] 
        task_info['num_datasets'] = num_datasets
    
    
    
    #####################
    # Stem plot for NeWT tasks 
    # Organize the method results by average task cluster performance
    method_results = []
    for model_spec in configs.model_specs:

        task_scores = []

        for task_info in task_order:

            # Get the dataset names that belong to this cluster (and subcluster)
            if task_info['sub_cluster'] is not None:
                cluster_dataset_names = newt_clusters_df[(newt_clusters_df['cluster'] == task_info['cluster']) & (newt_clusters_df['sub_cluster'] == task_info['sub_cluster'])]['name']
            else:
                cluster_dataset_names = newt_clusters_df[newt_clusters_df['cluster'] == task_info['cluster']]['name']

            # Get the model results on these datasets
            cluster_dataset_results = newt_dataset_results_df[newt_dataset_results_df.index.isin(cluster_dataset_names)]


            # Get the mean result for this model on these datasets
            method_cluster_results = cluster_dataset_results[model_spec['name']].mean()

            assert not pd.isna(method_cluster_results)

            task_scores.append(method_cluster_results)

        task_scores = np.array(task_scores)

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

        method_results.append(r)

    method_results_df = pd.DataFrame(method_results).set_index('name', drop=True)
    
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


    baseline_scores = method_results_df.loc['imagenet_supervised']['scores']
    exp_results = []
    for method_name in result_names:

        r = method_results_df.loc[method_name].copy()
        r['scores'] = r['scores'] - baseline_scores
        exp_results.append(r)

    result_df = pd.DataFrame(exp_results)

    task_labels = ["%s\n%d" % (task['name'], task['num_datasets']) for task in task_order]

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
        task_baseline_scores_y_pos=-.17
    )

    output_fp = os.path.join(output_dir, "newt_stem_plot.pdf")
    plt.savefig(output_fp, bbox_inches='tight', dpi=400)
    
    ##############
    # Latex Table of results
    # [Source Dataset, Train Loss, {Dataasets}, Mean ACC]
    
    # Get the results of each method for each NeWT cluster:
    # GVH: this needs to be refactored (didn't we do this above...)
    model_cluster_results = []
    for model_name, model_results in newt_model_results_df.iterrows():

        r = {'model_name' : model_name}
        for task_info in task_order:

            # Get the dataset names that belong to this cluster (and subcluster)
            if task_info['sub_cluster'] is not None:
                cluster_dataset_names = newt_clusters_df[(newt_clusters_df['cluster'] == task_info['cluster']) & (newt_clusters_df['sub_cluster'] == task_info['sub_cluster'])]['name']
            else:
                cluster_dataset_names = newt_clusters_df[newt_clusters_df['cluster'] == task_info['cluster']]['name']

            scores = []
            for dataset_name in cluster_dataset_names:
                scores.append(model_results[dataset_name])
            r[task_info['name'].replace('\n', ' ')] = np.mean(scores)

        r['mean_acc'] = np.mean(model_results.values)

        model_cluster_results.append(r)

    model_cluster_results_df = pd.DataFrame(model_cluster_results).set_index('model_name', drop=True)
    
    
    tasks = [task['name'].replace('\n', ' ') for task in task_order]

    #result_names = [model_spec['name'] for model_spec in configs.model_specs]
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


    num_cols = len(tasks)
    num_rows = len(result_names)
    
    table_fp = os.path.join(output_dir, "newt_latex_table.txt")
    
    with open(table_fp, "w") as f:
        with redirect_stdout(f):
    
            print("\\begin{table*}[t]")
            print("\\small")
            print("\\centering")
            print("\\begin{tabular}{|l | l | %s |}" % (" ".join(["c"] * (num_cols + 1))))
            print("\\hline")

            header = ["Source Dataset", "Train Loss"] + tasks + ["Mean ACC"]
            print("  &  \t".join(header) + "\\\\")
            print("\hline\hline")

            for model_name in result_names:

                model_spec = next(model_spec for model_spec in configs.model_specs if model_spec['name'] == model_name)

                model_scores = model_cluster_results_df.loc[model_spec['name']]

                ys = []
                ry = []
                for i, label in enumerate(tasks):

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

                row = [td, to] + ys + ["%0.3f" % model_scores['mean_acc']] # We want the mean across all tasks, not the clusters

                print("  &  \t".join(row) + " \\\\")        

            print("\\hline")
            print("\\end{tabular}")
            print("\\caption{}")
            print("\\label{table:}")
            print("\\end{table*}")
    


def parse_args():
    
    parser = argparse.ArgumentParser(description='Create the stem plot figure and latex table of results for the NeWT tasks.')
    
    parser.add_argument('--result_dir', dest='result_dir',
                        help='Path to the directory containing the NeWT results.', type=str,
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
    
    create_newt_plots(args.result_dir, args.output_dir)