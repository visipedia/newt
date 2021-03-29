from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def task_stem_plot(
    result_df,
    task_labels,
    task_space=2,
    task_offset=3,
    title='Average Group Performance',
    xlabel='tasks',
    ylabel='$\Delta$ ACC',
    figsize=(10, 5),
    rotate_x_tick_labels=True,
    task_baseline_scores=None,
    task_baseline_scores_x_offset=-.3,
    task_baseline_scores_y_pos=-.2
    ):
    """ Make a stem plot with results
    """

    num_methods = result_df.shape[0]
    num_tasks = result_df.iloc[0]['scores'].shape[0]

    fig = plt.figure(figsize=figsize)
    plt.title(title)


    offsets = np.linspace(0, task_space, num_methods) - task_space / 2.

    for j in range(num_tasks):
        plt.axvspan(j * task_offset + min(offsets) - 0.2, j * task_offset + max(offsets) + 0.2, facecolor='#dfe1df', alpha=1.)
        plt.plot([j * task_offset + min(offsets) - 0.2, j * task_offset + max(offsets) + 0.2], [0, 0], color='#777B7E', linestyle='-')

    i = 0
    for _, mtd in result_df.iterrows():

        ys = mtd['scores']
        c = mtd['color']
        s = offsets[i]
        ls = mtd['line_style']
        mf = mtd['marker_format']
        x = np.arange(num_tasks) * task_offset + s
        #plt.stem(x, ys, '%s%s' % (c, ls), markerfmt='%s%s' % (c, mf), use_line_collection=True, basefmt=" ")
        markerline, stemlines, baseline = plt.stem(x, ys, use_line_collection=True, basefmt=" ")
        plt.setp(stemlines, 'color', mtd['color'])
        plt.setp(stemlines, 'linestyle', mtd['line_style'])
        plt.setp(markerline, 'color', mtd['color'])
        plt.setp(markerline, 'marker', mtd['marker_format'])

        i += 1

    xticks = task_labels
    plt.xticks(np.arange(num_tasks) * task_offset, xticks, rotation='vertical' if rotate_x_tick_labels else None)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Print baseline performance on figure
    if task_baseline_scores is not None:
        ax = plt.gca()
        for j in range(num_tasks):
            ax.text(j * task_offset + task_baseline_scores_x_offset, task_baseline_scores_y_pos, "%0.2f" % (task_baseline_scores[j],))

    # Legend
    legend_names = []
    custom_lines = []

    for i, mtd in result_df.iterrows():
        legend_names.append(mtd['display_name'])
        custom_lines.append(Line2D([0], [0], marker=mtd['marker_format'], markersize=8, linestyle=mtd['line_style'], linewidth=1, color=mtd['color']))

    plt.legend(custom_lines, legend_names, loc='center left', bbox_to_anchor=(1, 0.5))