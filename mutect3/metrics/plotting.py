from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import math
from typing import List

# one or more simple plots of y data vs x data on shared axes
from tqdm.autonotebook import tqdm


def simple_plot(x_y_lab_tuples, x_label, y_label, title):
    fig = plt.figure()
    curve = fig.gca()
    for (x, y, lab) in x_y_lab_tuples:
        curve.plot(x, y, label=lab)
    curve.set_title(title)
    curve.set_xlabel(x_label)
    curve.set_ylabel(y_label)
    curve.legend()
    return fig, curve


def simple_plot_on_axis(ax, x_y_lab_tuples, x_label, y_label):
    for (x, y, lab) in x_y_lab_tuples:
        ax.plot(x, y, label=lab)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.legend()


# apply grouped bar plot to an axis (subplot) object
# heights by category is a dict of category to bar heights, where the nth bar height
# corresponds to the nth x label
def grouped_bar_plot_on_axis(ax, heights_by_category, x_labels, y_label):
    spacing = 5
    bar_width = 0.7 * spacing / len(heights_by_category)

    for n, (category, heights) in enumerate(heights_by_category.items()):
        offset = n * bar_width
        x_positions = [offset + spacing*i for i in range(len(heights))]
        ax.bar(x_positions, heights, width=bar_width, edgecolor='white', label=category)

    # Add xticks on the middle of the group bars
    # plt.xlabel('group', fontweight='bold')
    ticks_offset = bar_width * len(heights_by_category)/2
    ax.set_xticks([ticks_offset + spacing*i for i in range(len(x_labels))], labels=x_labels)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel(y_label)
    ax.legend()


# heights by category is a dict of category to bar heights, where the nth bar height
# corresponds to the nth x label
def grouped_bar_plot(heights_by_category, x_labels, y_label):
    fig, ax = plt.subplots()
    grouped_bar_plot_on_axis(ax, heights_by_category, x_labels, y_label)
    return fig, ax


# labels are 0 for non-artifact, 1 for artifact
# predictions_and_labels has form [[(pred, label), (pred, label). . . for roc 1], [likewise for roc 2] etc]
def plot_accuracy_vs_accuracy_roc_on_axis(lists_of_predictions_and_labels, curve_labels, axis):
    x_y_lab_tuples = []
    dots = []
    for predictions_and_labels, curve_label in zip(lists_of_predictions_and_labels, curve_labels):
        thresh_and_accs, _ = get_roc_data(predictions_and_labels)
        x_y_lab_tuples.append(([x[1] for x in thresh_and_accs], [x[2] for x in thresh_and_accs], curve_label))

        for threshold, art_acc, non_art_acc in thresh_and_accs:
            dots.append((art_acc, non_art_acc, 'ro' if threshold == 0 else 'go'))

    simple_plot_on_axis(axis, x_y_lab_tuples, "artifact accuracy", "non-artifact accuracy")
    for x, y, spec in dots:
        axis.plot(x, y, spec, markersize=2)  # point


# input is list of (predicted artifact logit, binary artifact/non-artifact label) tuples
# 1st output is (threshold, accuracy on artifacts, accuracy on non-artifacts) tuples
# 2nd output is the threshold that maximizes harmonic mean of accuracy on artifacts and accuracy on non-artifacts
def get_roc_data(predictions_and_labels):
    predictions_and_labels.sort(key=lambda p_and_l: p_and_l[0])  # sort from least to greatest artifact logit
    num_calls = len(predictions_and_labels)
    total_artifact = sum([label for _, label in predictions_and_labels]) + 0.0001
    total_non_artifact = num_calls - total_artifact + 0.0002
    # start at threshold = -infinity; that is, everything is called an artifact, and pick up one variant at a time
    thresh_and_accs = []  # tuples of threshold, accuracy on artifacts, accuracy on non-artifacts
    art_found, non_art_found = total_artifact, 0
    next_threshold = -10
    best_threshold, best_harmonic_mean = -99999, 0
    for pred_logit, label in predictions_and_labels:
        art_found -= label  # if labeled as artifact, one artifact has slipped below threshold
        non_art_found += (1 - label)  # if labeled as non-artifact, one non-artifact has been gained
        art_acc, non_art_acc = art_found / total_artifact, non_art_found / total_non_artifact
        harmonic_mean = 0 if (art_acc == 0 or non_art_acc == 0) else 1/((1/art_acc) + (1/non_art_acc))

        if harmonic_mean > best_harmonic_mean:
            best_harmonic_mean = harmonic_mean
            best_threshold = pred_logit

        if pred_logit > next_threshold:
            thresh_and_accs.append((next_threshold, art_acc, non_art_acc))
            next_threshold = math.ceil(pred_logit)
    return thresh_and_accs, best_threshold


def tidy_subplots(figure: Figure, axes, x_label: str = None, y_label: str = None,
                  column_labels: List[str] = None, row_labels: List[str] = None):
    """
    Combines various tidying operations on figures with subplots
    1.  Removes the individual axis legends and replaces with a single figure legend.  This assumes
        that all axes have the same lines.
    2.  Show x (y) labels and tick labels only in bottom row (leftmost column)
    3.  Apply column headings and row labels
    4.  Apply overall x and y labels to the figure as a whole

    figure matplotlib.figure.Figure
    axes:   2D array of matplotlib.axes.Axes

    We assume these have been generated together via figure, axes = plt.subplots(. . .)

    """
    handles, labels = figure.get_axes()[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center')

    for ax in figure.get_axes():
        ax.label_outer()  # y tick labels only shown in leftmost column, x tick labels only shown on bottom row
        ax.legend().set_visible(False)  # hide the redundant identical subplot legends

        # remove the subplot labels and title -- these will be given manually to the whole figure and to the outer rows
        ax.set_xlabel(None)
        ax.set_ylabel(None)
        ax.set_title(None)

    if x_label is not None:
        figure.supxlabel(x_label)

    if y_label is not None:
        figure.supylabel(y_label)

    if row_labels is not None:
        assert len(row_labels) == len(axes)
        for row_idx, label in enumerate(row_labels):
            axes[row_idx][0].set_ylabel(label)   # note that we use row 0 and set_title to make this a column heading

    if column_labels is not None:
        assert len(column_labels) == len(axes[0])
        for col_idx, label in enumerate(column_labels):
            axes[0][col_idx].set_title(label)

    figure.tight_layout()

