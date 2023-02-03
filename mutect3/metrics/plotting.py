from matplotlib import pyplot as plt

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


def simple_bar_plot_on_axis(ax, heights, x_labels, y_label):
    spacing = 3
    bar_width = 0.7 * spacing

    x_positions = [spacing*i for i in range(len(heights))]
    ax.bar(x_positions, heights, width=bar_width, edgecolor='white')
    ax.set_xticks(x_positions, labels=x_labels)
    plt.setp(ax.get_xticklabels(), rotation=90)
    ax.set_ylabel(y_label)
    # ax.legend()


def simple_bar_plot(heights, x_labels, y_label):
    fig, ax = plt.subplots()
    simple_bar_plot_on_axis(ax, heights, x_labels, y_label)
    return fig, ax


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


def histogram(data, title):
    fig = plt.figure()
    curve = fig.gca()
    curve.hist(data, bins=20)
    curve.set_title(title)
    return fig, curve


def hexbin(x,y):
    fig = plt.figure()
    curve = fig.gca()
    curve.hexbin(x, y)
    return fig, curve


# compute optimal F score over a single epoch pass over the test loader, optionally doing SGD on the AF spectrum
def plot_roc_curve(model, loader, normal_artifact=False):
    # tuples of (artifact prob, artifact label 0/1)
    predictions_and_labels = []

    model.freeze_all()
    print("Running model over all data to generate ROC curve")

    pbar = tqdm(enumerate(loader), mininterval=10)
    for _, batch in pbar:
        labels = batch.labels
        logits = model(batch, posterior=True, normal_artifact=normal_artifact)
        for n in range(batch.size()):
            predictions_and_labels.append((logits[n].item(), labels[n].item()))

    # sort tuples in ascending order of the model prediction
    predictions_and_labels.sort(key=lambda prediction_and_label: prediction_and_label[0])

    sensitivity = []
    precision = []

    # start at threshold = -infinity; that is, everything is called an artifact, and pick up one variant at a time
    total_true = sum([(1 - label) for _, label in predictions_and_labels])
    tp, fp = 0, 0
    for _, label in predictions_and_labels:
        fp = fp + label
        tp = tp + (1 - label)
        sensitivity.append(tp / total_true)
        precision.append(tp / (tp + fp + 0.00001))

    x_y_lab = [(sensitivity, precision, "ROC")]
    return simple_plot(x_y_lab, x_label="sensitivity", y_label="precision", title="ROC curve according to M3's own probabilities.")