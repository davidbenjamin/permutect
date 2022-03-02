from matplotlib import pyplot as plt

# one or more simple plots of y data vs x data on shared axes
from tqdm.notebook import tqdm


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


# compute optimal F score over a single epoch pass over the test loader, optionally doing SGD on the AF spectrum
def plot_roc_curve(model, loader, normal_artifact=False):
    # tuples of (artifact prob, artifact label 0/1)
    predictions_and_labels = []

    model.freeze_all()
    print("Running model over all data to generate ROC curve")

    pbar = tqdm(enumerate(loader))
    for _, batch in pbar:
        labels = batch.labels()
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