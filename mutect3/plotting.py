import matplotlib.pyplot as plt


def plot_learning_metrics(losses, names, type):
    """
    losses: iterable of lists of losses by epoch.  Each loss list must have same number of epochs.

    names: type of loss for plot label.  Must have one name per loss list.

    type: type of metric
    """
    fig = plt.figure()
    learning_curve = fig.gca()
    epochs = range(1, len(losses[0]) + 1)

    for loss, name in zip(losses, names):
        learning_curve.plot(epochs, loss, label=name)
    learning_curve.set_title("Learning curves: " + type)
    learning_curve.set_xlabel("epoch")
    learning_curve.set_ylabel(type)
    learning_curve.legend()
    return fig, learning_curve
