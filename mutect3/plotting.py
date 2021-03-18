import matplotlib.pyplot as plt


def plot_learning_curves(train_losses, valid_losses):
    fig = plt.figure()
    learning_curve = fig.gca()
    epochs = range(1, len(train_losses) + 1)
    learning_curve.plot(epochs, train_losses, label="training")
    learning_curve.plot(epochs, valid_losses, label="validation")
    learning_curve.set_title("Learning curves")
    learning_curve.set_xlabel("epoch")
    learning_curve.set_ylabel("loss")
    learning_curve.legend()
    return fig, learning_curve

#TODO: unify this with the function above
def plot_learning_metrics(valid_metrics, test_metrics):
    fig = plt.figure()
    learning_curve = fig.gca()
    epochs = range(1, len(valid_metrics) + 1)
    learning_curve.plot(epochs, valid_metrics, label="validation")
    learning_curve.plot(epochs, test_metrics, label="test")
    learning_curve.set_title("Maximal F score")
    learning_curve.set_xlabel("epoch")
    learning_curve.set_ylabel("metric")
    learning_curve.legend()
    return fig, learning_curve

# confusion_matrices is dict of alt count to 2x2 [[,],[,]] confusion matrices where 0/1 is non-artifact/artifact
# and 1st index is truth, 2nd index is prediction
def plot_sensitivities(confusion_matrices, name):
    counts = []
    variant_sensitivities = []
    artifact_sensitivities = []
    for alt_count_bin in sorted(confusion_matrices.keys()):
        matrix = confusion_matrices[alt_count_bin]
        if (matrix[0][0] + matrix[0][1]) == 0 or (matrix[1][0] + matrix[1][1]) == 0:
            continue

        counts.append(alt_count_bin)
        variant_sensitivities.append(matrix[0][0] / (matrix[0][0] + matrix[0][1]))
        artifact_sensitivities.append(matrix[1][1] / (matrix[1][0] + matrix[1][1]))

    fig = plt.figure()
    accuracy_curve = fig.gca()
    accuracy_curve.clear()
    accuracy_curve.plot(counts, variant_sensitivities, label = "variant sensitivity")
    accuracy_curve.plot(counts, artifact_sensitivities, label = "artifact sensitivity")
    accuracy_curve.set_title("Variant and artifact sensitivity by alt count for " + name)
    accuracy_curve.set_xlabel("alt count")
    accuracy_curve.set_ylabel("sensitivity")
    accuracy_curve.legend()
    return fig, accuracy_curve
