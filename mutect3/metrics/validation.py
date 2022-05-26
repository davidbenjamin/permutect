import math
from collections import defaultdict

from mutect3.metrics.plotting import simple_plot

# note the m2 filters to keep here are different from those used to generate the training data
# above, they were filters that are not artifacts, such as germline, contamination, and weak evidence
# here, they are artifact filters that we intend to use in M3, such as the normal artifact filter

def round_alt_count_for_binning(alt_count):
    if alt_count < 15:
        return alt_count
    else:
        return alt_count - alt_count % 5


class ValidationStats:
    def __init__(self):
        self.confusion_by_count = defaultdict(lambda: [[0, 0], [0, 0]])
        self.confusion = [[0, 0], [0, 0]]

        # artifact and non-artifact scores/logits by alt count
        self.artifact_scores = defaultdict(list)
        self.non_artifact_scores = defaultdict(list)

        self.missed_artifacts = defaultdict(list)
        self.missed_variants = defaultdict(list)

    # prediction and truth are 0 if not artifact, 1 if artifact
    def add(self, alt_count, truth, prediction, score, filters, position):
        alt_count_bin = round_alt_count_for_binning(alt_count)
        self.confusion_by_count[alt_count_bin][truth][prediction] += 1
        self.confusion[truth][prediction] += 1
        (self.artifact_scores if truth == 1 else self.non_artifact_scores)[alt_count_bin].append(score)

        if truth != prediction:
            (self.missed_variants if truth == 1 else self.missed_variants)[alt_count_bin].append(
                (score, position, filters))

    def confusion_matrices(self):
        return self.confusion_by_count

    def confusion(self):
        return self.confusion

    def sensitivity(self):
        denominator = self.confusion[0][0] + self.confusion[0][1]
        return 1.0 if denominator == 0 else self.confusion[0][0] / denominator

    def precision(self):
        denominator = self.confusion[0][0] + self.confusion[1][0]
        return 1.0 if denominator == 0 else self.confusion[0][0] / denominator

    def artifact_scores(self):
        return self.artifact_scores

    def non_artifact_scores(self):
        return self.non_artifact_scores

    def worst_missed_variants(self, alt_count):
        alt_count_bin = round_alt_count_for_binning(alt_count)
        # sort from the highest score to lowest
        return sorted(self.missed_variants[alt_count_bin], key=lambda x: -x[0])

    def worst_missed_artifacts(self, alt_count):
        alt_count_bin = round_alt_count_for_binning(alt_count)
        # sort from the highest score to lowest
        return sorted(self.missed_artifacts[alt_count_bin], key=lambda x: x[0])

    # confusion_matrices is dict of alt count to 2x2 [[,],[,]] confusion matrices where 0/1 is non-artifact/artifact
    # and 1st index is truth, 2nd index is prediction
    def plot_sensitivities(self, name):
        counts = []
        variant_sensitivities = []
        artifact_sensitivities = []
        for alt_count_bin in sorted(self.confusion_matrices().keys()):
            matrix = self.confusion_matrices()[alt_count_bin]
            if (matrix[0][0] + matrix[0][1]) == 0 or (matrix[1][0] + matrix[1][1]) == 0:
                continue

            counts.append(alt_count_bin)
            variant_sensitivities.append(matrix[0][0] / (matrix[0][0] + matrix[0][1]))
            artifact_sensitivities.append(matrix[1][1] / (matrix[1][0] + matrix[1][1]))

        x_y_lab = [(counts, variant_sensitivities, "variant sensitivity"),
                   (counts, artifact_sensitivities, "artifact sensitivity")]
        fig, curve = simple_plot(x_y_lab, x_label="alt count", y_label="sensitivity",
                                 title="Variant and artifact sensitivity by alt count for " + name)
        return fig, curve


def get_validation_stats(model, loader, thresholds=[0.0]):
    all_stats = [ValidationStats() for _ in thresholds]

    model.train(False)
    for batch in loader:
        labels = batch.labels()
        filters = [m2.filters() for m2 in batch.mutect_info()]
        alt_counts = batch.alt_counts()
        predictions = model(batch, posterior=True)
        positions = [datum.locus() for datum in batch]
        for n in range(batch.size()):
            truth = 1 if labels[n].item() > 0.5 else 0
            for stats, threshold in zip(all_stats, thresholds):
                pred = 1 if predictions[n] > threshold else 0
                stats.add(alt_counts[n].item(), truth, pred, predictions[n].item(), filters[n], positions[n])

    return all_stats


def show_validation_plots(model, loader, logit_threshold):
    m3_stats = get_validation_stats(model, loader, [logit_threshold])[0]
    m3_stats.plot_sensitivities("Mutect3 on test set")

    roc_thresholds = [-16 + 0.5 * n for n in range(64)]
    roc_stats = get_validation_stats(model, loader, roc_thresholds)
    sens = [stats.sensitivity() for stats in roc_stats]
    prec = [stats.precision() for stats in roc_stats]

    labeled_indices = range(16, 48, 4)

    # minimum distance to sens = 1, prec = 1 corner\n",
    distance_to_corner = min(math.sqrt((1 - x) ** 2 + (1 - y) ** 2) for x, y in zip(sens, prec))

    x_y_lab = [(sens, prec, "ROC")]
    roc_fig, roc_curve = simple_plot(x_y_lab, x_label="sensitivity", y_label="precision",
                                     title="ROC curve. Distance to corner: " + str(distance_to_corner))
    roc_curve.scatter([m3_stats.sensitivity()], [m3_stats.precision()])
    roc_curve.annotate("Mutect3", (m3_stats.sensitivity(), m3_stats.precision()))

    for n in labeled_indices:
        roc_curve.scatter(roc_stats[n].sensitivity(), roc_stats[n].precision())
        roc_curve.annotate(str(roc_thresholds[n]), (roc_stats[n].sensitivity(), roc_stats[n].precision()))