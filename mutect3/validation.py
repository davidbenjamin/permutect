from collections import defaultdict
import math
import matplotlib.pyplot as plt
from mutect3.networks import F_score

class TrainingMetrics:
    NLL = "negative log-likelihood"
    F = "optimal F score"

    def __init__(self):
        # metrics[metric type][training type] is a list by epoch
        self.metrics = defaultdict(lambda: defaultdict(list))

    def add(self, metric_type, train_type, value):
        self.metrics[metric_type][train_type].append(value)

    def plot_metrics(self, metric_type):
        metric_dict = self.metrics[metric_type]
        train_types = list(metric_dict.keys())
        epochs = range(1, len(metric_dict[train_types[0]]) + 1)

        fig = plt.figure()
        curve = fig.gca()

        for train_type in train_types:
            curve.plot(epochs, metric_dict[train_type], label=train_type)
        curve.set_title("Learning curves: " + metric_type)
        curve.set_xlabel("epoch")
        curve.set_ylabel(metric_type)
        curve.legend()
        return fig, curve

    def plot_all_metrics(self):
        for metric_type in self.metrics.keys():
            self.plot_metrics(metric_type)

class ValidationStats:
    def __init__(self):
        self.confusion_by_count = defaultdict(lambda: [[0, 0], [0, 0]])
        self.confusion = [[0, 0], [0, 0]]

        self.artifact_scores_by_count = defaultdict(list)
        self.non_artifact_scores_by_count = defaultdict(list)

        self.missed_artifacts_by_count = defaultdict(list)
        self.missed_variants_by_count = defaultdict(list)

    # prediction and truth are 0 if not artifact, 1 if artifact
    def add(self, alt_count, truth, prediction, score, filters, position):
        alt_count_bin = ValidationStats._round_alt_count_for_binning(alt_count)
        self.confusion_by_count[alt_count_bin][truth][prediction] += 1
        self.confusion[truth][prediction] += 1
        (self.artifact_scores_by_count if truth == 1 else self.non_artifact_scores_by_count)[alt_count_bin].append(
            score)

        if truth == 1 and prediction == 0:
            self.missed_artifacts_by_count[alt_count_bin].append((score, position, filters))
        elif truth == 0 and prediction == 1:
            self.missed_variants_by_count[alt_count_bin].append((score, position, filters))

    def confusion_matrices(self):
        return self.confusion_by_count

    def confusion(self):
        return self.confusion

    def sensitivity(self):
        denom = self.confusion[0][0] + self.confusion[0][1]
        return 1.0 if denom == 0 else self.confusion[0][0] / denom

    def precision(self):
        denom = self.confusion[0][0] + self.confusion[1][0]
        return 1.0 if denom == 0 else self.confusion[0][0] / denom

    def artifact_scores(self):
        return self.artifact_scores_by_count

    def non_artifact_scores(self):
        return self.non_artifact_scores_by_count

    def worst_missed_variants(self, alt_count):
        alt_count_bin = ValidationStats._round_alt_count_for_binning(alt_count)
        # sort from highest score to lowest
        return sorted(self.missed_variants_by_count[alt_count_bin], key=lambda x: -x[0])

    def worst_missed_artifacts(self, alt_count):
        alt_count_bin = ValidationStats._round_alt_count_for_binning(alt_count)
        # sort from highest score to lowest
        return sorted(self.missed_artifacts_by_count[alt_count_bin], key=lambda x: x[0])

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

        fig = plt.figure()
        accuracy_curve = fig.gca()
        accuracy_curve.plot(counts, variant_sensitivities, label="variant sensitivity")
        accuracy_curve.plot(counts, artifact_sensitivities, label="artifact sensitivity")
        accuracy_curve.set_title("Variant and artifact sensitivity by alt count for " + name)
        accuracy_curve.set_xlabel("alt count")
        accuracy_curve.set_ylabel("sensitivity")
        accuracy_curve.legend()
        return fig, accuracy_curve

    def _round_alt_count_for_binning(alt_count):
        if alt_count < 15:
            return alt_count
        else:
            return alt_count - alt_count % 5


# note the m2 filters to keep here are different from those used to generate the training data
# above, they were filters that are not artifacts, such as germline, contamination, and weak evidence
# here, they are artifact filters that we intend to use in M3, such as the normal artifact filter

# threshold is threshold of logit prediction for considering variant an artifact -- this is a quick way to
# explore translating likelihoods from balanced training to posteriors, which we will alter do in a principled way
def get_validation_stats(model, loader, thresholds=[0.0]):
    all_stats = [ValidationStats() for _ in thresholds]

    model.train(False)
    for batch in loader:
        labels = batch.labels()
        filters = [m2.filters() for m2 in batch.mutect_info()]
        alt_counts = batch.alt_counts()
        predictions, _ = model(batch, posterior = True)
        positions = [meta.locus() for meta in batch.metadata()]
        for n in range(batch.size()):
            truth = 1 if labels[n].item() > 0.5 else 0
            for stats, threshold in zip(all_stats, thresholds):
                pred = 1 if predictions[n] > threshold  else 0
                stats.add(alt_counts[n].item(), truth, pred, predictions[n].item(), filters[n], positions[n])

    return all_stats


# compute optimal F score over a single epoch pass over the test loader, optionally doing SGD on the AF spectrum
def get_optimal_f_score_and_train_spectra(model, loader, optimizer=None):
    # tuples of (artifact prob, artifact label 0/1)
    predictions_and_labels = []

    model.learn_spectrum_mode()
    for batch in loader:
        labels = batch.labels()
        logits, log_evidence = model(batch, posterior=True)
        for n in range(batch.size()):
            predictions_and_labels.append((logits[n].item(), labels[n].item()))
        if optimizer is not None:
            optimizer.zero_grad()
            loss = -log_evidence
            loss.backward()
            optimizer.step()

    # sort tuples in ascending order of the model prediction
    predictions_and_labels.sort(key=lambda tuple: tuple[0])

    # start at threshold = -infinity; that is, everything is called an artifact, and pick up one variant at a time
    total_true = sum([(1 - label) for _, label in predictions_and_labels])
    tp, fp, best_F = 0, 0, 0
    for pred, label in predictions_and_labels:
        fp = fp + label
        tp = tp + (1 - label)
        best_F = max(best_F, F_score(tp, fp, total_true))
    return best_F


# get the same stats for Mutect2 using the M2 filters and truth labels
def get_m2_validation_stats(loader):
    stats = ValidationStats()

    for batch in loader:
        labels = batch.labels()
        filters = [m2.filters() for m2 in batch.mutect_info()]
        alt_counts = batch.alt_counts()
        positions = [meta.locus() for meta in batch.metadata()]
        for n in range(batch.size()):
            truth = 1 if labels[n].item() > 0.5 else 0
            pred = 0 if 'PASS' in filters[n] else 1
            score = 1 if pred == 1 else -1
            stats.add(alt_counts[n].item(), truth, pred, score, filters[n], positions[n])

    return stats

def show_validation_plots(model, loader, logit_threshold):
    m3_stats = get_validation_stats(model, loader, [logit_threshold])[0]
    m3_stats.plot_sensitivities("Mutect3 on test set")

    m2_stats = get_m2_validation_stats(loader)
    m2_stats.plot_sensitivities("Mutect2 on test set")

    roc_thresholds = [-16 + 0.5 * n for n in range(64)]
    roc_stats = get_validation_stats(model, loader, roc_thresholds)
    sens = [stats.sensitivity() for stats in roc_stats]
    prec = [stats.precision() for stats in roc_stats]

    labeled_indices = range(16, 48, 4)

    # minimum distance to sens = 1, prec = 1 corner\n",
    distance_to_corner = min(math.sqrt((1 - x) ** 2 + (1 - y) ** 2) for x, y in zip(sens, prec))
    roc_fig = plt.figure()
    roc_curve = roc_fig.gca()
    roc_curve.plot(sens, prec, label="ROC curve. Distance to corner: " + str(distance_to_corner))
    roc_curve.set_xlabel("sensitivity")
    roc_curve.set_ylabel("precision")
    roc_curve.scatter([m2_stats.sensitivity()], [m2_stats.precision()])
    roc_curve.annotate("Mutect2", (m2_stats.sensitivity(), m2_stats.precision()))
    roc_curve.scatter([m3_stats.sensitivity()], [m3_stats.precision()])
    roc_curve.annotate("Mutect3", (m3_stats.sensitivity(), m3_stats.precision()))

    for n in labeled_indices:
        roc_curve.scatter(roc_stats[n].sensitivity(), roc_stats[n].precision())
        roc_curve.annotate(str(roc_thresholds[n]), (roc_stats[n].sensitivity(), roc_stats[n].precision()))