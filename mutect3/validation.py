from collections import defaultdict

from mutect3.threshold import F_score


def round_alt_count_for_binning(alt_count):
    if alt_count < 15:
        return alt_count
    else:
        return alt_count - alt_count % 5


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
        alt_count_bin = round_alt_count_for_binning(alt_count)
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
        return self.confusion[0][0] / (self.confusion[0][0] + self.confusion[0][1])

    def precision(self):
        return self.confusion[0][0] / (self.confusion[0][0] + self.confusion[1][0])

    def artifact_scores(self):
        return self.artifact_scores_by_count

    def non_artifact_scores(self):
        return self.non_artifact_scores_by_count

    def worst_missed_variants(self, alt_count):
        alt_count_bin = round_alt_count_for_binning(alt_count)
        # sort from highest score to lowest
        return sorted(self.missed_variants_by_count[alt_count_bin], key=lambda x: -x[0])

    def worst_missed_artifacts(self, alt_count):
        alt_count_bin = round_alt_count_for_binning(alt_count)
        # sort from highest score to lowest
        return sorted(self.missed_artifacts_by_count[alt_count_bin], key=lambda x: x[0])


# note the m2 filters to keep here are different from those used to generate the training data
# above, they were filters that are not artifacts, such as germline, contamination, and weak evidence
# here, they are artifact filters that we intend to use in M3, such as the normal artifact filter

# threshold is threshold of logit prediction for considering variant an artifact -- this is a quick way to
# explore translating likelihoods from balanced training to posteriors, which we will alter do in a principled way
def get_validation_stats(model, loader, m2_filters_to_keep={}, thresholds=[0.0]):
    all_stats = [ValidationStats() for _ in thresholds]

    model.train(False)
    for batch in loader:
        labels = batch.labels()
        filters = [m2.filters() for m2 in batch.mutect2_data()]
        alt_counts = batch.alt_counts()
        predictions = model(batch)
        positions = [meta.locus() for meta in batch.metadata()]
        for n in range(batch.size()):
            truth = 1 if labels[n].item() > 0.5 else 0
            for stats, threshold in zip(all_stats, thresholds):
                pred = 1 if (predictions[n] > threshold or filters[n].intersection(m2_filters_to_keep)) else 0
                stats.add(alt_counts[n].item(), truth, pred, predictions[n].item(), filters[n], positions[n])

    return all_stats


def get_optimal_f_score(model, loader, m2_filters_to_keep={}):
    # tuples of (artifact prob, artifact label 0/1)
    predictions_and_labels = []

    model.train(False)
    for batch in loader:
        labels = batch.labels()
        predictions = model(batch)
        filters = [m2.filters() for m2 in batch.mutect2_data()]
        for n in range(batch.size()):
            pred = 1 if filters[n].intersection(m2_filters_to_keep) else predictions[n].item()
            predictions_and_labels.append((pred, labels[n].item()))

    # sort tuples in ascending order of the model prediction
    predictions_and_labels.sort(key=lambda tuple: tuple[0])

    # start at threshold = -infinity; that is, everything is called an artifact
    # hence there are no false positives and every true variant is a false negative
    total_true = len([1 for pred, label in predictions_and_labels if label < 0.5])
    tp = 0
    fp = 0
    best_F = 0
    for pred, label in predictions_and_labels:
        # now increase the (implicit) threshold and call this variant good
        # picking up a tp or fp accordingly
        if label > 0.5:
            fp = fp + 1
        else:
            tp = tp + 1

        F = F_score(tp, fp, total_true)
        if F > best_F:
            best_F = F
    return best_F


# get the same stats for Mutect2 using the M2 filters and truth labels
def get_m2_validation_stats(loader):
    stats = ValidationStats()

    for batch in loader:
        labels = batch.labels()
        filters = [m2.filters() for m2 in batch.mutect2_data()]
        alt_counts = batch.alt_counts()
        positions = [meta.locus() for meta in batch.metadata()]
        for n in range(batch.size()):
            truth = 1 if labels[n].item() > 0.5 else 0
            pred = 0 if 'PASS' in filters[n] else 1
            score = 1 if pred == 1 else -1
            stats.add(alt_counts[n].item(), truth, pred, score, filters[n], positions[n])

    return stats