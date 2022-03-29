from collections import defaultdict

from mutect3.metrics.plotting import simple_plot


class LearningCurves:
    def __init__(self):
        # metrics[metric type] is a list by epoch
        self.metrics = defaultdict(list)

    def add(self, metric_type, value):
        self.metrics[metric_type].append(value)

    # return a list of (fig, curve) tuples
    def plot_curves(self):
        result = []
        for (metric_type, values) in self.metrics.items():
            epochs = range(1, len(values) + 1)
            x_y_lab = [(epochs, values, metric_type)]
            result.append(simple_plot(x_y_lab, x_label="epoch", y_label=metric_type, title="Learning curves: " + metric_type))
        return result
