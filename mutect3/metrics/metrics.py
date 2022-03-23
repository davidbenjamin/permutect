from collections import defaultdict

from mutect3.metrics.plotting import simple_plot


class LearningCurves:
    def __init__(self):
        # metrics[metric type] is a list by epoch
        self.metrics = defaultdict(list)

    def add(self, metric_type, value):
        self.metrics[metric_type].append(value)

    def plot_curves(self, metric_type):
        for metric_type in self.metrics.keys():
            values = self.metrics[metric_type]
        metric_dict = self.metrics[metric_type]
        epochs = range(1, len(values) + 1)
        x_y_lab = [(epochs, metric_dict, metric_type)]
        fig, curve = simple_plot(x_y_lab, x_label="epoch", y_label=metric_type, title="Learning curves: " + metric_type)
        return fig, curve
