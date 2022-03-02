from collections import defaultdict

from mutect3.metrics.plotting import simple_plot


class TrainingMetrics:
    def __init__(self):
        # metrics[metric type][training type] is a list by epoch
        self.metrics = defaultdict(lambda: defaultdict(list))

    def add(self, metric_type, train_type, value):
        self.metrics[metric_type][train_type].append(value)

    def plot_metrics(self, metric_type):
        metric_dict = self.metrics[metric_type]
        train_types = list(metric_dict.keys())
        epochs = range(1, len(metric_dict[train_types[0]]) + 1)
        x_y_lab = [(epochs, metric_dict[typ], typ) for typ in train_types]
        fig, curve = simple_plot(x_y_lab, x_label="epoch", y_label=metric_type, title="Learning curves: " + metric_type)
        return fig, curve