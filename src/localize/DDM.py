import statistics
from scipy.stats import norm


class DriftDetectionMethod:
    def __init__(self, detection_level: float, window_size: int):
        self.detection_level = detection_level
        self.detection_zscore = norm.ppf(detection_level)
        self.window_size = window_size
        self.collected_timeseries = []
        self.default_cluster = None
        self.non_default_cluster = None
        self.baseline = None
        self.baseline_standard_deviation = None

    def cluster(self, input: float):
        if self.default_cluster is not None:
            self.collected_timeseries.append(input)
            std_dev = statistics.stdev(self.collected_timeseries)
            if input + std_dev >= self.baseline + self.baseline_standard_deviation * self.detection_zscore:
                if self.default_cluster == 1:
                    output = self.default_cluster
                else:
                    output = self.non_default_cluster
            elif input - std_dev <= self.baseline - self.baseline_standard_deviation * self.detection_zscore:
                if self.default_cluster == 0:
                    output = self.default_cluster
                else:
                    output = self.non_default_cluster
            else:
                output = self.default_cluster

            self.collected_timeseries = self.collected_timeseries[1:]
            return output

        else:
            self.collected_timeseries.append(input)
            if self.window_size <= len(self.collected_timeseries):
                self.baseline = statistics.mean(self.collected_timeseries)
                self.baseline_standard_deviation = statistics.stdev(
                    self.collected_timeseries)
                print(statistics.mean(self.collected_timeseries))
                if statistics.mean(self.collected_timeseries) > 0.5:
                    self.default_cluster = 1
                    self.non_default_cluster = 0
                else:
                    self.default_cluster = 0
                    self.non_default_cluster = 1
            return round(input)
