import statistics
from scipy.stats import norm
from collections import defaultdict
import csv
from metrics import AP, AR
from tqdm import tqdm


def load_and_process_data(file_path):
    video_data = defaultdict(list)
    label_data = defaultdict(list)
    
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            file_path, label_str, value = row
            
            parts = file_path.split('/')
            video_info = parts[-2]  # e.g. "1_fake"
            filename = parts[-1]    # e.g. "077627_9_2.png"

            is_fake = 1 if "_fake" in video_info else 0

            filename_parts = filename.split('_')
            video_id = filename_parts[0]
            order = int(filename_parts[1])

            video_data[video_id].append((order, float(value)))
            label_data[video_id].append((order, is_fake))
    
    sorted_videos = []
    sorted_labels = []
    
    for video_id in sorted(video_data.keys()):
        video_data[video_id].sort()
        label_data[video_id].sort()

        video_list = [v[1] for v in video_data[video_id]]
        label_list = [l[1] for l in label_data[video_id]]

        if len(video_list) >= 10:
            sorted_videos.append(video_list)
            sorted_labels.append(label_list)
    
    if sorted_videos:
        min_length = min(len(lst) for lst in sorted_videos)
        print(f"Shortest list length: {min_length}")
    else:
        print("No videos with at least 10 examples found.")
    
    return sorted_videos, sorted_labels

class DriftDetectionMethod:
    def __init__(self, detection_level, window_size):
        self.detection_level = detection_level
        self.detection_zscore = norm.ppf(detection_level)
        self.window_size = window_size
        self.collected_timeseries = []
        self.default_cluster = None
        self.non_default_cluster = None
        self.baseline = None
        self.baseline_standard_deviation = None

    def cluster(self, input):
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
                self.baseline_standard_deviation = statistics.stdev(self.collected_timeseries)
                print(statistics.mean(self.collected_timeseries))
                if statistics.mean(self.collected_timeseries) > 0.5:
                    self.default_cluster = 1
                    self.non_default_cluster = 0
                else:
                    self.default_cluster = 0
                    self.non_default_cluster = 1
            return round(input)

def calculate_fpr_fnr(predictions, labels):
    fp = fn = tp = tn = 0
    for pred_seq, label_seq in zip(predictions, labels):
        for pred, actual in zip(pred_seq, label_seq):
            if actual == 1 and pred == 0:
                fn += 1  # False Negative
            elif actual == 0 and pred == 1:
                fp += 1  # False Positive
            elif actual == 1 and pred == 1:
                tp += 1  # True Positive
            elif actual == 0 and pred == 0:
                tn += 1  # True Negative
    
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return fpr, fnr


if __name__ == "__main__":
    videos, labels = load_and_process_data("oli_prep_84-4_acc__test-set_preds_combined.csv")
    ddm = DriftDetectionMethod(0.9, 3)
    
    predictions = []
    for input in tqdm(videos, desc="Processing Videos..."):
        output_col = []
        for s in input:
            output_col.append(ddm.cluster(s))
        predictions.append(output_col)

    ap_calculator = AP(iou_thresholds=[0.5, 0.75, 0.95])
    print(ap_calculator(predictions, labels))

    ar_calculator = AR(n_proposals_list=[1, 3, 5, 10, 20, 50, 100])
    print(ar_calculator(predictions, labels))
    
    fpr, fnr = calculate_fpr_fnr(predictions, labels)
    print(f"False Positive Rate (FPR): {fpr:.4f}")
    print(f"False Negative Rate (FNR): {fnr:.4f}")
    print("Done")

        