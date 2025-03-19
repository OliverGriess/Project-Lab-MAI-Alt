from localize.TimeSerieClustering import TimeSeriesClustering
from localize.utils import load_and_process_data
from eval.utils import evaluate_whole_dataset

if __name__ == "__main__":
    # videos, labels, combined_data = load_and_process_data(["data84.csv"])
    videos, labels, combined_data = load_and_process_data(
        ["data91.csv", "data_batch_16.csv", "data_batch_17.csv", "data_batch_18.csv", "data_batch_13.csv", "data_batch_14.csv", "data_batch_12.csv", "data_batch_11.csv"])
    # videos, labels, combined_data = load_and_process_data_2("pred_final_new_model.csv")
    # Example usage:
    tsc_instance = TimeSeriesClustering()
    evaluate_whole_dataset(videos, labels, "my_tsc_state.joblib")
    print("")
