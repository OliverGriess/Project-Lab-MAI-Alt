import argparse
import csv
from matplotlib import pyplot as plt


def visualize_pred_dist(preds_path: str, output_path: str):
    """
    Visualize the distribution of prediction scores for real and fake images

    Args:
        preds_path: Path to the CSV file containing the prediction scores (each row contains an image path, a real label, and a predicted score)
    """
    real_preds = []
    fake_preds = []

    with open(preds_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            img_path, y_real, y_pred = row
            if y_real == '0':  # Real images
                real_preds.append(float(y_pred))
            else:  # Fake images
                fake_preds.append(float(y_pred))

    plt.figure(figsize=(10, 6))
    plt.hist(real_preds, bins=50, alpha=0.6, color='blue', label='Real Images')
    plt.hist(fake_preds, bins=50, alpha=0.6, color='red', label='Fake Images')

    plt.xlabel('Prediction Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Prediction Scores for Real and Fake Images')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    plt.savefig(output_path)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--predictions", type=str, default="./predictions.csv")
    args.add_argument("--output", type=str, default="./pred_dist.png")
    args = args.parse_args()
    visualize_pred_dist(args.predictions, args.output)
