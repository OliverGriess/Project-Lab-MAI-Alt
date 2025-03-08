from collections import defaultdict
import csv
from matplotlib import pyplot as plt
import numpy as np

def visualize_pred_dist(preds_path):
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
    plt.savefig(f'pred_dist.png')

if __name__ == "__main__":
    preds_path = "./oli_prep_91-1_acc__train-set.csv"
    visualize_pred_dist(preds_path)
    # best lower treshold is 0.4 and higher treshold is 0.8 for histeresis thresholding