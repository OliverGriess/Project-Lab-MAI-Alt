

import csv
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
import numpy as np

def get_metrics(preds_path):
    y_true, y_pred = [], []
    with open(preds_path, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            img_path, _y_real, _y_pred = row
            y_true.append(float(_y_real))
            y_pred.append(float(_y_pred))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_pred = np.where(y_pred >= 0.5, 1, 0)

    print("processed ", len(y_true), "preds")

    # Get AP
    ap = average_precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = cm.ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = accuracy_score(y_true, y_pred)
    return ap, fpr, fnr, acc

if __name__ == "__main__":
    preds_path = "./oli_prep_91-1_acc__test-set_preds_sorted.csv"
    ap, fpr, fnr, acc = get_metrics(preds_path)
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")