import torch
import numpy as np
from data.DataLoader import create_dataloader
import torch.utils.data
from models import build_model
from sklearn.metrics import average_precision_score, confusion_matrix, accuracy_score
from tqdm import tqdm
import csv
from options.test_options import TestOptions


def store_predictions(preds_path, y_trues, y_preds, img_paths):
    with open(preds_path, "a") as pred_csv:
        writer = csv.writer(pred_csv)
        for i in range(len(y_trues)):
            writer.writerow([img_paths[i], y_trues[i], y_preds[i]])


def validate(model, loader, gpu_id, store_preds=False, preds_path="./preds.csv"):
    print("validating...", flush=True)
    device = torch.device(
        f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_true, y_pred = [], []
        for data in tqdm(loader, desc="Validation"):
            img = data["img"]
            label = data["label"]
            crops = data["crops"]
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            features = model.get_features(img_tens).to(device)

            _y_pred = model(crops_tens, features)[
                0].sigmoid().flatten().tolist()
            y_pred.extend(_y_pred)
            _y_true = label.flatten().tolist()
            y_true.extend(_y_true)

            if store_preds:
                store_predictions(preds_path, _y_true,
                                  _y_pred, data["img_path"])

    y_true = np.array(y_true)
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)

    # Get metrics
    ap = average_precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = cm.ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = accuracy_score(y_true, y_pred)
    return ap, fpr, fnr, acc


if __name__ == "__main__":
    opt = TestOptions().parse(print_options=False)

    predictions_path = "./predictions.csv"
    store_preds = True  # if the predictions should be stored as csv file

    device = torch.device(
        f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using cuda {opt.gpu} for inference.")

    model = build_model(opt.arch)
    state_dict = torch.load(opt.model_path, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    print("Model loaded.")
    model.eval()
    model.to(device)

    loader = create_dataloader(opt, train=False)

    ap, fpr, fnr, acc = validate(model, loader, gpu_id=[
                                 opt.gpu], store_preds=store_preds, preds_path=predictions_path)
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")
