




import argparse

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix,accuracy_score
import torch

from LipFD.models import build_model
from dataloader.LAV_DF import LAV_DF, create_lav_df_dataloader
from localisation.preprocessed_dataloader import Preprocessed_LAVDF
def hysteresis_threshold_1d(scores, low_thr, high_thr):
    """
    Apply hysteresis thresholding on a 1D array of scores.

    Parameters
    ----------
    scores : 1D list or numpy array of floats
        The scores (e.g. deep-fake likelihood) in [0, 1].
    low_thr : float
        The lower threshold for hysteresis.
    high_thr : float
        The upper threshold for hysteresis.

    Returns
    -------
    labels : numpy array of int
        An array of the same length as 'scores'.
        labels[i] = 1 --> fake
        labels[i] = 0 --> real
    """
    scores = np.array(scores, dtype=float)
    n = len(scores)

    # Step 1: Initialize labels to 0 (real)
    labels = np.zeros(n, dtype=int)

    # Step 2: Mark definitely fake (>= high_thr) as 1
    labels[scores >= high_thr] = 1

    # Step 3: Mark uncertain/floating values (-1) where low_thr <= score < high_thr
    uncertain_mask = (scores >= low_thr) & (scores < high_thr)
    labels[uncertain_mask] = -1

    # Step 4: Expand fake regions into uncertain neighbors until no more changes
    changed = True
    while changed:
        changed = False
        for i in range(n):
            if labels[i] == 1:
                # Check left neighbor
                if i > 0 and labels[i - 1] == -1:
                    labels[i - 1] = 1
                    changed = True
                # Check right neighbor
                if i < n - 1 and labels[i + 1] == -1:
                    labels[i + 1] = 1
                    changed = True

    # Step 5: All remaining -1 become 0 (real)
    labels[labels == -1] = 0

    return labels


def validate(model, loader, gpu_id):
    print("validating...")
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    print(device)
    with torch.no_grad():
        y_true, y_pred = [], []
        i=0
        for img, crops ,label in loader:
            print("processing batch ",i, "of ",len(loader))
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            features = model.get_features(img_tens).to(device)

            y_pred.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
            i+=1
    y_true = np.array(y_true)
    np.savetxt("/work/scratch/kurse/kurs00079/om43juhy/Project-Lab-MAI-Alt/y_true.txt", y_true, fmt='%f')
    y_pred = hysteresis_threshold_1d(y_pred,0.3,0.7)

    # Get AP
    ap = average_precision_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    tp, fn, fp, tn = cm.ravel()
    fnr = fn / (fn + tp)
    fpr = fp / (fp + tn)
    acc = accuracy_score(y_true, y_pred)
    return ap, fpr, fnr, acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_root", type=str, default="/work/scratch/kurse/kurs00079/data/LAV-DF/LipFD_preprocess")
    parser.add_argument("--data_root_lavdf", type=str, default="/work/scratch/kurse/kurs00079/data/LAV-DF")
    parser.add_argument("--max_sample", type=int, default=1000, help="max number of validate samples")
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--data_label", type=str, default="val")
    parser.add_argument("--arch", type=str, default="CLIP:ViT-L/14")
    parser.add_argument("--ckpt", type=str, default="LipFD/checkpoints/ckpt.pth")
    parser.add_argument("--gpu", type=int, default=0)

    opt = parser.parse_args()
    device = torch.device(f"cuda:{opt.gpu}" if torch.cuda.is_available() else "cpu")
    
    # Load Model
    model = build_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    print("Model loaded.")
    print(device)
    model.eval()
    model.to(device)
    
    preprocessed_dataset = Preprocessed_LAVDF(opt)
    loader = data_loader = torch.utils.data.DataLoader(
        preprocessed_dataset, batch_size=opt.batch_size, shuffle=True
    )


    LAVDF_dataset = LAV_DF(data_root="/work/scratch/kurse/kurs00079/data/LAV-DF")
    with torch.no_grad():
        ap, fpr, fnr, acc = validate(model, loader, gpu_id=[opt.gpu])
        print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")