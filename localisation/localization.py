




import argparse

import numpy as np
from sklearn.metrics import average_precision_score, confusion_matrix,accuracy_score
import torch

from LipFD.models import build_model
from dataloader.LAV_DF import LAV_DF, create_lav_df_dataloader
from localisation.preprocessed_dataloader import Preprocessed_LAVDF

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
    y_pred = np.where(np.array(y_pred) >= 0.5, 1, 0)

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