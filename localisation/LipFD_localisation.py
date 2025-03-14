

import argparse
from sklearn.base import accuracy_score
from sklearn.metrics import average_precision_score, confusion_matrix
import torch
import torchvision.transforms as transforms

from LipFD.data.datasets import AVLip
from LipFD.models import build_model
from LipFD.validate import validate
from dataloader.LAV_DF import create_lav_df_dataloader
import numpy as np

from localisation.preprocessed_dataloader import Preprocessed_LAVDF


def process_chunck(frames, img, crops_tens, chunk_start, chunk_end):
    for t_idx in range(chunk_start, chunk_end):
        frame = frames[0, t_idx]  # Shape: [C, H, W]
        # the images in the AVLip Dataset had height and width of 1000 and 2500 respectively 
        frame = transforms.Resize((1000, 2500))(frame)
        # Process crops for the current frame
        crops = [[transforms.Resize((224, 224))(frame[:, 500:, i:i + 500]) for i in range(5)], [], []]
        crop_idx = [(28, 196), (61, 163)]

        for i in range(len(crops[0])):
            crops[1].append(transforms.Resize((224, 224))
                                        (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            crops[2].append(transforms.Resize((224, 224))
                                        (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))

        frame = transforms.Resize((1120, 1120))(frame)
        img.append(frame)
        for scale_idx in range(3):
            for crop_idx in range(5):
                crops_tens[scale_idx][crop_idx].append(crops[scale_idx][crop_idx])
     # Stack imgs for the processed chunk
    imgs = torch.stack(img)  # Shape: [chunk_size, 3, 1120, 1120]

    # Stack crops for the processed chunk and convert to tensors
    for scale_idx in range(3):
        for crop_idx in range(5):
            # Stack along the frame dimension to form tensors of shape (chunk_size, 3, 224, 224)
            crops_tens[scale_idx][crop_idx] = torch.stack(crops_tens[scale_idx][crop_idx])

    return imgs, crops_tens
def validate(model, loader, gpu_id):
    print("validating...")
    device = torch.device(f"cuda:{gpu_id[0]}" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        y_true, y_pred = [], []
        for img, crops in loader:
            img_tens = img.to(device)
            crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
            features = model.get_features(img_tens).to(device)

            y_pred.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())
            y_true.extend(label.flatten().tolist())
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
    parser.add_argument("--real_list_path", type=str, default="./datasets/val/0_real")
    parser.add_argument("--fake_list_path", type=str, default="./datasets/val/1_fake")
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
    
    dataset = Preprocessed_LAVDF(opt)
    loader = data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=opt.batch_size, shuffle=True
    )
    ap, fpr, fnr, acc = validate(model, loader, gpu_id=[opt.gpu])
    print(f"acc: {acc} ap: {ap} fpr: {fpr} fnr: {fnr}")

    with torch.no_grad():
        scores= []
        for batch in loader:
            frames = batch["frames"]  # Shape: [# batch size, T, C, H, W]
            labels = batch["n_fakes"]  # Shape: [1]
            img = []
            crops_tens = [[[] for _ in range(5)] for _ in range(3)]  # 3 scales x 2 crop indices

            # Process frames in chunks of 16
            chunk_size = 16
            for chunk_start in range(0, frames.size(1), chunk_size):
                chunk_end = min(chunk_start + chunk_size, frames.size(1))
                imgs, crops_tens = process_chunck(frames, img, crops_tens, chunk_start, chunk_end)
                imgs_tens = imgs.to(device)
                crops_tens = [[t.to(device) for t in sublist] for sublist in crops_tens]
                features = model.get_features(imgs_tens).to(device)
                scores.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())

                # Clear img and crops_tens to reduce memory usage
                img = []
                crops_tens = [[[] for _ in range(5)] for _ in range(3)]
        scores = np.where(np.array(scores) >= 0.5, 1, 0)
        print(scores)