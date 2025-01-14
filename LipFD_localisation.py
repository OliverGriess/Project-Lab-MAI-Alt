

import argparse
import torch
import torchvision.transforms as transforms

from LipFD.models import build_model
from dataloader.LAV_DF import create_lav_df_dataloader
import numpy as np


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
    model = build_model(opt.arch)
    state_dict = torch.load(opt.ckpt, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    print("Model loaded.")
    model.eval()
    model.to(device)
    data_loader = create_lav_df_dataloader(data_root="/work/scratch/kurse/kurs00079/data/LAV-DF")
    with torch.no_grad():
        scores= []
        for batch in data_loader:
            frames = batch["frames"].to(device)  # Shape: [batch_size, T, C, H, W]
            labels = batch["n_fakes"].to(device)  # Shape: [batch_size]
            for batch_idx in range(frames.size(0)):
                for t in range(frames.size(1)):  # Iterate through the temporal dimension
                    frame = frames[batch_idx, t]  # Shape: [C, H, W]
                    # Permute the frame
                    
                    # Normalize the frame
                    frame = transforms.Normalize(
                        mean=[0.48145466, 0.4578275, 0.40821073],
                        std=[0.26862954, 0.26130258, 0.27577711],
                    )(frame)  

                    crops = [[transforms.Resize((224, 224))(frame[:, max(0, frame.size(1) - 500):, i:i + 500]) for i in range(0, frame.size(2), 500)], [], []]
                    crop_idx = [(28, 196), (61, 163)]
                    for i in range(len(crops[0])):
                        crops[1].append(transforms.Resize((224, 224))
                                        (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
                        crops[2].append(transforms.Resize((224, 224))
                                        (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
                    frame = transforms.Resize((1120, 1120))(frame)

                    frame_tens = frame.to(device)
                    crops_tens = [[t.to(device) for t in sublist] for sublist in crops]
                    features = model.get_features(frame_tens).to(device)
                    
                    scores.extend(model(crops_tens, features)[0].sigmoid().flatten().tolist())


        scores = np.where(np.array(scores) >= 0.5, 1, 0)
        print(scores)