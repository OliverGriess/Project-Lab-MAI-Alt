import os
import cv2
import torchvision.transforms as transforms
import torch
from torch.utils.data import Dataset

from LipFD import utils
from LipFD.preprocess import N_EXTRACT, WINDOW_LEN
from dataloader.LAV_DF import LAV_DF


class Preprocessed_LAVDF(Dataset):
    def __init__(self, opt):
        self.dataset = LAV_DF(opt.data_root_lavdf)

        self.total_list = utils.get_list(opt.data_root)

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        print(f"Attempting to read image at path: {img_path}")
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        label =0
        filename = os.path.basename(img_path)
        fileId, group = filename.split('_')
        group = int(group.split('.')[0])  # Convert group to int
        # Use filter to find the matching sample
        filtered_samples = filter(lambda s: s["file_id"] == fileId, self.dataset.train_list)
        
        # Get the first match (filter returns an iterator)
        sample = next(filtered_samples, None)
        video_capture = cv2.VideoCapture(sample["video_path"])
        frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        video_capture.release()
        # Calculate the starting frame for each group
        start_frame = round(group * (frame_count - WINDOW_LEN - 1) / (N_EXTRACT - 1))

        # Calculate the frame indices for the group
        frame_indices = [start_frame + i for i in range(WINDOW_LEN)]
        fake_periods = sample["fake_periods"]
        fake_frame_ranges = [
            (int(period[0] * fps), int(period[1] * fps)) for period in fake_periods
        ]
        
        for frame_index in frame_indices:
            fake = any(start <= frame_index <= end for start, end in fake_frame_ranges)
            if fake:  label = 1 
        crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])(img)
        # crop images
        # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
        crops = [[transforms.Resize((224, 224))(img[:, 500:, i:i + 500]) for i in range(5)], [], []]
        crop_idx = [(28, 196), (61, 163)]
        for i in range(len(crops[0])):
            crops[1].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            crops[2].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
        img = transforms.Resize((1120, 1120))(img)
        return img, crops,label
