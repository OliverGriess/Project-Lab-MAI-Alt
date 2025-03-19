
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch
import cv2
from . import utils

MAX_SAMPLES = 100_000


class LavDF(Dataset):
    def __init__(self, opt):
        assert opt.data_label in ["train", "val"]
        self.data_label = opt.data_label

        self.real_list = utils.get_list(opt.real_list_path)

        self.fake_list = utils.get_list(opt.fake_list_path)

        if opt.data_label == "train":  # make sure the number of real and fake samples is balanced to avoid bias during training
            global_min = min(len(self.real_list), len(self.fake_list))
            max_samples = min(MAX_SAMPLES, global_min)

            self.real_list = self.real_list[:max_samples]
            self.fake_list = self.fake_list[:max_samples]

        self.label_dict = dict()
        for i in self.real_list:
            self.label_dict[i] = 0
        for i in self.fake_list:
            self.label_dict[i] = 1

        print("Using ", len(self.real_list), " real and ", len(
            self.fake_list), " fake images in ", opt.data_label, " set", flush=True)
        self.total_list = self.real_list + self.fake_list

    def __len__(self):
        return len(self.total_list)

    def __getitem__(self, idx):
        img_path = self.total_list[idx]
        label = self.label_dict[img_path]
        img = torch.tensor(cv2.imread(img_path), dtype=torch.float32)
        img = img.permute(2, 0, 1)
        crops = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])(img)
        # crop images
        # crops[0]: 1.0x, crops[1]: 0.65x, crops[2]: 0.45x
        crops = [[transforms.Resize((224, 224))(
            img[:, 500:, i:i + 500]) for i in range(5)], [], []]
        crop_idx = [(28, 196), (61, 163)]
        for i in range(len(crops[0])):
            crops[1].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[0][0]:crop_idx[0][1], crop_idx[0][0]:crop_idx[0][1]]))
            crops[2].append(transforms.Resize((224, 224))
                            (crops[0][i][:, crop_idx[1][0]:crop_idx[1][1], crop_idx[1][0]:crop_idx[1][1]]))
        img = transforms.Resize((1120, 1120))(img)

        return {"img": img, "crops": crops, "label": label, "img_path": img_path}


def create_dataloader(opt, train=True):
    dataset = LavDF(opt)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=train,
        num_workers=int(opt.num_threads),
    )
    return data_loader
