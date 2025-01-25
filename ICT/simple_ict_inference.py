import os

import torchvision.transforms as transforms
# from ict_eval import face_learner
import argparse
import torch
from PIL import Image

from models import ict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for ICT DeepFake detection')

    parser.add_argument("-model", "--model_path", default='./checkpoints/ICT_Base.pth', type=str)

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert args.device.type == "cuda", "CUDA is not available"

    # args.model_path = os.path.join('./PRETRAIN/', args.model_path)
    model = ict.combface_base_patch8_112()
    dict = torch.load(args.model_path)
    model.load_state_dict(dict['model'])
    model.to(args.device)
    model.eval()


    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])


    img = Image.open("../deepfake_test.jpg").convert('RGB')
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(args.device)

    print(img.shape)

    with torch.no_grad():
        inner_emb, outer_emb = model(img)
        print("inner_emb: ", inner_emb.shape)
        print("outer_emb: ", outer_emb.shape)
        print("euclidean_distance: ", torch.norm(inner_emb - outer_emb, dim=1))

