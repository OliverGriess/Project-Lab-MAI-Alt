import os
import numpy as np

import torchvision.transforms as transforms
# from ict_eval import face_learner
import argparse
import torch
from PIL import Image

from models import ict
import matplotlib.pyplot as plt


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

    pic_list_real = os.listdir("DATASET/pics_real")
    pic_list_fake = os.listdir("DATASET/pics_fake")
    scores_real = []
    scores_fake = []
    for i in pic_list_real:
        img = Image.open(f"DATASET/pics_real/{i}").convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(args.device)

        with torch.no_grad():
            inner_emb, outer_emb = model(img)
            #print("inner_emb: ", inner_emb.shape)
            #print("outer_emb: ", outer_emb.shape)
            e_dis = torch.norm(inner_emb - outer_emb, dim=1).cpu().numpy()[0]
            #print("euclidean_distance: ", e_dis)
            scores_real.append(e_dis)

    for i in pic_list_fake:
        img = Image.open(f"DATASET/pics_fake/{i}").convert('RGB')
        img = transform(img)
        img = img.unsqueeze(0)
        img = img.to(args.device)

        with torch.no_grad():
            inner_emb, outer_emb = model(img)
            #print("inner_emb: ", inner_emb.shape)
            #print("outer_emb: ", outer_emb.shape)
            e_dis = torch.norm(inner_emb - outer_emb, dim=1).cpu().numpy()[0]
            #print("euclidean_distance: ", e_dis)
            scores_fake.append(e_dis)

    
    # Calculate metrics
    mean_real = np.mean(scores_real)
    std_dev_real = np.std(scores_real)
    mean_fake = np.mean(scores_fake)
    std_dev_fake = np.std(scores_fake)

    # Find the optimal threshold for classification
    all_scores = np.concatenate([scores_real, scores_fake])
    thresholds = np.linspace(min(all_scores), max(all_scores), 500)
    accuracy = []

    for threshold in thresholds:
        predictions = all_scores <= threshold
        true_labels = np.concatenate([np.ones_like(scores_real), np.zeros_like(scores_fake)])
        acc = np.mean(predictions == true_labels)
        accuracy.append(acc)

    optimal_threshold = thresholds[np.argmax(accuracy)]
    optimal_accuracy = max(accuracy)

    # Visualization
    plt.figure(figsize=(12, 6))

    # Histogram
    plt.hist(scores_real, bins=20, alpha=0.7, label="Real Pictures", color="blue", density=True)
    plt.hist(scores_fake, bins=20, alpha=0.7, label="Fake Pictures", color="red", density=True)

    # Add threshold line
    plt.axvline(optimal_threshold, color='green', linestyle='--', label=f"Optimal Threshold: {optimal_threshold:.2f}")

    # Labels and legend
    plt.title("Score Distribution for Real and Fake Pictures")
    plt.xlabel("Score")
    plt.ylabel("Density")
    plt.legend()

    # Display metrics
    print(f"Metrics for Real Pictures: Mean = {mean_real:.2f}, Std Dev = {std_dev_real:.2f}")
    print(f"Metrics for Fake Pictures: Mean = {mean_fake:.2f}, Std Dev = {std_dev_fake:.2f}")
    print(f"Optimal Threshold for Classification: {optimal_threshold:.2f}")
    print(f"Optimal Accuracy: {optimal_accuracy:.2f}")

    plt.show()
