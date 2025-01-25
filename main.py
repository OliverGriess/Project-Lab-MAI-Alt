import os

from dataloader.LAV_DF.utils import get_real_fake_list, get_real_fake_list_for_imgs

if __name__ == "__main__":
    root_dir = "../../data/LAV-DF/LipFD_preprocess"
    metadata_path = os.path.join(root_dir, "metadata.json")
    real_list, fake_list = get_real_fake_list(root_dir, metadata_path)
    real = get_real_fake_list_for_imgs(root_dir, real_list)
    fake = get_real_fake_list_for_imgs(root_dir, fake_list)

    print(len(real))
    print(len(fake))


    with open("real_list.txt", "w") as f:
        for item in real:
            f.write(item + "\n")
    with open("fake_list.txt", "w") as f:
        for item in fake:
            f.write(item + "\n")

