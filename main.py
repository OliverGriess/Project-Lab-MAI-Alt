import os

from dataloader.LAV_DF.utils import  get_list

if __name__ == "__main__":
    root_dir = "../../data/LAVDF-LIPFD/train"
    real_path = os.path.join(root_dir, "0_real")
    fake_path = os.path.join(root_dir, "1_fake")
    real = get_list(real_path)
    fake = get_list(fake_path)

    print(len(real))
    print(len(fake))


    with open("real_list_train.txt", "w") as f:
        for item in real:
            f.write(item + "\n")
    with open("fake_list_train.txt", "w") as f:
        for item in fake:
            f.write(item + "\n")


