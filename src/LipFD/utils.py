import os


def get_list(path: str) -> list:
    r"""Recursively read all files in root path"""
    image_list = list()
    for f in os.listdir(path):
        if f.split('.')[1] in ['png', 'jpg', 'jpeg']:
            img_path = os.path.join(path, f)
            # check if file size is >= 1KB (needed since some files are corrupted and thus size is 0)
            if os.path.getsize(img_path) >= 1024:
                image_list.append(img_path)
            else:
                print("EMPTY FILE: " + img_path, flush=True)
    return sorted(image_list)
