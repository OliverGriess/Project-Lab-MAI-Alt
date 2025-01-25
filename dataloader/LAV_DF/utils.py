import json
import os


def read_json(path: str, object_hook=None):
    with open(path, "r") as f:
        return json.load(f, object_hook=object_hook)


def get_real_fake_list_for_imgs(root_dir: str, path_list):
    res_list = []
    for file_path in path_list:
        file_name = os.path.basename(file_path)
        file_id = file_name.split(".")[0]
        for i in range(10):
            if os.path.exists(os.path.join(root_dir, f"{file_id}_{i}.png")):
                res_list.append(os.path.join(root_dir, f"{file_id}_{i}.png"))
            else:
                break
    return res_list


def get_real_fake_list(root_dir: str, metadata_path: str) -> tuple[list[dict], list[dict]]:
    metadata = read_json(metadata_path)
    real_list = [os.path.join(root_dir, item["file"])
                 for item in metadata if not item["modify_audio"] and not item["modify_video"]]
    fake_list = [os.path.join(root_dir, item["file"])
                 for item in metadata if item["modify_audio"] or item["modify_video"]]

    return real_list, fake_list
