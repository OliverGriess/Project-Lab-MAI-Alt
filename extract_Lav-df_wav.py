import subprocess
from concurrent.futures import ThreadPoolExecutor
import os
import json
MAX_THREADS = 16  # Number of threads for multithreading


def extract_wav(video_path, wav_path):
    ffmpeg_exe = "/work/scratch/kurse/kurs00079/ng33rete/new_approach/ffmpeg-7.0.2-amd64-static/ffmpeg"
    command = f"{ffmpeg_exe} -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {wav_path}"
    subprocess.run(command, shell=True)

def extract(root_dir, wav_root):
    os.makedirs(wav_root, exist_ok=True)

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    wav_path = os.path.join(wav_root, file.replace(".mp4", ".wav"))
                    executor.submit(extract_wav, video_path, wav_path)

if __name__ == "__main__":
    LAV_DF_ROOT = "/work/scratch/kurse/kurs00079/data/LAV-DF/train"
    wav_root = "/work/scratch/kurse/kurs00079/data/LAV-DF-WAV/train"

    metadata_path = "/work/scratch/kurse/kurs00079/data/LAV-DF/metadata.json"
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    fake_list = []

    for item in metadata:
        is_fake = item["n_fakes"] > 0
        if not is_fake: continue

        is_train = item["file"].startswith("train")
        if not is_train: continue

        file = item["file"].split("/")[-1].replace(".wav", ".mp4")
    
        fake_list.append(file)

    with open("fake_list_train.txt", "w") as f:
        for item in fake_list:
            f.write(item + "\n")

    
    with open("fake_list_train.txt", "r") as f:
        fakes = f.readlines()

    print(fakes[0])



        # video_path = os.path.join(LAV_DF_ROOT, item["video_path"])
        # wav_path = os.path.join(wav_root, item["video_path"].replace(".mp4", ".wav"))
        # extract_wav(video_path, wav_path)