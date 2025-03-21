import os
import cv2
import json
import numpy as np
import librosa
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from librosa import feature as audio
import uuid  # For generating unique file names

############ Custom parameter ##############
WINDOW_LEN = 5   # frames of each window

audio_root = "E://LAV-DF/LAV-DF/dev_wav"
video_root = "E://LAV-DF/LAV-DF/dev"
metadata_path = "E://LAV-DF/LAV-DF/metadata.json"
output_root = "./datasets/AVLips_dev"
MAX_THREADS = 16  # Number of threads for multithreading
############################################

def get_spectrogram(audio_file, output_path):
    data, sr = librosa.load(audio_file)
    mel = librosa.power_to_db(audio.melspectrogram(y=data, sr=sr), ref=np.min)
    plt.imsave(output_path, mel, cmap='viridis')

def process_directories():
    if not os.path.exists(output_root):
        os.makedirs(output_root, exist_ok=True)
    if not os.path.exists("./temp"):
        os.makedirs("./temp", exist_ok=True)

    output_real_dir = os.path.join(output_root, "0_real")
    output_fake_dir = os.path.join(output_root, "1_fake")

    if not os.path.exists(output_real_dir):
        os.makedirs(output_real_dir, exist_ok=True)
    if not os.path.exists(output_fake_dir):
        os.makedirs(output_fake_dir, exist_ok=True)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    metadata_dict = {entry["file"]: entry for entry in metadata}

    tasks = []
    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        for root, _, files in os.walk(audio_root):
            for file in tqdm(files, desc="Submitting Tasks"):
                video_path = os.path.join(video_root, file.replace(".wav", ".mp4"))
                audio_path = os.path.join(root, file)

                if not os.path.exists(audio_path):
                    print(f"Missing audio file for {file}")
                    continue

                metadata_entry = metadata_dict.get("dev/" + file.replace(".wav", ".mp4"), {})
                fake_periods = metadata_entry.get("fake_periods", [])

                tasks.append(executor.submit(
                    process_video_and_audio,
                    video_path, audio_path, output_real_dir, output_fake_dir, fake_periods, metadata_entry
                ))

        for task in tqdm(tasks, desc="Processing Tasks"):
            task.result()  # Wait for all tasks to complete

def process_video_and_audio(video_path, audio_path, output_real_dir, output_fake_dir, fake_periods, metadata_entry):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_sequence = list(range(frame_count))
    frame_list = []
    current_frame = 0

    while current_frame < frame_count:
        ret, frame = video_capture.read()
        if not ret:
            print(f"Error reading frame in {video_path} at {current_frame}")
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame_list.append(cv2.resize(frame, (500, 500)))
        current_frame += 1

    video_capture.release()

    # Generate a unique file name for the spectrogram
    temp_spectrogram_path = f"./temp/mel_{uuid.uuid4().hex}.png"
    get_spectrogram(audio_path, temp_spectrogram_path)
    mel = plt.imread(temp_spectrogram_path) * 255
    mel = mel.astype(np.uint8)

    # Clean up the temporary spectrogram file
    os.remove(temp_spectrogram_path)

    mapping = mel.shape[1] / frame_count
    group = 0
    for i in range(len(frame_list)):
        idx = i % WINDOW_LEN
        is_fake = False
        for period in fake_periods:
            start_frame = int(period[0] * fps)
            end_frame = int(period[1] * fps)
            if start_frame <= i < end_frame:
                is_fake = True
                break

        output_dir = output_fake_dir if is_fake else output_real_dir

        if idx == 0:
            try:
                begin = int(np.round(frame_sequence[i] * mapping))
                end = int(np.round((frame_sequence[i] + WINDOW_LEN) * mapping))
                sub_mel = cv2.resize(
                    mel[:, begin:end], (500 * WINDOW_LEN, 500)
                )
                x = np.concatenate(frame_list[i:i + WINDOW_LEN], axis=1)
                combined = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

                output_path = os.path.join(
                    output_dir, f"{os.path.basename(video_path).split('.')[0]}_{group}.png"
                )
                plt.imsave(output_path, combined)
                group += 1
            except ValueError as e:
                pass

if __name__ == "__main__":
    process_directories()
