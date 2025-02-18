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
import random

############ Custom parameter ##############
WINDOW_LEN = 5   # frames of each window

audio_root = "E://LAV-DF/LAV-DF/dev_wav"
video_root = "E://LAV-DF/LAV-DF/dev"
metadata_path = "E://LAV-DF/LAV-DF/metadata.json"
output_root = "E://Uni/AVLips"
MAX_THREADS = 1  # Number of threads for multithreading
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
                    video_path, audio_path, output_real_dir, output_fake_dir, fake_periods
                ))

        for task in tqdm(tasks, desc="Processing Tasks"):
            task.result()  # Wait for all tasks to complete

def process_video_and_audio(video_path, audio_path, output_real_dir, output_fake_dir, fake_periods):
    video_capture = cv2.VideoCapture(video_path)
    fps = video_capture.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

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
    fake_samples = []
    true_samples = []

    # Identify fake and true windows
    for i in range(0, frame_count - WINDOW_LEN + 1, WINDOW_LEN):
        fake_count = 0
        first_frame_fake = False
        for j in range(i, i + WINDOW_LEN):
            is_frame_fake = False
            for period in fake_periods:
                start_frame = int(period[0] * fps)
                end_frame = int(period[1] * fps)
                if start_frame <= j < end_frame:
                    is_frame_fake = True
                    break
            if is_frame_fake:
                fake_count += 1
            if j == i and is_frame_fake:
                first_frame_fake = True

        if fake_count > 0:
            fake_samples.append((i, fake_count, first_frame_fake))
        else:
            true_samples.append(i)

    # Balance true samples to match fake samples count
    true_samples = random.sample(true_samples, min(len(true_samples), len(fake_samples)))

    # Process fake samples
    process_samples(video_path, frame_list, mel, fake_samples, mapping, output_fake_dir, is_fake=True)

    # Process true samples
    process_samples(video_path, frame_list, mel, true_samples, mapping, output_real_dir, is_fake=False)

def process_samples(video_path, frame_list, mel, sample_indices, mapping, output_dir, is_fake=False):
    group = 0
    for item in sample_indices:

        # naming scheme for number of fake frames per video
        if is_fake:
            i, fake_count, first_frame_fake = item
            suffix = f"_{fake_count}" if not first_frame_fake else f"_-{fake_count}"
        else:
            i = item
            suffix = ""

        try:
            begin = int(np.round(i * mapping))
            end = int(np.round((i + WINDOW_LEN) * mapping))
            sub_mel = cv2.resize(mel[:, begin:end], (500 * WINDOW_LEN, 500))
            x = np.concatenate(frame_list[i:i + WINDOW_LEN], axis=1)
            combined = np.concatenate((sub_mel[:, :, :3], x[:, :, :3]), axis=0)

            output_path = os.path.join(output_dir, f"{os.path.basename(video_path).split('.')[0]}_{group}{suffix}.png")
            plt.imsave(output_path, combined)
            group += 1
        except ValueError:
            pass

if __name__ == "__main__":
    process_directories()
