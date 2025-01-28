import os
import json
from pathlib import Path
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import sqlite3

# Create a database and table
conn = sqlite3.connect("metadata.db")
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS metadata (
    window_start_idx INTEGER,
    window_end_idx INTEGER,
    fake_indices TEXT,
    window_path TEXT,
    video_path TEXT
)
""")
conn.commit()
WINDOW_LEN =5
# Insert metadata
def save_to_db(start_idx, fake_indices, out_path, video_path):
    cursor.execute("""
    INSERT INTO metadata (window_start_idx, window_end_idx, fake_indices, window_path, video_path)
    VALUES (?, ?, ?, ?, ?)
    """, (start_idx, start_idx + WINDOW_LEN - 1, str(fake_indices), out_path, video_path))
    conn.commit()
# Batch insert metadata
metadata_batch = []

def save_to_db_batch(start_idx, fake_indices, out_path, video_path):
    global metadata_batch
    metadata_batch.append((
        start_idx, 
        start_idx + WINDOW_LEN - 1, 
        str(fake_indices), 
        out_path, 
        str(video_path)
    ))
    # Insert in batches of 100 for efficiency
    if len(metadata_batch) >= 100:
        cursor.executemany("""
        INSERT INTO metadata (window_start_idx, window_end_idx, fake_indices, window_path, video_path)
        VALUES (?, ?, ?, ?, ?)
        """, metadata_batch)
        conn.commit()
        metadata_batch = []

def finalize_db_batch():
    global metadata_batch
    if metadata_batch:  # Insert remaining data
        cursor.executemany("""
        INSERT INTO metadata (window_start_idx, window_end_idx, fake_indices, window_path, video_path)
        VALUES (?, ?, ?, ?, ?)
        """, metadata_batch)
        conn.commit()
        metadata_batch = []


############################
# Custom Parameters
############################
WINDOW_LEN = 5        # Frames in each window
MAX_WORKERS = 16     # Number of threads
SAMPLE_RATE = 16000
OUTPUT_SIZE = (500, 500)
OUTPUT_ROOT = "/work/scratch/kurse/kurs00079/data/LAVDF-LIPFD"  # Root dir for saving
METADATA_JSON = "/work/scratch/kurse/kurs00079/data/LAV-DF/metadata.json"
metadata = []
############################
# Create output root if needed
os.makedirs(OUTPUT_ROOT, exist_ok=True)

import io
import soundfile as sf

def extract_audio_segment(video_path, start_t=None, end_t=None, sr=SAMPLE_RATE):
    """
    Extract audio samples from a video segment [start_t, end_t] using ffmpeg.
    Returns a 1D numpy array (float32) and sampling rate.
    If start_t/end_t are None, extracts the entire audio.
    """
    ffmpeg_exe = "/work/scratch/kurse/kurs00079/om43juhy/Project-Lab-MAI-Alt/ffmpeg-7.0.2-amd64-static/ffmpeg"

    # Use ffmpeg directly (assumes it's in PATH)
    cmd = [ffmpeg_exe, "-y"]
    
    # Add segment timing options if provided
    if start_t is not None:
        cmd += ["-ss", str(start_t)]
    if end_t is not None:
        cmd += ["-to", str(end_t)]
    
    cmd += [
        "-i", str(video_path),
        "-vn",                 # no video
        "-acodec", "pcm_s16le",
        "-ac", "1",            # mono
        "-ar", str(sr),        # set sampling rate
        "-f", "wav",
        "pipe:1"               # output to stdout
    ]

    try:
        # Run the ffmpeg command and capture the output
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        wav_data, err = p.communicate()
        if p.returncode != 0:
            raise RuntimeError(err.decode("utf-8"))
        if not wav_data:
            raise ValueError("No WAV data received.")

        # Wrap the WAV data in a BytesIO buffer
        buf = io.BytesIO(wav_data)
        audio, current_sr = sf.read(buf, dtype='float32')
        
        # Resample the audio if necessary
        if current_sr != sr:
            audio = librosa.resample(audio, orig_sr=current_sr, target_sr=sr)
            current_sr = sr

        return audio, current_sr
    except Exception as e:
        print(f"Audio extraction error: {video_path}, segment [{start_t},{end_t}]: {e}", flush=True)
        return np.array([], dtype=np.float32), sr



def get_mel_spectrogram(audio_data, sr):
    """
    Compute mel spectrogram (in dB) from 1D audio_data.
    Returns a 2D numpy array (float).
    """
    if len(audio_data) == 0:
        return None
    mel = librosa.feature.melspectrogram(y=audio_data, sr=sr)
    db_mel = librosa.power_to_db(mel, ref=np.min)
    return db_mel

def get_frames_in_segment(video_path, start_frame, end_frame):
    """
    Loads frames from start_frame to end_frame (inclusive) using a while-loop,
    converting BGR->RGBA, returning them in a list.
    Imitates the original code's approach.
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Could not open video: {video_path}", flush=True)
        return frames

    # Set the position to start_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_frame = start_frame
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frames.append(frame)
        current_frame += 1
    cap.release()
    return frames

def concat_fake_segments(video_path, fake_periods):
    """
    For a fake video, gather frames and audio from each fake period
    and concatenate them into a single "fully fake" sequence.
    Returns (frames_list, audio_data, sr, fps).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}", flush=True)
        return [], np.array([]), 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()

    combined_frames = []
    combined_audio = np.array([], dtype=np.float32)
    sr = SAMPLE_RATE  # we fix sample rate

    for (start_s, end_s) in fake_periods:
        start_frame = int(np.floor(start_s * fps))
        end_frame   = int(np.floor(end_s * fps))
        seg_frames = get_frames_in_segment(video_path, start_frame, end_frame)
        combined_frames.extend(seg_frames)

        seg_audio, _ = extract_audio_segment(video_path, start_t=start_s, end_t=end_s, sr=sr)
        if len(seg_audio) > 0:
            combined_audio = np.concatenate([combined_audio, seg_audio])

    return combined_frames, combined_audio, sr, fps

def get_all_frames(video_path):
    """
    For a real video, gather all frames using the original style of reading.
    Returns (frames_list, fps, frame_count).
    """
    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Cannot open video: {video_path}", flush=True)
        return frames, 0, 0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frames.append(frame)
    cap.release()
    return frames, fps, frame_count

def save_windows_as_images(frames, fake_frames, mel_spectrogram, output_dir, base_name, video_path):
    """
    Replicates the original approach:
      - computes windows of frames
      - extracts corresponding mel spectrogram slice
      - applies a colormap to the mel slice to produce a colored spectrogram
      - combines the colored spectrogram with video frames, and saves the final image.
    """
    if len(frames) < WINDOW_LEN:
        print(f"Not enough frames for {base_name}", flush=True)
        return

    # Convert the mel spectrogram to an 8-bit image (values from 0 to 255)
    if mel_spectrogram is None:
        print(f"No mel spectrogram for {base_name}", flush=True)
        return
    mel_min, mel_max = mel_spectrogram.min(), mel_spectrogram.max()
    mel_img = (255 * (mel_spectrogram - mel_min) / (mel_max - mel_min + 1e-8)).astype(np.uint8)
    time_bins = mel_img.shape[1]

    frame_count = len(frames)
    mapping = time_bins / frame_count  # mapping from frame index to mel time bin

    # Select N_EXTRACT starting indexes
    N_EXTRACT = frame_count - WINDOW_LEN + 1
    
    for i in range(0, N_EXTRACT):
        
        start_idx = i
        # Get the window of frames
        sub_frames = frames[start_idx:start_idx + WINDOW_LEN]
        fake_indices=[
                idx
                for idx in range(start_idx, start_idx + WINDOW_LEN)
                if any(fake_start <= idx <= fake_end for fake_start, fake_end in fake_frames)
            ]        # Determine the corresponding mel spectrogram slice
        begin_bin = int(np.round(start_idx * mapping))
        end_bin = int(np.round((start_idx + WINDOW_LEN) * mapping))
        if end_bin > time_bins:
            end_bin = time_bins
        sub_mel = mel_img[:, begin_bin:end_bin]

        # Resize sub_mel to match the frames width (OUTPUT_SIZE[0] * WINDOW_LEN, OUTPUT_SIZE[1])
        sub_mel_resized = cv2.resize(sub_mel, (OUTPUT_SIZE[0] * WINDOW_LEN, OUTPUT_SIZE[1]))
        # Instead of converting to RGBA directly, apply a colormap (e.g., VIRIDIS) for a colored effect.
        colored_sub_mel = cv2.applyColorMap(sub_mel_resized, cv2.COLORMAP_VIRIDIS)
        # Convert the colored image from BGR to RGBA
        colored_sub_mel = cv2.cvtColor(colored_sub_mel, cv2.COLOR_BGR2RGBA)

        # Resize the frames to the desired output size
        sub_frames_resized = [cv2.resize(f, OUTPUT_SIZE) for f in sub_frames]
        # Concatenate frames horizontally
        frames_concat = np.concatenate(sub_frames_resized, axis=1)

        # Combine the colored mel spectrogram above the frames
        combined = np.concatenate((colored_sub_mel, frames_concat), axis=0)
        group_id = i 
        out_path = os.path.join(output_dir, f"{base_name}_{group_id}.png")
        plt.imsave(out_path, combined)
        save_to_db_batch(start_idx, fake_indices, out_path, video_path)


def process_entry(entry):
    """
    Process a single metadata entry:
      - If real: read entire video for frames + entire audio
      - If fake: read only fake segments for frames + audio
      - Then produce windows + mel spectrogram
    """
    video_path = Path(f"/work/scratch/kurse/kurs00079/data/LAV-DF/{entry['file']}")
    if not video_path.exists():
        print(f"Video file not found: {video_path}", flush=True)
        return

    n_fakes = entry["n_fakes"]
    fake_periods = entry.get("fake_periods", [])
    split = entry["split"]  # e.g. train, val, test
    modify_video = entry["modify_video"]
    modify_audio = entry["modify_audio"]

    # Prepare top-level folder for this split
    split_dir = os.path.join(OUTPUT_ROOT, split)
    os.makedirs(split_dir, exist_ok=True)

    if n_fakes == 0:
        # Real video
        out_dir = os.path.join(split_dir, "0_real")
        os.makedirs(out_dir, exist_ok=True)
        frames, fps, frame_count = get_all_frames(video_path)
        audio_data, sr = extract_audio_segment(video_path, sr=SAMPLE_RATE)
    else:
        # Fake video: select subfolder based on modifications
        if modify_audio and not modify_video:
            subfolder = "audio"
        elif modify_video and not modify_audio:
            subfolder = "video"
        else:
            subfolder = "both"
        
        out_dir = os.path.join(split_dir, "1_fake", subfolder)
        os.makedirs(out_dir, exist_ok=True)
        frames, fps, frame_count = get_all_frames(video_path)
        audio_data, sr = extract_audio_segment(video_path, sr=SAMPLE_RATE)

    if len(frames) == 0 or len(audio_data) == 0:
        print(f"Skipping {video_path} due to insufficient frames or audio.", flush=True)
        return
    fake_frames = []
    for start_s, end_s in fake_periods:
        start_frame = int(np.floor(start_s * fps))
        end_frame = int(np.floor(end_s * fps))
        fake_frames.append((start_frame, end_frame))

    mel_spectrogram = get_mel_spectrogram(audio_data, sr)
    base_name = os.path.splitext(os.path.basename(str(video_path)))[0]
    save_windows_as_images(frames, fake_frames, mel_spectrogram, out_dir, base_name,video_path)

def main():
    with open(METADATA_JSON, 'r') as f:
        metadata = json.load(f)
    # For testing purposes, limit to first 5 entries
    futures = []
    print(len(metadata))

    CHUNK_SIZE = 6000  # Number of entries to process per chunk
    NEW_METAJSON = "/work/scratch/kurse/kurs00079/data/LAVDF-LIPFD/metadata.json"
    with open(NEW_METAJSON, 'w') as f:
        json.dump([], f, indent=4)  # Write an empty array to the file
    for i in range(132000, len(metadata), CHUNK_SIZE):
        chunk = metadata[i:i + CHUNK_SIZE]  # Extract a chunk of CHUNK_SIZE entries
        futures = []

        # Use ThreadPoolExecutor for processing the chunk
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for entry in chunk:
                futures.append(executor.submit(process_entry, entry))
            # Use tqdm to monitor progress
            for _ in tqdm(as_completed(futures), total=len(futures), desc=f"starting from  {i}", unit="file"):
                pass
        finalize_db_batch()

if __name__ == "__main__":
    main()
