import os
import json
import argparse
from tqdm import tqdm
import cv2
import numpy as np
import tensorflow as tf
from deepfake_detector import live_video_prediction, image_prediction
from deepfake_detector.face_detection import model  # Import the model for finetuning
from deepfake_detector.face_detection import preprocess_frame

def process_lavdf_videos(data_dir, output_dir=None, threshold=0.5, max_samples=None):
    """
    Process videos from the LAVDF dataset using deepfake_detector.
    
    Args:
        data_dir (str): Path to the LAVDF dataset
        output_dir (str): Directory to save results (optional)
        threshold (float): Threshold for classifying as fake
        max_samples (int): Maximum number of samples to process
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Filter for test split videos
    test_samples = [item for item in metadata if item.get('split') == 'test']
    
    # Limit samples if specified
    if max_samples is not None and max_samples < len(test_samples):
        test_samples = test_samples[:max_samples]
    
    print(f"Processing {len(test_samples)} videos from LAVDF dataset...")
    
    # Process each video
    results = []
    for i, sample in enumerate(tqdm(test_samples)):
        video_path = os.path.join(data_dir, sample['file'])
        
        # Get ground truth (1 for fake, 0 for real)
        is_fake = 1 if sample.get('fake_periods', []) else 0
        
        print(f"\nProcessing video {i+1}/{len(test_samples)}: {sample['file']}")
        print(f"Ground truth: {'FAKE' if is_fake else 'REAL'}")
        
        # Process with deepfake_detector
        try:
            # Use live_video_prediction but don't display the video (just process it)
            # We'll modify the function to return results instead of showing the video
            result = process_video_without_display(video_path, threshold)
            
            # Store results
            results.append({
                'file': sample['file'],
                'ground_truth': is_fake,
                'prediction': result['is_fake'],
                'score': result['score'],
                'is_correct': result['is_fake'] == is_fake
            })
            
            print(f"Prediction: {'FAKE' if result['is_fake'] else 'REAL'} (score: {result['score']:.4f})")
            print(f"Correct: {'✓' if result['is_fake'] == is_fake else '✗'}")
            
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    # Calculate overall accuracy
    if results:
        correct_count = sum(1 for r in results if r['is_correct'])
        accuracy = correct_count / len(results)
        
        print(f"\nProcessed {len(results)} videos")
        print(f"Overall accuracy: {accuracy:.2%} ({correct_count}/{len(results)})")
        
        # Save results if output directory is specified
        if output_dir:
            save_results(results, accuracy, output_dir)
    else:
        print("No videos were processed successfully.")

def process_video_without_display(video_path, threshold=0.5, window_size=5):
    """
    Process a video file with deepfake_detector using sliding windows.
    
    Args:
        video_path (str): Path to the video file
        threshold (float): Threshold for classifying as fake
        window_size (int): Size of sliding window
        
    Returns:
        dict: Dictionary with prediction results
    """
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(video_path)
    
    # Try to load sequence model, fall back to frame-by-frame if not available
    try:
        sequence_model = tf.keras.models.load_model('best_sequence_model.h5')
        use_sequence_model = True
        face_buffer = []
        window_predictions = []
    except:
        use_sequence_model = False
        frame_predictions = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # Get the largest face
            x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, (224, 224))
            face = face.astype('float32') / 255.0
            
            if use_sequence_model:
                face_buffer.append(face)
                if len(face_buffer) == window_size:
                    window = np.array([face_buffer])
                    pred = sequence_model.predict(window)[0][0]
                    window_predictions.append(pred)
                    face_buffer.pop(0)
            else:
                try:
                    from deepfake_detector.face_detection import predict_face
                    prediction = predict_face(face)
                    frame_predictions.append(prediction)
                except Exception as e:
                    print(f"Error predicting face: {e}")
    
    cap.release()
    
    # Calculate final prediction
    if use_sequence_model:
        if window_predictions:
            avg_prediction = max(window_predictions)  # Use max since any fake window makes video fake
        else:
            avg_prediction = 0.5
    else:
        if frame_predictions:
            avg_prediction = sum(frame_predictions) / len(frame_predictions)
        else:
            avg_prediction = 0.5
    
    return {
        'score': avg_prediction,
        'is_fake': avg_prediction > threshold
    }

def save_results(results, accuracy, output_dir):
    """
    Save detection results to a file.
    
    Args:
        results (list): List of result dictionaries
        accuracy (float): Overall accuracy
        output_dir (str): Directory to save results
    """
    with open(os.path.join(output_dir, 'detection_results.txt'), 'w') as f:
        f.write("LAVDF Deepfake Detection Results\n")
        f.write("===============================\n\n")
        f.write(f"Videos processed: {len(results)}\n")
        f.write(f"Overall accuracy: {accuracy:.2%} ({sum(1 for r in results if r['is_correct'])}/{len(results)})\n\n")
        
        # Calculate metrics by class
        real_samples = [r for r in results if r['ground_truth'] == 0]
        fake_samples = [r for r in results if r['ground_truth'] == 1]
        
        real_correct = sum(1 for r in real_samples if r['is_correct'])
        fake_correct = sum(1 for r in fake_samples if r['is_correct'])
        
        if real_samples:
            f.write(f"Real videos accuracy: {real_correct/len(real_samples):.2%} ({real_correct}/{len(real_samples)})\n")
        if fake_samples:
            f.write(f"Fake videos accuracy: {fake_correct/len(fake_samples):.2%} ({fake_correct}/{len(fake_samples)})\n")
        
        f.write("\nVideo Details:\n")
        for i, r in enumerate(results):
            f.write(f"{i+1}. {r['file']}\n")
            f.write(f"   Ground truth: {'FAKE' if r['ground_truth'] else 'REAL'}\n")
            f.write(f"   Prediction: {'FAKE' if r['prediction'] else 'REAL'} (score: {r['score']:.4f})\n")
            f.write(f"   Correct: {'Yes' if r['is_correct'] else 'No'}\n\n")
    
    print(f"Results saved to {os.path.join(output_dir, 'detection_results.txt')}")

class LAVDFSequenceGenerator(tf.keras.utils.Sequence):
    """
    Data generator for LAVDF dataset that loads and processes videos on-the-fly.
    Similar to PyTorch's DataLoader, this generator loads data in batches during training
    instead of loading everything into memory at once.
    """
    def __init__(self, metadata, data_dir, batch_size=32, window_size=5, shuffle=True):
        self.metadata = metadata
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.window_size = window_size
        self.shuffle = shuffle
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create list of all possible windows from all videos
        self.windows = []
        
        print("Scanning videos to identify all possible windows...")
        for sample in tqdm(self.metadata):
            video_path = os.path.join(self.data_dir, sample['file'])
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                
                # Skip videos with too few frames
                if total_frames < self.window_size:
                    continue
                
                # Get fake periods
                fake_periods = sample.get('fake_periods', [])
                
                # Add windows with metadata
                for start_frame in range(0, total_frames - self.window_size + 1, 2):  # Step by 2 to reduce data
                    # Check if any frame in window is fake
                    window_is_fake = False
                    for frame_idx in range(start_frame, start_frame + self.window_size):
                        current_time = frame_idx / fps
                        if any(start <= current_time <= end for start, end in fake_periods):
                            window_is_fake = True
                            break
                    
                    self.windows.append({
                        'video_path': video_path,
                        'start_frame': start_frame,
                        'end_frame': start_frame + self.window_size,
                        'is_fake': window_is_fake
                    })
            except Exception as e:
                print(f"Error scanning {video_path}: {e}")
        
        print(f"Found {len(self.windows)} valid windows across {len(self.metadata)} videos")
        self.on_epoch_end()
    
    def __len__(self):
        """Return the number of batches per epoch"""
        return int(np.ceil(len(self.windows) / self.batch_size))
    
    def __getitem__(self, index):
        """Get a batch of data"""
        # Get batch indexes
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_windows = [self.windows[i] for i in batch_indexes]
        
        # Initialize batch arrays
        batch_x = []  # Face windows
        batch_y = []  # Labels
        
        # Process each window in batch
        for window_info in batch_windows:
            try:
                # Extract faces from window
                window_faces = self._extract_faces_from_window(window_info)
                
                # If we got all faces for this window, add it to batch
                if window_faces is not None and len(window_faces) == self.window_size:
                    batch_x.append(window_faces)
                    batch_y.append(1 if window_info['is_fake'] else 0)
            except Exception as e:
                print(f"Error processing window from {window_info['video_path']}: {e}")
        
        if not batch_x:  # If batch is empty, return a small dummy batch
            return np.zeros((1, self.window_size, 224, 224, 3)), np.zeros((1,))
        
        return np.array(batch_x), np.array(batch_y)
    
    def _extract_faces_from_window(self, window_info):
        """Extract faces from frames in a window"""
        cap = cv2.VideoCapture(window_info['video_path'])
        cap.set(cv2.CAP_PROP_POS_FRAMES, window_info['start_frame'])
        
        window_faces = []
        
        # Read window_size frames
        for _ in range(self.window_size):
            ret, frame = cap.read()
            if not ret:
                cap.release()
                return None
            
            # Detect and extract faces
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            if len(faces) > 0:
                # Get the largest face
                x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])
                face = frame[y:y+h, x:x+w]
                
                # Preprocess face
                face = preprocess_frame(face)
                window_faces.append(face[0])  # Remove batch dimension
            else:
                cap.release()
                return None  # Skip windows where we couldn't detect a face in every frame
        
        cap.release()
        return np.array(window_faces)
    
    def on_epoch_end(self):
        """Shuffle indexes after each epoch if shuffle is set to True"""
        self.indexes = np.arange(len(self.windows))
        if self.shuffle:
            np.random.shuffle(self.indexes)

def finetune_model(data_dir, epochs=5, batch_size=32, learning_rate=1e-4, window_size=5):
    """
    Finetune the existing deepfake detector model on LAVDF dataset using sliding windows.
    Uses memory-efficient batch processing instead of loading entire dataset at once.
    
    Args:
        data_dir (str): Path to the LAVDF dataset
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        learning_rate (float): Learning rate for training
        window_size (int): Size of sliding window for temporal context
    """
    print("Loading LAVDF dataset for finetuning...")
    
    # Load metadata
    metadata_path = os.path.join(data_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Use test split videos for quick testing (limiting to 100 videos for faster training)
    test_samples = [item for item in metadata if item.get('split') == 'test'][:100]
    
    print(f"Using {len(test_samples)} videos for training")
    
    # Create data generator
    train_generator = LAVDFSequenceGenerator(
        metadata=test_samples,
        data_dir=data_dir,
        batch_size=batch_size, 
        window_size=window_size,
        shuffle=True
    )
    
    # Create a sequence model that wraps the original model
    sequence_input = tf.keras.Input(shape=(window_size, 224, 224, 3))
    
    # Process each frame through the base model
    frame_features = []
    for i in range(window_size):
        frame_input = tf.keras.layers.Lambda(lambda x: x[:, i])(sequence_input)
        frame_output = model(frame_input)
        frame_features.append(frame_output)
    
    # Combine frame predictions
    combined_features = tf.keras.layers.Average()(frame_features)
    
    sequence_model = tf.keras.Model(sequence_input, combined_features)
    
    # Freeze early layers of the base model
    for layer in model.layers[:-2]:
        layer.trainable = False
    
    # Compile sequence model
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    sequence_model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    # Create model name based on window size
    model_base_name = f'finetuned_sequence_model_w{window_size}'
    
    # Create callback to save model after each epoch
    class SaveModelPerEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            epoch_model_name = f'{model_base_name}_e{epoch+1}.h5'
            self.model.save(epoch_model_name)
            print(f"\nSaved model for epoch {epoch+1} as {epoch_model_name}")
    
    # Quick test finetuning
    print("\nStarting sequence model finetuning...")
    history = sequence_model.fit(
        train_generator,
        epochs=epochs,
        verbose=1,
        callbacks=[SaveModelPerEpoch()]
    )
    
    # Save final model
    final_model_name = f'{model_base_name}_final.h5'
    sequence_model.save(final_model_name)
    print("\nFinetuning completed!")
    print(f"Final model saved as '{final_model_name}'")
    
    return history, sequence_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process LAVDF videos with deepfake_detector')
    parser.add_argument('--data_dir', type=str, default=r"D:\extra_model\audio-visual-forensics\LAV-DF", help='Path to the LAVDF dataset')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    parser.add_argument('--threshold', type=float, default=0.5, help='Threshold for classifying as fake')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to process')
    parser.add_argument('--finetune', action='store_true', default=False, help='Finetune the model on LAVDF dataset')
    parser.add_argument('--epochs', type=int, default=150, help='Number of epochs for finetuning')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for finetuning')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for finetuning')
    parser.add_argument('--window_size', type=int, default=5, help='Size of sliding window for temporal context')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    if args.finetune:
        # Finetune the model
        history, sequence_model = finetune_model(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            window_size=args.window_size
        )
    
    # Process LAVDF videos
    process_lavdf_videos(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_samples=args.max_samples
    )

if __name__ == '__main__':
    main() 