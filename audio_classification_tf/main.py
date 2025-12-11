import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_io as tfio
import sys
from pathlib import Path
import csv
import scipy.signal

YAMNET_MODEL = "https://tfhub.dev/google/yamnet/1"

def load_class_names(class_map_path):
    class_names = []
    with tf.io.gfile.GFile(class_map_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            class_names.append(row["display_name"])
    return class_names

def ensure_sample_rate(waveform, original_sr, desired_sr=16000):
    if original_sr != desired_sr:
        desired_length = int(round(len(waveform) * float(desired_sr) / original_sr))
        waveform = scipy.signal.resample(waveform, desired_length)
    return waveform

def load_audio_mono(wav_path):
    file_contents = tf.io.read_file(wav_path)
    audio, sr = tf.audio.decode_wav(file_contents, desired_channels=1)
    waveform = audio[:, 0].numpy()
    return sr.numpy(), waveform.astype(np.float32)

def main():
    if len(sys.argv) < 2:
        print("Usage: python main.py <audio_file.wav>")
        return

    audio_path = sys.argv[1]

    if not Path(audio_path).exists():
        print(f"Error: file not found: {audio_path}")
        return

    print("Loading YAMNet model...")
    model = hub.load(YAMNET_MODEL)

    class_map_path = model.class_map_path().numpy()
    class_names = load_class_names(class_map_path)

    print(f"Loading audio: {audio_path}")
    sr, waveform = load_audio_mono(audio_path)

    waveform = ensure_sample_rate(waveform, sr)

    # Run model
    scores, embeddings, spectrogram = model(waveform)
    mean_scores = tf.reduce_mean(scores, axis=0)

    top_n = 5
    top_indices = tf.argsort(mean_scores, direction="DESCENDING")[:top_n].numpy()

    print("\nTop predictions:")
    for idx in top_indices:
        print(f"{class_names[idx]}: {mean_scores[idx].numpy():.4f}")

if __name__ == "__main__":
    main()
