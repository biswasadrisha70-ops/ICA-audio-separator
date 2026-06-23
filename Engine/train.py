import os
import numpy as np
import librosa

DATASET_DIR = "audio_source_separation/mixed_dataset/mixtures"   # adjust path

SR = 16000                # target sample rate
FRAME_LEN = 1024          # samples per frame
HOP_LEN = FRAME_LEN       # non-overlapping frames

examples = []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith(".wav"):
        continue

    path = os.path.join(DATASET_DIR, fname)

    y, _ = librosa.load(path, sr=SR, mono=True)

    n_frames = len(y) // FRAME_LEN
    y = y[: n_frames * FRAME_LEN]

    frames = y.reshape(n_frames, FRAME_LEN)

    examples.append(frames)

min_frames = min(x.shape[0] for x in examples)

X = np.stack(
    [x[:min_frames] for x in examples],
    axis=0
)

print(X.shape)
