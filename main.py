import numpy as np
from src.preprocess import load_file, filter_eeg, segment_eeg, label_segments
from src.tf_methods import generate_tf_rep, plot_tf_rep

# === Configuration ===
FILE_PATH = 'chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_03.edf'
SEGMENT_LENGTH = 30  # seconds
FS = 256
METHOD = 'SPEC'  # Options: 'GK', 'SPEC', etc.

SEGMENT_IDX = 200        # Which segment to plot
CHANNEL_IDX = 6         # Which channel in that segment
DURATION_SEC = 30     # How much of the segment to visualize (in seconds)

# === Main Process ===
if __name__ == "__main__":
    print(f"Loading file: {FILE_PATH}")
    raw = load_file(FILE_PATH)
    raw = filter_eeg(raw)
    #print("hi", raw.annotations)
    
    print("Segmenting signal...")
    segments = segment_eeg(raw, segment_length=SEGMENT_LENGTH, fs=FS)
    labels = label_segments(segments, FILE_PATH, segment_length=SEGMENT_LENGTH, stepsize=15, fs=FS)

    print(f"Total segments: {len(segments)}")
    print(f"Selected Segment: {SEGMENT_IDX}, Label: {'Seizure' if labels[SEGMENT_IDX] else 'Non-Seizure'}")

    # Extract the desired EEG channel and trim
    eeg_signal = segments[SEGMENT_IDX][CHANNEL_IDX]
    eeg_signal = eeg_signal[:DURATION_SEC * FS]

    print(f"Generating TF representation using method: {METHOD}")
    tfr, t, f = generate_tf_rep(eeg_signal, method=METHOD, fs=FS)

    title = f"{METHOD} TF Rep | Segment {SEGMENT_IDX} | Ch {CHANNEL_IDX} | {DURATION_SEC}s"
    plot_tf_rep(tfr, t, f, title=title)
