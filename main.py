import numpy as np
import re
import mne
import os       
from src.preprocess import load_file, filter_eeg, segment_eeg, label_segments
from src.tf_methods import generate_tf_rep, plot_tf_rep

# === Configuration ===
FILE_PATH = 'chb-mit-scalp-eeg-database-1.0.0/chb01/chb01_03.edf' #download from kaggle
SEGMENT_LENGTH = 30  # seconds
FS = 256
METHOD = 'SPEC'  # Options: 'GK', 'SPEC', etc.

SEGMENT_IDX = 200        # Which segment to plot
CHANNEL_IDX = 6         # Which channel in that segment
DURATION_SEC = 30     # How much of the segment to visualize (in seconds)




def get_start(file_name, summary_text):
    file_pattern = f"File Name: {file_name}\n.*?Number of Seizures in File: (\\d+)"
    file_match = re.search(file_pattern, summary_text, re.DOTALL)

    if file_match:
        num_seizures = int(file_match.group(1))
        if num_seizures > 0:
            seizure_section = re.search(
                f"File Name: {file_name}.*?(Seizure Start Time.*?)(?:File Name:|$)",
                summary_text, re.DOTALL
            )
            if seizure_section:
                seizure_info = seizure_section.group(1)
                # Find all seizure start/end times
                start_times = re.findall(r"Seizure Start Time: (\d+) seconds", seizure_info)
                end_times = re.findall(r"Seizure End Time: (\d+) seconds", seizure_info)
                return start_times, end_times
    return [], []


def load_all_patient_data(patient_id, data_path, valid_channels):
        """
        Loads all EEG data for a patient into a dictionary.
        Returns: {filename: {'channels': [ch_names], 'data': np.ndarray (channels x samples)}}
        """
        patient_dir = os.path.join(data_path, f'chb{patient_id:02d}')
        if not os.path.exists(patient_dir):
            raise FileNotFoundError(f"Patient directory {patient_dir} not found.")

        edf_files = [f for f in os.listdir(patient_dir) if f.endswith('.edf')]
        all_data = {}

        for edf_file in edf_files:
            file_path = os.path.join(patient_dir, edf_file)
            try:
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                # Keep only valid channels that exist in this file
                channels_to_keep = [ch for ch in raw.ch_names if ch in valid_channels]
                raw.pick_channels(channels_to_keep)
                data = raw.get_data()
                all_data[edf_file] = {
                    'channels': raw.ch_names,
                    'data': data
                }
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

        return all_data


def create_1_segments(all_data, start_time, end_time, segment_length=30, stepsize=15, fs=256):
    """
    Segments all data into chunks of specified length and step size.
    Returns: {filename: [segments]}
    """
    segments = []
    start_idx = int(start_time - 70 * 60 * fs)
    end_idx = int(start_time - 10 * 60 * fs)
    hour = all_data[:, start_idx:end_idx];
    n_segments = int((hour.shape[1] - segment_length * fs) // (stepsize * fs) + 1)

    for i in range (0, len(hour)/stepsize):
        start = i * stepsize * fs
        end = start + segment_length * fs
        temp = hour[:, start:end]
        segment = (1, temp) 
        segments.append(segment)
    return segments



arrr = [
    "FP1-F7", "F7-T7", "T7-P7", "P7-O1",
    "FP1-F3", "F3-C3", "C3-P3", "P3-O1",
    "FP2-F4", "F4-C4", "C4-P4", "P4-O2",
    "FP2-F8", "F8-T8", "T8-P8", "P8-O2",
    "FZ-CZ", "CZ-PZ", "P7-T7", "T7-FT9",
    "FT9-FT10", "FT10-T8", "T8-P8"
]




    


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
    
    load_all_patient_data(1, 'chb-mit-scalp-eeg-database-1.0.0/chb01', arrr)



    
