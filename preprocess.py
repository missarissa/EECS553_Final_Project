import mne
import numpy as np
import os
import re
import torch
import tensorly as tl
from tensorly.decomposition import parafac
from tensorly.tenalg import mode_dot

# 30 second chunks

def load_file(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    if(raw.info['sfreq'] != 256):
        raw.resample(256)
    return raw

def filter_eeg(raw, l_freq=0.5, h_freq=40.0):
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw

def append_all_data(files_arr, path):  
    all_data = []
    for file in files_arr:
        raw = load_file(file)
        raw = filter_eeg(raw)
        data = raw.get_data()
        all_data.append(data)
    return np.concatenate(all_data, axis=1)


def segment_eeg(raw, segment_length=30, stepsize = 15, fs=256):
    data = raw.get_data()
    n_samples = int(segment_length * fs)
    n_samples_step = int(stepsize * fs)
    n_samples_total = data.shape[1]

    segments = []

    start = 0
    while start + n_samples <= n_samples_total:
        end = start + n_samples
        segment = data[:, start:end]
        segments.append(segment)
        start += n_samples_step

    return np.array(segments)

def tensor_decomp(X):
    tl.set_backend('pytorch')
    factors = parafac(X, rank = 3)

    A, B, C = factors.factors

    X_approx = tl.kruskal_to_tensor(factors)

    P = torch.linalg.pinv(C)
    X_reduced = mode_dot(X, P, mode=2) 

    return X_reduced

def label_segments(segments, file_path, segment_length=30, stepsize=15, fs=256):
    n_segments = segments.shape[0]
    labels = np.zeros(n_segments)
    filesarr = os.listdir( 'chb01_01.edf', 'chb01_02.edf', 'chb01_01.edf', 'chb01_03.edf', 'chb01_04.edf', 'chb01_05.edf', 'chb01_06.edf', 'chb01_07.edf', 'chb01_08.edf', 'chb01_09.edf', 'chb01_10.edf', 'chb01_11.edf', 'chb01_12.edf', 'chb01_13.edf', 'chb01_14.edf', 'chb01_15.edf', 'chb01_16.edf', 'chb01_17.edf', 'chb01_18.edf')
    file_name = os.path.basename(file_path)
    all_data = append_all_data(filesarr, file_path)
    summary_path = 'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01-summary.txt'  # Update this
    
    with open(summary_path, 'r') as f:
        summary_text = f.read()
    

    file_pattern = f"File Name: {file_name}\n.*?Number of Seizures in File: (\d+)"
    file_match = re.search(file_pattern, summary_text, re.DOTALL)
    
    if file_match:
        num_seizures = int(file_match.group(1))
        
        if num_seizures > 0:
            seizure_section = re.search(f"File Name: {file_name}.*?(Seizure Start Time.*?)(?:File Name:|$)", 
                                      summary_text, re.DOTALL)
            
            if seizure_section:
                seizure_info = seizure_section.group(1)
                
                # Find all seizure start/end times
                start_times = re.findall(r"Seizure Start Time: (\d+) seconds", seizure_info)
                end_times = re.findall(r"Seizure End Time: (\d+) seconds", seizure_info)
                
                for start_time, end_time in zip(start_times, end_times):
                    start_sec = int(start_time)
                    end_sec = int(end_time)
                    
                    # Convert to segment indices (considering overlap)
                    start_segment = start_sec // stepsize
                    end_segment = (end_sec // stepsize) + 1
                    
                    # Mark segments as seizures
                    for i in range(start_segment, min(end_segment, n_segments)):
                        segment_start = i * stepsize
                        segment_end = segment_start + segment_length
                        
                        # Check if there's overlap with the seizure
                        if segment_end > start_sec and segment_start < end_sec:
                            labels[i] = 1
    
    return labels

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

def parse_seizure_times(annot_file):
    seizure_times_dict = {}
    if not os.path.exists(annot_file):
        print(f"Warning: Annotation file {annot_file} not found.")
        return seizure_times_dict

    current_file = None
    seizure_times = []
    with open(annot_file, 'r') as f:
        for line in f:
            line = line.strip()
            file_match = re.match(r"File Name: (chb\d+_\d+\.edf)", line)
            if file_match:
                if current_file and seizure_times:
                    seizure_times_dict[current_file] = seizure_times
                current_file = file_match.group(1)
                seizure_times = []
                continue
            start_match = re.match(r"Seizure \d+ Start Time: (\d+) seconds", line)
            if start_match:
                try:
                    start_time = int(start_match.group(1))
                    seizure_times.append(start_time)
                except ValueError:
                    continue
    if current_file and seizure_times:
        seizure_times_dict[current_file] = seizure_times
    return seizure_times_dict

def extract_labeled_segments(all_data, seizure_times_dict, sample_rate=256, window_size=30, step_size=15):
    """
    all_data: dict from load_all_patient_data
    seizure_times_dict: {filename: [seizure_start_seconds, ...]}
    Returns: segments, labels
    """
    segments = []
    labels = []

    window_samples = window_size * sample_rate
    step_samples = step_size * sample_rate

    # Gather all seizure times across all files for interictal logic
    all_seizure_times = []
    file_offsets = {}
    total_samples = 0
    for fname, d in all_data.items():
        file_offsets[fname] = total_samples / sample_rate
        total_samples += d['data'].shape[1]
        for t in seizure_times_dict.get(fname, []):
            all_seizure_times.append(file_offsets[fname] + t)
    all_seizure_times = sorted(all_seizure_times)

    # Find interictal periods (intervals between seizures >= 5 hours)
    interictal_periods = []
    for i in range(len(all_seizure_times) - 1):
        t1 = all_seizure_times[i]
        t2 = all_seizure_times[i + 1]
        if t2 - t1 >= 5 * 3600:
            mid = (t1 + t2) / 2
            interictal_periods.append((mid - 1800, mid + 1800))  # 1 hour centered in the gap

    # Go through all data in order
    concat_data = []
    concat_channels = None
    current_time = 0
    for fname in sorted(all_data.keys()):
        d = all_data[fname]
        if concat_channels is None:
            concat_channels = d['channels']
        concat_data.append(d['data'])
        n_samples = d['data'].shape[1]
        current_time += n_samples / sample_rate
    full_data = np.concatenate(concat_data, axis=1)

    n_total_samples = full_data.shape[1]
    for start in range(0, n_total_samples - window_samples + 1, step_samples):
        end = start + window_samples
        segment = full_data[:, start:end]
        segment_time = start / sample_rate

        # Label as preictal if within 70-10 min before any seizure
        label = 0
        for sz_time in all_seizure_times:
            preictal_start = sz_time - 70 * 60
            preictal_end = sz_time - 10 * 60
            if preictal_start <= segment_time < preictal_end:
                label = 1
                break

        # Label as interictal only if fully inside a valid interictal period
        if label == 0:
            for int_start, int_end in interictal_periods:
                if segment_time >= int_start and (segment_time + window_size) <= int_end:
                    label = 0
                    break
            else:
                # Not in interictal, not in preictal: skip
                continue

        segments.append(segment)
        labels.append(label)

    return np.array(segments), np.array(labels)

def get_patient_segments_and_labels(patient_id, data_path, valid_channels, window_size=30, step_size=15, sample_rate=256):
    all_data = load_all_patient_data(patient_id, data_path, valid_channels)
    annot_file = os.path.join(data_path, f'chb{patient_id:02d}', f'chb{patient_id:02d}-summary.txt')
    seizure_times_dict = parse_seizure_times(annot_file)
    segments, labels = extract_labeled_segments(all_data, seizure_times_dict, sample_rate, window_size, step_size)
    return segments, labels