import mne
import numpy as np
import os
import re

# 30 second chunks

def load_file(path):
    raw = mne.io.read_raw_edf(path, preload=True)
    if(raw.info['sfreq'] != 256):
        raw.resample(256)
    return raw

def filter_eeg(raw, l_freq=0.5, h_freq=40.0):
    raw.filter(l_freq, h_freq, fir_design='firwin')
    return raw

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

# def label_segments(segments, raw, annotations, segment_length=30, fs=256):
#     n_samples = int(segment_length * fs)
#     n_segments = segments.shape[0]
#     labels = np.zeros(n_segments)

#     for ann in annotations:
#         if ann['description'] == 'Seizure':
#             start_time = ann['onset']
#             end_time = start_time + ann['duration']

#             start_segment = int(start_time / segment_length)
#             end_segment = int(end_time / segment_length) + 1

#             for i in range(start_segment, min(end_segment, n_segments)):
#                 labels[i] = 1
#     return labels


def label_segments(segments, file_path, segment_length=30, stepsize=15, fs=256):
    n_segments = segments.shape[0]
    labels = np.zeros(n_segments)
    
    # Extract the file name from the path
    file_name = os.path.basename(file_path)
    
    # Parse the summary file to get seizure information
    # You'll need to specify the path to your summary file
    summary_path = 'chb-mit-scalp-eeg-database-1.0.0\chb01\chb01-summary.txt'  # Update this
    
    with open(summary_path, 'r') as f:
        summary_text = f.read()
    
    # Find the section for this file
    file_pattern = f"File Name: {file_name}\n.*?Number of Seizures in File: (\d+)"
    file_match = re.search(file_pattern, summary_text, re.DOTALL)
    
    if file_match:
        num_seizures = int(file_match.group(1))
        
        if num_seizures > 0:
            # Extract seizure times
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
