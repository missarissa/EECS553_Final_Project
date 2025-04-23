import mne
import numpy as np
import torch
import torch.nn as nn
import torchaudio
from torch.utils.data import Dataset, DataLoader
from scipy import signal
import tensorly as tl
from tensorly.decomposition import parafac
from vit_pytorch import ViT
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
import os
import re
from tqdm import tqdm
import psutil

# Note: CHB-MIT dataset for patient chb21 must be manually downloaded from https://physionet.org/content/chbmit/1.0.0/chb21/
# Place the chb21 folder in './chbmit/chb21/' with .edf files and chb21-summary.txt

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
tl.set_backend('pytorch')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")

# Parameters
SAMPLE_RATE = 256  # Hz
WINDOW_SIZE = 30  # seconds
OVERLAP = 15  # seconds
N_FFT = 1024
HOP_LENGTH = 128
RANK = 15
IMG_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0005

# CHB-MIT dataset path
DATA_PATH = 'chb-mit-scalp-eeg-database-1.0.0'

class CHBMITDataset(Dataset):
    def __init__(self, patient_id, data_path, window_size=30, overlap=15, sample_rate=256):
        self.patient_id = patient_id
        self.data_path = data_path
        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.segments = []
        self.labels = []
        self.load_all_patient_data()

    # def _load_data(self):
    #     patient_dir = os.path.join(self.data_path, f'chb{self.patient_id:02d}')
    #     if not os.path.exists(patient_dir):
    #         raise FileNotFoundError(f"Patient directory {patient_dir} not found.")
    #     edf_files = ['chb21_19.edf', 'chb21_20.edf', 'chb21_21.edf', 'chb21_22.edf']
    #     edf_files = [f for f in edf_files if os.path.exists(os.path.join(patient_dir, f))]
    #     if not edf_files:
    #         raise FileNotFoundError(f"No valid .edf files found in {patient_dir}.")

    #     annot_file = os.path.join(patient_dir, f'chb{self.patient_id:02d}-summary.txt')
    #     seizure_times_dict = self._parse_seizure_times(annot_file)

    #     for edf_file in edf_files:
    #         file_path = os.path.join(patient_dir, edf_file)
    #         try:
    #             valid_channels = [
    #                 'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
    #                 'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
    #                 'FZ-CZ', 'CZ-PZ',
    #                 'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
    #                 'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
    #                 'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
    #             ]
    #             raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
    #             seen = set()
    #             channels_to_keep = []
    #             for ch in raw.ch_names:
    #                 if ch in valid_channels and ch not in seen:
    #                     channels_to_keep.append(ch)
    #                     seen.add(ch)
    #             raw.pick_channels(channels_to_keep)
    #         except Exception as e:
    #             print(f"Error reading {file_path}: {e}")
    #             continue

    #         raw.filter(l_freq=1, h_freq=50, method='fir', verbose=False)
    #         seizure_times = seizure_times_dict.get(edf_file, [])
    #         data = raw.get_data()
    #         n_samples = data.shape[1]
    #         window_samples = self.window_size * self.sample_rate
    #         step_samples = (self.window_size - self.overlap) * self.sample_rate

    #         for start in range(0, n_samples - window_samples + 1, step_samples):
    #             end = start + window_samples
    #             segment = data[:, start:end]
    #             segment = (segment - segment.mean()) / (segment.std() + 1e-10)
    #             segment_time = start / self.sample_rate
    #             label = self._label_segment(segment_time, seizure_times)
    #             if label is not None:
    #                 print(f"File {edf_file}, Label {label} at {segment_time:.0f}s, seizure times: {seizure_times}, from file {file_path}")
    #                 self.segments.append(segment)
    #                 self.labels.append(label)
    valid_channels = [
        'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
        'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
        'FZ-CZ', 'CZ-PZ',
        'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
        'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
        'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
    ]

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
    
    # def label_segments_for_patient(patient_id, data_path, valid_channels):
    def label_segments(segments, file_path, segment_length=30, stepsize=15, fs=256):
        
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
        concat_times = []
        current_time = 0
        for fname in sorted(all_data.keys()):
            d = all_data[fname]
            if concat_channels is None:
                concat_channels = d['channels']
            concat_data.append(d['data'])
            n_samples = d['data'].shape[1]
            concat_times.append((fname, current_time, current_time + n_samples / sample_rate))
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

        return segments, labels

    


    

# all_patient_data = load_all_patient_data(21, DATA_PATH, valid_channels)


    

    def _parse_seizure_times(self, annot_file):
        seizure_times_dict = {}
        if not os.path.exists(annot_file):
            print(f"Warning: Annotation file {annot_file} not found.")
            return seizure_times_dict

        current_file = None
        seizure_times = []
        total_seizures = 0

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
                        total_seizures += 1
                    except ValueError as e:
                        print(f"Error parsing seizure time in {annot_file}: {line} ({e})")
                        continue

        if current_file and seizure_times:
            seizure_times_dict[current_file] = seizure_times

        print(f"Found {total_seizures} seizure start times across {len(seizure_times_dict)} files in {annot_file}")
        return seizure_times_dict

    def _label_segment(self, segment_time, seizure_times):
        segment_end = segment_time + self.window_size
        for seizure_time in seizure_times:
            pre_ictal_start = seizure_time - 70 * 60
            pre_ictal_end = seizure_time - 10 * 60
            seizure_end = seizure_time + 60
            if (segment_time <= pre_ictal_end and segment_end >= pre_ictal_start):
                return 1
            if (segment_time <= seizure_end and segment_end >= seizure_time):
                return None
        return 0

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        return self.segments[idx], self.labels[idx]

def apply_tensor_decomposition(segments, rank=RANK):
    tf_tensors = []
    for segment in segments:
        n_channels, n_samples = segment.shape
        tf_segment = []
        for ch in range(n_channels):
            freqs, times, Zxx = signal.stft(segment[ch], fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT-HOP_LENGTH)
            tf_segment.append(np.abs(Zxx))
        tf_tensor = np.stack(tf_segment, axis=2)
        tf_tensors.append(tf_tensor)

    super_slices_list = []
    for tf_tensor in tf_tensors:
        tf_tensor = torch.tensor(tf_tensor, dtype=torch.float32).to(device)
        weights, factors = parafac(tf_tensor, rank=rank, init='random', tol=1e-6)
        A, B, C = factors
        C = C.to(device)
        P = torch.linalg.pinv(C.T @ C) @ C.T
        X_tilde = tl.tenalg.mode_dot(tf_tensor, P, mode=2)
        super_slices_list.append(X_tilde.cpu().numpy())

    return super_slices_list

class SpectrogramDataset(Dataset):
    def __init__(self, super_slices, labels, transform=None):
        self.super_slices = super_slices
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.super_slices)

    def __getitem__(self, idx):
        super_slice = self.super_slices[idx]
        label = self.labels[idx]
        super_slice = (super_slice - super_slice.min()) / (super_slice.max() - super_slice.min() + 1e-10)
        super_slice = torch.tensor(super_slice, dtype=torch.float32)
        super_slice = super_slice.permute(2, 0, 1)
        if super_slice.shape[0] < 3:
            super_slice = super_slice.repeat(3, 1, 1)[:3]
        elif super_slice.shape[0] > 3:
            super_slice = super_slice[:3]
        super_slice = torch.nn.functional.interpolate(super_slice.unsqueeze(0), size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False).squeeze(0)
        if self.transform:
            super_slice = self.transform(super_slice)
        return super_slice, label

def train_vit_model(train_loader, val_loader, num_classes=2):
    model = ViT(
        image_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        num_classes=num_classes,
        dim=512,
        depth=6,
        heads=8,
        mlp_dim=1024,
        dropout=0.1,
        emb_dropout=0.1
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        try:
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        except Exception as e:
            print(f"Error during training at epoch {epoch+1}: {e}")
            raise

        model.eval()
        val_preds, val_labels = [], []
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds, average='binary', zero_division=0)
        precision = precision_score(val_labels, val_preds, average='binary', zero_division=0)
        try:
            auc = roc_auc_score(val_labels, val_preds)
        except ValueError:
            auc = float('nan')

        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss/len(val_loader):.4f}, '
              f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, '
              f'Precision: {precision:.4f}, AUC: {auc:.4f}')

        scheduler.step()

    return model

def main():
    dataset = CHBMITDataset(patient_id=21, data_path=DATA_PATH, window_size=WINDOW_SIZE, overlap=OVERLAP, sample_rate=SAMPLE_RATE)
    segments, labels = [], []
    for segment, label in dataset:
        if segment is not None and label is not None:
            segments.append(segment)
            labels.append(label)
    print(f"Dataset: {len(segments)} segments, {sum(1 for l in labels if l == 1)} pre-ictal, {sum(1 for l in labels if l == 0)} interictal")
    print(f"Memory used: {psutil.virtual_memory().used / 1e9:.2f} GB")

    global class_weights
    pre_ictal_count = sum(1 for l in labels if l == 1)
    interictal_count = sum(1 for l in labels if l == 0)
    total = pre_ictal_count + interictal_count
    class_weights = torch.tensor([interictal_count/total, pre_ictal_count/total]).to(device)
    print(f"Class weights: {class_weights.tolist()}")

    super_slices = apply_tensor_decomposition(segments, rank=RANK)
    print(f"Super slices shape: {[s.shape for s in super_slices[:5]]}")
    print(f"Memory used: {psutil.virtual_memory().used / 1e9:.2f} GB")

    n_samples = len(super_slices)
    indices = np.random.permutation(n_samples)
    train_idx = indices[:int(4/6*n_samples)]
    val_idx = indices[int(4/6*n_samples):int(5/6*n_samples)]
    test_idx = indices[int(5/6*n_samples):]

    train_slices = [super_slices[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_slices = [super_slices[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_slices = [super_slices[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    train_dataset = SpectrogramDataset(train_slices, train_labels)
    val_dataset = SpectrogramDataset(val_slices, val_labels)
    test_dataset = SpectrogramDataset(test_slices, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    model = train_vit_model(train_loader, val_loader)

    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(test_labels, test_preds)
    recall = recall_score(test_labels, test_preds, average='binary', zero_division=0)
    precision = precision_score(test_labels, test_preds, average='binary', zero_division=0)
    try:
        auc = roc_auc_score(test_labels, test_preds)
    except ValueError:
        auc = float('nan')

    print(f'Test Results: Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, '
          f'Precision: {precision:.4f}, AUC: {auc:.4f}')

if __name__ == '__main__':
    main()