import mne
import numpy as np
import torch
import torch.nn as nn
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

# CHB-MIT dataset for chb21 needs to be downloaded from https://physionet.org/content/chbmit/1.0.0/chb21/
# Place the .edf and summary.txt files in './chbmit/chb21/'

torch.manual_seed(42)
np.random.seed(42)
tl.set_backend('pytorch')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}, RAM: {psutil.virtual_memory().total / 1e9:.2f} GB")

SAMPLE_RATE = 256
WINDOW_SIZE = 30
OVERLAP = 15
N_FFT = 1024
HOP_LENGTH = 128
RANK = 15
IMG_SIZE = 224
PATCH_SIZE = 16
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0003
WEIGHT_DECAY = 0.001
DATA_PATH = './chbmit/'

class CHBMITDataset(Dataset):
    def __init__(self, patient_id, data_path, window_size=30, overlap=15, sample_rate=256):
        self.patient_id = patient_id
        self.data_path = data_path
        self.window_size = window_size
        self.overlap = overlap
        self.sample_rate = sample_rate
        self.segments = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        patient_dir = os.path.join(self.data_path, f'chb{self.patient_id:02d}')
        if not os.path.exists(patient_dir):
            raise FileNotFoundError(f"Patient directory {patient_dir} not found.")
        edf_files = [f'chb21_{i:02d}.edf' for i in range(17, 24)] #Change range for the files you want to use
        edf_files = [f for f in edf_files if os.path.exists(os.path.join(patient_dir, f))]
        if not edf_files:
            raise FileNotFoundError(f"No valid .edf files found in {patient_dir}.")

        annot_file = os.path.join(patient_dir, f'chb{self.patient_id:02d}-summary.txt')
        seizure_times_dict = self._parse_seizure_times(annot_file)

        for edf_file in edf_files:
            file_path = os.path.join(patient_dir, edf_file)
            try:
                valid_channels = [
                    'FP1-F7', 'F7-T7', 'T7-P7', 'P7-O1',
                    'FP1-F3', 'F3-C3', 'C3-P3', 'P3-O1',
                    'FZ-CZ', 'CZ-PZ',
                    'FP2-F4', 'F4-C4', 'C4-P4', 'P4-O2',
                    'FP2-F8', 'F8-T8', 'T8-P8', 'P8-O2',
                    'P7-T7', 'T7-FT9', 'FT9-FT10', 'FT10-T8'
                ]
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
                print(f"Raw channels in {edf_file}: {raw.ch_names}")
                seen = set()
                channels_to_keep = []
                for ch in raw.ch_names:
                    base_ch = ch.strip()
                    if base_ch in valid_channels and base_ch not in seen:
                        channels_to_keep.append(ch)
                        seen.add(base_ch)
                print(f"Selected {len(channels_to_keep)} channels: {channels_to_keep}")
                if not channels_to_keep:
                    print(f"No valid channels found in {file_path}. Available channels: {raw.ch_names}")
                    continue
                raw.pick(channels_to_keep, verbose=False)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            raw.filter(l_freq=1, h_freq=50, method='fir', verbose=False)
            seizure_times = seizure_times_dict.get(edf_file, [])
            data = raw.get_data()
            n_samples = data.shape[1]
            window_samples = self.window_size * self.sample_rate
            step_samples = (self.window_size - self.overlap) * self.sample_rate

            for start in range(0, n_samples - window_samples + 1, step_samples):
                end = start + window_samples
                segment = data[:, start:end]
                segment = (segment - segment.mean()) / (segment.std() + 1e-10)
                segment_time = start / self.sample_rate
                label = self._label_segment(segment_time, seizure_times)
                if label is not None:
                    self.segments.append(segment)
                    self.labels.append(label)
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
            if pre_ictal_start <= segment_time < pre_ictal_end or \
               (segment_end > pre_ictal_start and segment_time < pre_ictal_end):
                return 1
            if segment_time < seizure_end and segment_end > seizure_time:
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
        dim=128,
        depth=2,
        heads=2,
        mlp_dim=256,
        dropout=0.2,
        emb_dropout=0.2
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_loss = float('inf')
    patience = 3
    counter = 0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

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

        val_loss = val_loss / len(val_loader)
        accuracy = accuracy_score(val_labels, val_preds)
        recall = recall_score(val_labels, val_preds, average='binary', zero_division=0)
        precision = precision_score(val_labels, val_preds, average='binary', zero_division=0)
        auc = roc_auc_score(val_labels, val_preds) if len(set(val_labels)) > 1 else float('nan')
        print(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}, '
              f'Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, '
              f'Precision: {precision:.4f}, AUC: {auc:.4f}')
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
        else:
            counter += 1
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
    if pre_ictal_count == 0 or interictal_count == 0:
        raise ValueError("No valid segments found for one or both classes. Check data loading and labeling.")
    total = pre_ictal_count + interictal_count
    class_weights = torch.tensor([total/(2*interictal_count), total/(2*pre_ictal_count)]).to(device)
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
    auc = roc_auc_score(test_labels, test_preds) if len(set(test_labels)) > 1 else float('nan')
    print(f'Test Results: Accuracy: {accuracy:.4f}, Recall: {recall:.4f}, '
           f'Precision: {precision:.4f}, AUC: {auc:.4f}')

if __name__ == '__main__':
    main()
