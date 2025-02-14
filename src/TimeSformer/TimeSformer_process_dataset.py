import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class DanceDataset(Dataset):
    def __init__(self, video_paths, labels, seq_length=8, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.seq_length = seq_length
        self.transform = transform

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        frames = self.load_video(video_path)
        return frames, torch.tensor(label, dtype=torch.long)

    def load_video(self, path):
        cap = cv2.VideoCapture(path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, self.seq_length, dtype=int)
        frames = []
        
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (224, 224))
            if self.transform:
                frame = self.transform(frame)
            frames.append(frame)
        cap.release()
        return torch.stack(frames)  # (seq_length, 3, 224, 224)
    
    def count_sequences(self):
        total_sequences = 0
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            num_sequences = frame_count // self.seq_length
            total_sequences += num_sequences
        return total_sequences