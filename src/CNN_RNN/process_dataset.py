import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class DanceDataset(Dataset):
    def __init__(self, video_paths, labels, transform=None, seq_length=30):
        self.video_paths = video_paths
        self.labels = labels  # Labels (0 = Salsa, 1 = Bachata)
        self.transform = transform
        self.seq_length = seq_length  # Frames per sequence

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx] 
        label = self.labels[idx]

        cap = cv2.VideoCapture(video_path)

        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Frame count

        frame_indices = np.linspace(0, frame_count - 1, self.seq_length, dtype=int)
        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()

            if not ret:
                break 
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # To RGB
            frame = cv2.resize(frame, (224, 224))  # For CNN dimension

            if self.transform:
                frame = self.transform(frame)

            frames.append(frame)

        cap.release()

        # Padding if tot_frames < eq_length
        while len(frames) < self.seq_length:
            frames.append(torch.zeros(3, 224, 224))  # Black frame (padding)

        frames = torch.stack(frames)  # Frames list to tensore
        return frames, torch.tensor(label, dtype=torch.long)  # Return frames sequence and labels
    

    def count_sequences(self):
        total_sequences = 0
        for video_path in self.video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            num_sequences = frame_count // self.seq_length
            total_sequences += num_sequences
        return total_sequences
