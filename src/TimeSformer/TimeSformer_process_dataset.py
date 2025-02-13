import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class DanceDataset(Dataset):
    def __init__(self, video_paths, labels, num_frames=8, transform=None):
        self.video_paths = video_paths
        self.labels = labels
        self.num_frames = num_frames
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
        frame_indices = np.linspace(0, total_frames-1, self.num_frames, dtype=int)
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
        return torch.stack(frames)  # (num_frames, 3, 224, 224)