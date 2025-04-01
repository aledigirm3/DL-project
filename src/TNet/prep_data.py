import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ansi_colors import *
os.chdir(current_dir)

def build_tensor_dataset(dataset_folder: str, window_size: int):

    X_windows = []
    y_labels = []
    total_windows = 0

    for file in tqdm(os.listdir(dataset_folder), desc="Processing files", unit="file"):
        if file.endswith(".csv"):
            file_path = os.path.join(dataset_folder, file)

            df = pd.read_csv(file_path)
            num_features = (df.shape[1] - 3)  # Exclude video_id, frame, label

            # Sliding windows
            for i in range(0, len(df), window_size):  
                window = df.iloc[i:i+window_size, 2:-1].values.astype(np.float32)
                total_windows += 1

                if len(window) != 30:
                    print(len(window))
                    print(f"\n{RED}ERROR: window size != {window_size} {RESET}")
                    sys.exit(0)
                    
                    
                label = df.iloc[i, -1]
                X_windows.append(window)
                y_labels.append(label)

    print(f"\n{CYAN}Total windows: {total_windows}{RESET}")


    # To numpy array
    X_windows = np.array(X_windows)
    y_labels = np.array(y_labels)

    # Dataset shuffle
    indices = np.random.permutation(len(X_windows))  # Random permutation of indices
    X_windows = X_windows[indices]
    y_labels = y_labels[indices]


    # To PyTorch tensor
    X_tensor = torch.tensor(X_windows, dtype=torch.float32)  # Shape: (num_seq, 30, num_features)
    y_tensor = torch.tensor(y_labels, dtype=torch.long)  # Shape: (num_seq,)


    print("Shape X:", X_tensor.shape)  # (num_seq, 30, num_features)
    print("Shape y:", y_tensor.shape)  # (num_seq,)

    # Build PyTorch datasets (train/val/test)
    dataset = TensorDataset(X_tensor, y_tensor)

    return dataset