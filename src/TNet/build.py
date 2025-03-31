import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
import sys
import os
from tqdm import tqdm
import torch.nn as nn
from collections import Counter

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_eval.training import train
from train_eval.evaluation import evaluate
from ansi_colors import *
os.chdir(current_dir)

timesnet_path = os.getenv('TIMESNET_PATH') # TO FIX

timesnet_path = 'C:/Users/aless/dev/package/Time-Series-Library'

if timesnet_path:
    sys.path.append(timesnet_path)

else:
    print(f"\n{RED}(user) TimesNet path not found{RESET}\n")
    sys.exit(0)
from models.TimesNet import Model # type: ignore

class Config:
    def __init__(self, **kwargs):
        # Imposta ogni parametro come attributo della classe
        for key, value in kwargs.items():
            setattr(self, key, value)



# ====================================================================================================================

def count_labels(dataset):
    labels = [label.item() for _, label in dataset]
    return Counter(labels)
# ===========================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{CYAN}{device}{RESET}")
print(f"Cuda version: {torch.version.cuda}")
print(f"Torch version: {torch.__version__}\n")
# ===========================================================================================


dataset_folder = './processedDataset/'

window_size = 30
bs = 16
num_classes = 2
learning_rate = 0.001
num_epochs = 40
patience = 5 # Early stopping

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

total_size = len(dataset)
train_size = int(0.7 * total_size)
val_size = int(0.15 * total_size)
test_size = total_size - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_counts = count_labels(train_dataset)
val_counts = count_labels(val_dataset)
test_counts = count_labels(test_dataset)

print("\n")
print(f"- {CYAN}Train Labels:{RESET} {train_counts}")
print(f"- {CYAN}Validation Labels:{RESET} {val_counts}")
print(f"- {CYAN}Test Labels:{RESET} {test_counts}")


train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)


# Check one batch
print("\n")
print("Batch check:")
for batch_X, batch_y in train_dataloader:
    print("Batch X Shape:", batch_X.shape)  # (batch_size, seq_len, num_features)
    print("Batch y Shape:", batch_y.shape)  # (batch_size,)
    break

print("\n")

configs_dict = {
    'task_name': 'classification',
    'seq_len': window_size,            # input seq.
    'label_len': 30,                   # 
    'pred_len': 0,                     # There are no predictions (classification)
    'e_layers': 1,                     # TimesNet blocks
    'enc_in': num_features,            # Numero di ingressi nel modello
    'd_model': 256,                    # Model size
    'd_ff': 512,                       # (2 d_model)
    'c_out': 2,                        # Numero di output del modello
    'embed': 'learned',                # Embedding ('fixed' o 'learned')
    'freq': 's',                       # (second)
    'dropout': 0.1,
    'num_class': 2,                    # Numero di classi per la classificazione
    'top_k': 3,
    'num_kernels': 8,
}

configs = Config(**configs_dict)


model = Model(configs)
model = model.to(device)
print(f"{CYAN}{model}{RESET}")

train(model, train_dataloader, val_dataloader, learning_rate, num_epochs, patience, device=device, needDotLogits=False, isTnet=True)
evaluate(model, test_dataloader, device='cpu', needDotLogits=False, isTnet=True)




