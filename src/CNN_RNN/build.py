import sys
import os
import time
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from torchvision import transforms
from model import CNN_LSTM
from CNN_RNN_process_dataset import DanceDataset
from sklearn.model_selection import train_test_split


current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_eval.training import train
from train_eval.evaluation import evaluate
import data_loader
import paths
from ansi_colors import *
os.chdir(current_dir)


# ===========================================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"{CYAN}{device}{RESET}")
print(f"Cuda version: {torch.version.cuda}")
print(f"Torch version: {torch.__version__}\n")
# ===========================================================================================
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
# ===========================================================================================


# Params
seq_length = 90  # Frames per video
lstm_hidden_size = 256  # Hidden state LSTM
num_classes = 2
bs = 8
learning_rate = 0.001
num_epochs = 40
patience = 8 # Early stopping


video_paths, labels = data_loader.load_video_paths_and_labels(f"../{paths.DATASET}")

# Data splits
video_paths_train, video_paths_temp, labels_train, labels_temp = train_test_split(
    video_paths, labels, test_size=0.3, random_state=seed
)
video_paths_val, video_paths_test, labels_val, labels_test = train_test_split(
    video_paths_temp, labels_temp, test_size=0.6, random_state=seed  # 50% test, 50% validation
)

'''prefix_salsa = "../../salsa_bachata_dataset\\salsa"
count_salsa = sum(1 for path in video_paths_train if path.startswith(prefix_salsa))
prefix_bachata = "../../salsa_bachata_dataset\\bachata"
count_bachata = sum(1 for path in video_paths_train if path.startswith(prefix_bachata))

print(f"Numero di path che iniziano con '{prefix_bachata}': {count_bachata}")
print(f"Numero di path che iniziano con '{prefix_salsa}': {count_salsa}")'''


#sys.exit(0)

# Dataset and DataLoader
train_dataset = DanceDataset(video_paths_train, labels_train, transform=transforms.ToTensor(), seq_length=seq_length)
test_dataset = DanceDataset(video_paths_test, labels_test, transform=transforms.ToTensor(), seq_length=seq_length)
val_dataset = DanceDataset(video_paths_val, labels_val, transform=transforms.ToTensor(), seq_length=seq_length)

print(f"Number of sequences: {train_dataset.count_sequences() + test_dataset.count_sequences() + val_dataset.count_sequences()} (train + val + test)\n")

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)

# Model init
model = CNN_LSTM(seq_length=seq_length, lstm_hidden_size=lstm_hidden_size, num_classes=num_classes)
model = model.to(device)


start_time = time.time()

train(model, train_dataloader, val_dataloader, learning_rate, num_epochs, patience, device)
evaluate(model, test_dataloader, device)

end_time = time.time()
execution_time = end_time - start_time
print(f"{CYAN}Execution time: {execution_time / 60} min{RESET}\n")


