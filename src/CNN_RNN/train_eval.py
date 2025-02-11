import sys
import os
import time
import torch.optim as optim
import torch.nn as nn
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
import data_loader
import paths
from ansi_colors import *
os.chdir(current_dir)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"{CYAN}{device}{RESET}")
print(f"Cuda version: {torch.version.cuda}")
print(f"Torch version: {torch.__version__}\n")

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)

# =====================================================================================================

def evaluate(model, test_dataloader):
    model.eval()
    correct, total = 0, 0
    tp, fp, fn = 0, 0, 0 # For bachata

    with torch.no_grad():
        for videos, labels in test_dataloader:
            videos, labels = videos.to(device), labels.to(device)
            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item() 
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = 100 * correct / total
    precision = tp / (tp + fp) if (tp + fp) != 0 else 0
    recall = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"\n-{GREEN}Accuracy: {accuracy:.2f}%{RESET}")
    print(f"\n-{GREEN}Precision: {precision:.2f}{RESET}")
    print(f"\n-{GREEN}Recall: {recall:.2f}{RESET}")
    print(f"\n-{GREEN}F1: {f1_score:.2f}{RESET}\n")

# =====================================================================================================


video_paths, labels = data_loader.load_video_paths_and_labels(f"../{paths.DATASET}")


# Params
seq_length = 60  # Frames per video
lstm_hidden_size = 256  # Hidden state LSTM
num_classes = 2
bs = 32
learning_rate = 0.0004
num_epochs = 30

# Early stopping parameters
patience = 5
best_val_loss = float('inf')
epochs_without_improvement = 0
best_model_state = None

# Data splits
video_paths_train, video_paths_temp, labels_train, labels_temp = train_test_split(
    video_paths, labels, test_size=0.3, random_state=seed
)
video_paths_val, video_paths_test, labels_val, labels_test = train_test_split(
    video_paths_temp, labels_temp, test_size=0.5, random_state=seed  # 50% test, 50% validation
)

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
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()  # Loss function

# Train
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for videos, labels in train_dataloader:

        videos, labels = videos.to(device), labels.to(device)

        optimizer.zero_grad()  # Reset gradients

        outputs = model(videos)
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss / len(train_dataloader)}")

    # Validation phase
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for videos, labels in val_dataloader:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            loss = criterion(outputs, labels)
            val_loss += loss.item()

    val_loss /= len(val_dataloader)
    print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss}\n")

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0

        best_model_state = model.state_dict()

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= patience:
        print(f"Early stopping triggered after {epoch+1} epochs.")
        break

if best_model_state is not None:
    model.load_state_dict(best_model_state)

evaluate(model, test_dataloader)

end_time = time.time()
execution_time = end_time - start_time
print(f"{CYAN}Execution time: {execution_time / 60} min{RESET}\n")


