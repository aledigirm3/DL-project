import sys
import os
import torch.optim as optim
import torch.nn as nn
import torch
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

# Data splits
video_paths_train, video_paths_test, labels_train, labels_test = train_test_split(
    video_paths, labels, test_size=0.2, random_state=42
)

# Dataset and DataLoader
train_dataset = DanceDataset(video_paths_train, labels_train, transform=transforms.ToTensor(), seq_length=seq_length)
test_dataset = DanceDataset(video_paths_test, labels_test, transform=transforms.ToTensor(), seq_length=seq_length)

print(f"Number of sequences: {train_dataset.count_sequences() + test_dataset.count_sequences()} (train + test)\n")

#sys.exit(0)

train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=False)

# Model init
model = CNN_LSTM(seq_length=seq_length, lstm_hidden_size=lstm_hidden_size, num_classes=num_classes)
model = model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()  # Loss function

# Train
num_epochs = 10
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

evaluate(model, test_dataloader)


