import sys
import os
import torch
import numpy as np
import random
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader, random_split
from TimeSformer_process_dataset import DanceDataset
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


# ==================================== Evaluation =============================================================================
def evaluate(model, test_loader):

    model.eval()

    correct = 0
    total = 0
    tp_b, fp_b, fn_b = 0, 0, 0 # For bachata
    tp_s, fp_s, fn_s = 0, 0, 0 # For salsa

    with torch.no_grad():
        for videos, labels in test_loader:
            videos, labels = videos.to(device), labels.to(device)

            outputs = model(videos)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
 # Bachata
            tp_b += ((predicted == 1) & (labels == 1)).sum().item()
            fp_b += ((predicted == 1) & (labels == 0)).sum().item() 
            fn_b += ((predicted == 0) & (labels == 1)).sum().item()

            # Salsa
            tp_s += ((predicted == 0) & (labels == 0)).sum().item()
            fp_s += ((predicted == 0) & (labels == 1)).sum().item() 
            fn_s += ((predicted == 1) & (labels == 0)).sum().item()

    accuracy = 100 * correct / total
    print(f"\n- {GREEN}Accuracy: {accuracy:.2f}%{RESET}")

    # Bachata
    precision = tp_b / (tp_b + fp_b) if (tp_b + fp_b) != 0 else 0
    recall = tp_b / (tp_b + fn_b) if (tp_b + fn_b) != 0 else 0

    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"\n{RED}BACHATA performance metrics:{RESET}")
    print(f"\n- {GREEN}Precision: {precision:.2f}{RESET}")
    print(f"\n- {GREEN}Recall: {recall:.2f}{RESET}")
    print(f"\n- {GREEN}F1: {f1_score:.2f}{RESET}\n")

    # Salsa
    precision = tp_s / (tp_s + fp_s) if (tp_s + fp_s) != 0 else 0
    recall = tp_s / (tp_s + fn_s) if (tp_s + fn_s) != 0 else 0

    if (precision + recall) == 0:
        f1_score = 0
    else:
        f1_score = 2 * (precision * recall) / (precision + recall)

    print(f"\n{RED}SALSA performance metrics:{RESET}")
    print(f"\n- {GREEN}Precision: {precision:.2f}{RESET}")
    print(f"\n- {GREEN}Recall: {recall:.2f}{RESET}")
    print(f"\n- {GREEN}F1: {f1_score:.2f}{RESET}\n")
# =============================================================================================================================


# ==================================== Training ===============================================================================

def train(model, train_dataloader, val_dataloader, num_epochs=10):
    
    # Early stopping parameters
    patience = patience
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for videos, labels in train_dataloader:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(videos)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataloader):.4f}")
        
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
# =============================================================================================================================



# ================================== 4. Dataset e Training/Test Split =========================================================

# Data augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

bs = 8
learning_rate = 0.0001
patience = 8
num_epochs = 40

video_paths, labels = data_loader.load_video_paths_and_labels(f"../{paths.DATASET}")

# Data splits
video_paths_train, video_paths_temp, labels_train, labels_temp = train_test_split(
    video_paths, labels, test_size=0.3, random_state=seed
)
video_paths_val, video_paths_test, labels_val, labels_test = train_test_split(
    video_paths_temp, labels_temp, test_size=0.6, random_state=seed  # 50% test, 50% validation
)

# Dataset and DataLoader
train_dataset = DanceDataset(video_paths_train, labels_train, transform=transform)
test_dataset = DanceDataset(video_paths_test, labels_test, transform=transform)
val_dataset = DanceDataset(video_paths_val, labels_val, transform=transform)

print(f"Number of sequences: {train_dataset.count_sequences() + test_dataset.count_sequences() + val_dataset.count_sequences()} (train + val + test)\n")

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)


model = timm.create_model('timesformer_base_patch16_224', pretrained=True, num_classes=2) # Pre-trained on Kinetics-400
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

train(model, train_dataloader, val_dataloader, num_epochs ,patience)
evaluate(model, test_dataloader)
# =============================================================================================================================
