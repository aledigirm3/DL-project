import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import timm
from torch.utils.data import DataLoader, random_split
from TimeSformer_process_dataset import DanceDataset

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


# ========================================== Load TimeSformer model ===========================================================
model = timm.create_model('timesformer_base_patch16_224', pretrained=True, num_classes=2) # Pre-trained on Kinetics-400
model = model.to(device)
# =============================================================================================================================


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
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

def train(model, train_loader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for videos, labels in train_loader:
            videos, labels = videos.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(videos)

            loss = criterion(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")
# =============================================================================================================================



# ================================== 4. Dataset e Training/Test Split =========================================================

# Data augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

video_paths, labels = data_loader.load_video_paths_and_labels(f"../{paths.DATASET}")

dataset = DanceDataset(video_paths, labels, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

train(model, train_loader)
evaluate(model, test_loader)
# =============================================================================================================================
