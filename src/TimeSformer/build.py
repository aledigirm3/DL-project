import sys
import os
import torch
import time
import numpy as np
import random
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from TimeSformer_process_dataset import DanceDataset
from sklearn.model_selection import train_test_split
from transformers import AutoModelForVideoClassification, AutoConfig



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
#device = 'cpu'
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


# Data augmentation
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Params
seq_length = 30  # Frames per video
bs = 8
num_classes = 2
learning_rate = 0.00003
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

# Dataset and DataLoader
train_dataset = DanceDataset(video_paths_train, labels_train, seq_length,transform=transform)
test_dataset = DanceDataset(video_paths_test, labels_test, seq_length, transform=transform)
val_dataset = DanceDataset(video_paths_val, labels_val, seq_length, transform=transform)

print(f"Number of sequences: {train_dataset.count_sequences() + test_dataset.count_sequences() + val_dataset.count_sequences()} (train + val + test)\n")

train_dataloader = DataLoader(train_dataset, batch_size=bs, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=bs, shuffle=False)
val_dataloader = DataLoader(val_dataset, batch_size=bs, shuffle=False)


config = AutoConfig.from_pretrained('facebook/timesformer-base-finetuned-k400')
config.num_frames = seq_length
config.num_labels = num_classes

 # Pre-trained on Kinetics-400
model = AutoModelForVideoClassification.from_pretrained(
    'facebook/timesformer-base-finetuned-k400',
    config=config,
    ignore_mismatched_sizes=True
    )

model = model.to(device)


start_time = time.time()

train(model, train_dataloader, val_dataloader, learning_rate, num_epochs, patience, device)
evaluate(model, test_dataloader, device)

end_time = time.time()
execution_time = end_time - start_time
print(f"{CYAN}Execution time: {execution_time / 60} min{RESET}\n")
