import torch
from torch.utils.data import DataLoader
import sys
import os
from collections import Counter
from prep_data import build_tensor_dataset

current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from train_eval.training import train
from train_eval.evaluation import evaluate
from ansi_colors import *
os.chdir(current_dir)


timesnet_path = '../../Time-Series-Library'

if timesnet_path:
    sys.path.append(timesnet_path)

else:
    print(f"\n{RED}TimesNet path not found{RESET}\n")
    sys.exit(0)
from models.TimesNet import Model # type: ignore

class Config:
    def __init__(self, **kwargs):
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

num_features = 34

print(f"{GREEN}TRAIN{RESET}")
train_dataset = build_tensor_dataset(dataset_folder + 'train', window_size)
print(f"{GREEN}VAL{RESET}")
val_dataset = build_tensor_dataset(dataset_folder + 'val', window_size)
print(f"{GREEN}TEST{RESET}")
test_dataset = build_tensor_dataset(dataset_folder + 'test', window_size)

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
    'e_layers': 2,                     # TimesNet blocks
    'enc_in': num_features,            # N. inputs of model
    'd_model': 64,                    # Model size
    'd_ff': 128,                       # (2 * d_model)
    'c_out': 2,                        # N. outputs of model
    'embed': 'learned',                # Embedding ('fixed' o 'learned')
    'freq': 's',                       # (second)
    'dropout': 0.1,
    'num_class': 2,                    # N. of classes for classification
    'top_k': 3,
    'num_kernels': 8,
}

configs = Config(**configs_dict)


model = Model(configs)
model = model.to(device)
print(f"{CYAN}{model}{RESET}")

train(model, train_dataloader, val_dataloader, learning_rate, num_epochs, patience, device=device, needDotLogits=False, isTnet=True)
evaluate(model, test_dataloader, device=device, needDotLogits=False, isTnet=True)




