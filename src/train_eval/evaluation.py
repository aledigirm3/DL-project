import sys
import os
import torch


current_dir = os.getcwd()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from ansi_colors import *
os.chdir(current_dir)


def evaluate(model, test_dataloader, device='cpu'):
    model.eval()
    correct, total = 0, 0
    tp_b, fp_b, fn_b = 0, 0, 0 # For bachata
    tp_s, fp_s, fn_s = 0, 0, 0 # For salsa

    with torch.no_grad():
        for videos, labels in test_dataloader:
            videos, labels = videos.to(device), labels.to(device)
            
            outputs = model(videos) # (outputs.logits -> TimeSformer)

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
