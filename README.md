# DL-project
Projects for the course Deep Learning at Roma Tre University.

# CNN_RNN Results
Using ResNet101 + LSTM. ResNet weights are frozen for the entire training
### seq_length = 90, num_epochs = 40 (patience = 8), lr = 0.001, batch_size = 8
## LOG:
Early stopping triggered after 22 epochs.

- Accuracy: 61.11%

### BACHATA performance metrics:

- Precision: 0.70

- Recall: 0.64

- F1: 0.67


### SALSA performance metrics:

- Precision: 0.50

- Recall: 0.57

- F1: 0.53
---
Execution time: 47.938197072347 min

(Cuda version: 12.1, Torch version: 2.5.1+cu121)


---

# TimeSformer Results
Using TimeSformer pretrained on Kinetics-400 and fine-tuned on salsa_bachata_dataset
### seq_length = 20, num_epochs = 40 (patience = 7), lr = 0.00002, batch_size = 16
## LOG:
Early stopping triggered after 9 epochs.

- Accuracy: 66.67%

### BACHATA performance metrics:

- Precision: 0.78

- Recall: 0.64

- F1: 0.70


### SALSA performance metrics:

- Precision: 0.56

- Recall: 0.71

- F1: 0.63

---

Execution time: 202.22746453285217 min

(cpu, Torch version: 2.5.1)