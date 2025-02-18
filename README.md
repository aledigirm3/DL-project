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