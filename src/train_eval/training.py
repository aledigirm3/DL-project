import torch
import torch.nn as nn
import torch.optim as optim


# Train
def train(model, train_dataloader, val_dataloader, learning_rate, num_epochs, patience, device='cpu'):
    
    # Early stopping parameters
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    best_model_state = None
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()  # Loss function
    
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