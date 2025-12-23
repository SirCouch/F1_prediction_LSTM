import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from model import F1Predictor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import json
import os

def train_model():
    # Load data
    try:
        h_X = np.load('history_X.npy')
        d_ids = np.load('driver_ids.npy')
        t_ids = np.load('team_ids.npy')
        g_pos = np.load('grid_pos.npy')
        y = np.load('y.npy')
        
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
            
    except FileNotFoundError:
        print("Data files not found. Run data_loader.py first.")
        return

    # Convert to float32/long (for IDs)
    h_X = torch.from_numpy(h_X).float()
    d_ids = torch.from_numpy(d_ids).long()
    t_ids = torch.from_numpy(t_ids).long()
    g_pos = torch.from_numpy(g_pos).float().unsqueeze(1) # (N, 1)
    y = torch.from_numpy(y).float().view(-1, 1)

    # Split data indices
    N = len(y)
    indices = np.arange(N)
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)

    # Create datasets
    train_dataset = TensorDataset(
        h_X[train_idx], 
        d_ids[train_idx], 
        t_ids[train_idx], 
        g_pos[train_idx], 
        y[train_idx]
    )
    
    val_dataset = TensorDataset(
        h_X[val_idx], 
        d_ids[val_idx], 
        t_ids[val_idx], 
        g_pos[val_idx], 
        y[val_idx]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)

    # Hyperparameters
    hidden_size = 64
    num_epochs = 100
    learning_rate = 0.001
    
    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = F1Predictor(
        num_drivers=metadata['num_drivers'],
        num_teams=metadata['num_teams'],
        hidden_size=hidden_size
    ).to(device)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    val_losses = []

    print("Starting training...")
    for epoch in range(num_epochs):
        model.train()
        batch_losses = []
        
        for batch in train_loader:
            # Unpack batch
            b_h_X, b_d_ids, b_t_ids, b_g_pos, b_y = [b.to(device) for b in batch]
            
            # Forward pass
            outputs = model(b_d_ids, b_t_ids, b_h_X, b_g_pos)
            loss = criterion(outputs, b_y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            batch_losses.append(loss.item())
        
        avg_train_loss = np.mean(batch_losses)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_batch_losses = []
        with torch.no_grad():
            for batch in val_loader:
                b_h_X, b_d_ids, b_t_ids, b_g_pos, b_y = [b.to(device) for b in batch]
                
                outputs = model(b_d_ids, b_t_ids, b_h_X, b_g_pos)
                loss = criterion(outputs, b_y)
                val_batch_losses.append(loss.item())
        
        avg_val_loss = np.mean(val_batch_losses)
        val_losses.append(avg_val_loss)
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

    # Save the model
    torch.save(model.state_dict(), 'f1_lstm_model.pth')
    print("Model saved as f1_lstm_model.pth")

    # Plot losses
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig('loss_plot.png')
    print("Loss plot saved as loss_plot.png")

if __name__ == "__main__":
    train_model()
