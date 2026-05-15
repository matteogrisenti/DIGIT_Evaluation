import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# Import the decoder we built in the previous step
from force_pipeline import SparshT1ForceDecoder

# --- Custom Dataset to load our saved embeddings ---
class EmbeddingDataset(Dataset):
    def __init__(self, data_path):
        self.data = torch.load(data_path)
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        return item["embedding"], item["force"]

# --- Main Training Script ---
def train_probe(data_path, epochs=50, batch_size=32, lr=1e-4):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Load Data
    print("Loading precomputed embeddings...")
    dataset = EmbeddingDataset(data_path)
    # Shuffle is critical so the model learns generalized features, not the sequence of your data
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True) 
    
    # 2. Initialize the Blank Decoder
    model = SparshT1ForceDecoder().to(device)
    
    # 3. Setup Loss Function and Optimizer
    # MSE (Mean Squared Error) is standard for regression tasks like force estimation
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    print(f"Starting training on {device}...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for batch_embeddings, batch_forces in dataloader:
            # Move data to RTX 2050
            batch_embeddings = batch_embeddings.to(device)
            batch_forces = batch_forces.to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass: guess the forces
            predicted_forces = model(batch_embeddings)
            
            # Calculate the error
            loss = criterion(predicted_forces, batch_forces)
            
            # Backward pass: calculate how to fix the weights
            loss.backward()
            
            # Update the weights
            optimizer.step()
            
            running_loss += loss.item()
            
        avg_loss = running_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{epochs}] | Average MSE Loss: {avg_loss:.4f}")
        
    # 4. Save the trained weights!
    save_name = "my_trained_digit_t1_probe.pth"
    torch.save(model.state_dict(), save_name)
    print(f"\nTraining Complete! Weights saved to {save_name}")

if __name__ == "__main__":
    train_probe("digit_t1_embeddings.pt", epochs=50)