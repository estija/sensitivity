import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import matplotlib.pyplot as plt

# Constants
P = 113  # Modulo value
VOCAB_SIZE = P + 1  # Numbers from 0 to 112, plus "=" token
EQUAL_TOKEN = P  # Assign "=" token the index P
SEQ_LEN = 3  # "a b ="

class ModAddDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs  # Shape: (num_samples, seq_len)
        self.targets = targets  # Shape: (num_samples,)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        x = self.inputs[idx]  # Sequence of token indices
        y = self.targets[idx]  # Target value
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)

# Generate all possible pairs (a, b)
all_pairs = [(a, b) for a in range(P) for b in range(P)]
total_pairs = len(all_pairs)
print(f"Total possible pairs: {total_pairs}")

# Assign indices to "=" token
equal_token_idx = EQUAL_TOKEN

# Create all possible input sequences and targets
all_inputs = []
all_targets = []
for a, b in all_pairs:
    input_seq = [a, b, equal_token_idx]
    target = (a + b) % P
    all_inputs.append(input_seq)
    all_targets.append(target)

all_inputs = np.array(all_inputs)
all_targets = np.array(all_targets)

def train_and_evaluate(seed, num_epochs=20000, noise_std=0.1):
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Shuffle the data
    indices = np.arange(total_pairs)
    np.random.shuffle(indices)
    shuffled_inputs = all_inputs[indices]
    shuffled_targets = all_targets[indices]

    # Split into training and test sets (30% for training)
    train_size = int(0.3 * total_pairs)
    train_inputs = shuffled_inputs[:train_size]
    train_targets = shuffled_targets[:train_size]
    test_inputs = shuffled_inputs[train_size:]
    test_targets = shuffled_targets[train_size:]

    print(f"Seed {seed}: Training samples: {len(train_inputs)}, Test samples: {len(test_inputs)}")

    # Create Dataset objects
    train_dataset = ModAddDataset(train_inputs, train_targets)
    test_dataset = ModAddDataset(test_inputs, test_targets)

    # Data loaders (full batch gradient descent)
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

    # Define the model
    class SimpleTransformer(nn.Module):
        def __init__(self, vocab_size, seq_len, d_model, num_heads, dim_feedforward, dropout=0.0):
            super(SimpleTransformer, self).__init__()
            self.d_model = d_model

            # Token and positional embeddings
            self.token_embed = nn.Embedding(vocab_size, d_model)
            self.pos_embed = nn.Parameter(torch.zeros(seq_len, d_model))

            # Transformer Encoder Layer
            self.attention = nn.MultiheadAttention(embed_dim=d_model, num_heads=num_heads, dropout=dropout)
            self.fc1 = nn.Linear(d_model, dim_feedforward)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(dim_feedforward, d_model)

            # Output layer (unembedding)
            self.fc_out = nn.Linear(d_model, vocab_size)

        def forward(self, src, noise_std=0.0):
            # src shape: (batch_size, seq_len)
            batch_size, seq_len = src.size()

            # Token embedding
            token_embeddings = self.token_embed(src)  # (batch_size, seq_len, d_model)

            # Add positional embeddings
            embeddings = token_embeddings + self.pos_embed  # (batch_size, seq_len, d_model)

            # Add Gaussian noise to embeddings if noise_std > 0
            if noise_std > 0.0:
                noise = torch.randn_like(embeddings[:, :2, :]) * noise_std
                embeddings[:, :2, :] = embeddings[:, :2, :] + noise

            # Prepare for multihead attention: (seq_len, batch_size, d_model)
            embeddings = embeddings.transpose(0, 1)

            # Self-attention
            attn_output, _ = self.attention(embeddings, embeddings, embeddings)
            attn_output = attn_output.transpose(0, 1)  # (batch_size, seq_len, d_model)

            # MLP
            x = self.fc1(attn_output)  # (batch_size, seq_len, dim_feedforward)
            x = self.relu(x)
            x = self.fc2(x)  # (batch_size, seq_len, d_model)

            # Read output above "=" token (last token)
            output_token = x[:, -1, :]  # (batch_size, d_model)

            # Output logits
            logits = self.fc_out(output_token)  # (batch_size, vocab_size)

            return logits

    # Model parameters
    d_model = 128
    num_heads = 4
    dim_feedforward = 512
    dropout = 0.0  # No dropout as per description

    # Instantiate the model
    model = SimpleTransformer(
        vocab_size=VOCAB_SIZE,
        seq_len=SEQ_LEN,
        d_model=d_model,
        num_heads=num_heads,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )

    # Move model to device (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (AdamW with weight decay parameter λ = 1)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1.0)

    # Training loop
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    sensitivities = []

    for epoch in range(1, num_epochs + 1):
        # Training
        model.train()
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(x_batch)  # (batch_size, vocab_size)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

        # Record training loss and accuracy
        train_losses.append(loss.item())
        _, predicted = torch.max(outputs.data, 1)
        correct = (predicted == y_batch).sum().item()
        accuracy = correct / y_batch.size(0)
        train_accuracies.append(accuracy)

        # Sensitivity calculation on training data
        model.eval()
        with torch.no_grad():
            # Original predictions
            outputs_clean = model(x_batch)
            _, predicted_clean = torch.max(outputs_clean.data, 1)

            # Predictions with corrupted embeddings
            outputs_noisy = model(x_batch, noise_std=noise_std)
            _, predicted_noisy = torch.max(outputs_noisy.data, 1)

            # Calculate sensitivity
            changed_predictions = (predicted_clean != predicted_noisy).sum().item()
            sensitivity = changed_predictions / y_batch.size(0)
            sensitivities.append(sensitivity)

        # Evaluation on test set every 100 epochs
        if epoch % 100 == 0 or epoch == 1:
            with torch.no_grad():
                for x_test, y_test in test_loader:
                    x_test = x_test.to(device)
                    y_test = y_test.to(device)
                    outputs_test = model(x_test)
                    test_loss = criterion(outputs_test, y_test)
                    _, predicted_test = torch.max(outputs_test.data, 1)
                    correct_test = (predicted_test == y_test).sum().item()
                    test_accuracy = correct_test / y_test.size(0)
                    test_losses.append(test_loss.item())
                    test_accuracies.append(test_accuracy)

            print(f"Seed {seed}, Epoch {epoch}/{num_epochs}, "
                  f"Train Loss: {loss.item():.4f}, Train Acc: {accuracy*100:.2f}%, "
                  f"Test Loss: {test_loss.item():.4f}, Test Acc: {test_accuracy*100:.2f}%, "
                  f"Sensitivity: {sensitivity*100:.2f}%")

    # Convert recorded values to numpy arrays
    train_losses = np.array(train_losses)
    test_losses = np.array(test_losses)
    train_accuracies = np.array(train_accuracies)
    test_accuracies = np.array(test_accuracies)
    sensitivities = np.array(sensitivities)

    # Return metrics
    return train_losses, test_losses, train_accuracies, test_accuracies, sensitivities

# Run training and evaluation for different random seeds
num_runs = 5
seeds = [42, 43, 44, 45, 46]
num_epochs = 20000
noise_std = 0.1

# Initialize lists to store metrics for all runs
all_train_losses = []
all_test_losses = []
all_train_accuracies = []
all_test_accuracies = []
all_sensitivities = []

for seed in seeds:
    train_losses, test_losses, train_accuracies, test_accuracies, sensitivities = train_and_evaluate(
        seed, num_epochs=num_epochs, noise_std=noise_std)
    all_train_losses.append(train_losses)
    all_test_losses.append(test_losses)
    all_train_accuracies.append(train_accuracies)
    all_test_accuracies.append(test_accuracies)
    all_sensitivities.append(sensitivities)

# Plotting the metrics
x = num_epochs

# Epochs where test evaluation was performed
test_epochs = np.arange(1, x + 1)[(np.arange(x) + 1) % 100 == 0]
if 1 not in test_epochs:
    test_epochs = np.insert(test_epochs, 0, 1)

epochs = np.arange(1, x + 1)

plt.figure(figsize=(15, 15))

# Train Loss Plot
plt.subplot(4, 1, 1)
for i in range(num_runs):
    plt.plot(epochs, all_train_losses[i], label=f'Run {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Train Loss')
plt.title('Train Loss over Epochs')
plt.legend()
plt.grid(True)

# Test Loss Plot
plt.subplot(4, 1, 2)
for i in range(num_runs):
    plt.plot(test_epochs, all_test_losses[i], label=f'Run {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Test Loss')
plt.title('Test Loss over Epochs')
plt.legend()
plt.grid(True)

# Train Accuracy Plot
plt.subplot(4, 1, 3)
for i in range(num_runs):
    plt.plot(epochs, all_train_accuracies[i], label=f'Run {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Train Accuracy')
plt.title('Train Accuracy over Epochs')
plt.legend()
plt.grid(True)

# Sensitivity Plot
plt.subplot(4, 1, 4)
for i in range(num_runs):
    plt.plot(epochs, all_sensitivities[i] * 100, label=f'Run {i+1}')
plt.xlabel('Epoch')
plt.ylabel('Sensitivity (%)')
plt.title('Sensitivity over Epochs')
plt.legend()
plt.grid(True)

plt.tight_layout()

# Save the plot to a local file
plt.savefig('training_results.png')

print("Plot saved as 'training_results.png' in the current directory.")
