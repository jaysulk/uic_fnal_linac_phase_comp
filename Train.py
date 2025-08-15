import os
import json
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, classification_report, accuracy_score
import scipy.fft as spft
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import TensorDataset, DataLoader

# Load accelerator component data:
# - Input: Cavity control signals
# - Output: Target parameters
# - Response matrix: System behavior model
# - Sensor positions: BPM locations in accelerator
input = pd.read_csv('input.csv').values.tolist()
output = pd.read_csv('output.csv').values.tolist()
response_matrix = pd.read_csv('response_matrix.csv').values.tolist()
json_file = open('sensor_positions.json')

def fetch_data(file, datacols, cuts, setdevs):
    """Load and preprocess CSV data from accelerator diagnostics"""
    try:
        dataset = pd.read_csv(file)
    except pd.errors.ParserError as e:
        # Handle problematic CSV formatting by inspecting lines
        print(f"ParserError when reading {file}: {e}")
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                # Diagnostic printing for problematic lines
                if i < 10 or 620 < i < 640:
                    fields = line.strip().split(',')
                    print(f"Line {i+1}: Number of fields = {len(fields)}")
                    if i+1 == 629:
                        print(f"Problematic Line 629 content: {line.strip()}")
        raise e

    # Clean column names by replacing special characters
    dataset.columns = dataset.columns.str.replace("[()]", "_", regex=True)
    
    # Filter relevant columns using regex patterns
    cols = list(dataset.filter(regex='|'.join(datacols)))
    setdevs = ['L:%s_' % d for d in setdevs]
    cols = [col for col in cols if col not in setdevs]

    # Process column names and remove unwanted columns
    subset = dataset.loc[:, cols]
    subset.columns = subset.columns.str.replace("_R_|_S_", "", regex=True)
    subset.drop(list(subset.filter(regex=r'\\.1|Time|step|iter|stamp')), axis=1, inplace=True)

    # Apply query filters if provided
    if len(cuts) > 0:
        subset.query(cuts, inplace=True)

    # Clean missing values
    subset.dropna(inplace=True)
    return subset

def load_BPMphase_data_multi(cavs, files, dropdevs, scan=True):
    """Load beam position monitor (BPM) phase data from multiple files"""
    dfs = []
    for file in files:
        # Fetch relevant columns (cavities, BPMs, beam flags)
        df = fetch_data(file, cavs + ['BF', 'BPM', 'SQ'], '', 
                       ['%s_S' % cav[2:] for cav in cavs] if scan else [])
        
        try:
            # Remove unwanted devices and diagnostic columns
            df = df.drop(list(df.filter(regex=r'20|B:|SS|SQT')), axis=1)
            df = df.drop(list(df.filter(regex=r'|'.join(dropdevs))), axis=1)
        except:
            continue

        # Phase unwrapping for values crossing ±180° boundary
        for col in df.columns:
            if abs(df[col].min() - df[col].max()) > 350:
                # Adjust values crossing circular boundary
                if np.sign(df[col]).mean() < 0:
                    df[col] = df[col].apply(lambda x: x if x < 0 else x - 360)
                else:
                    df[col] = df[col].apply(lambda x: x if x > 0 else x + 360)
        dfs.append(df)
    return dfs

# Load BPM positions and filter noisy devices
BPM_positions = json.load(json_file)
devices_to_drop = ['L:BPM2OF', 'L:BPM3IF', 'L:BPM3OF', 'L:BPM5IF', "L:BPM4IF", "L:BPM5OF", "L:D44BF"]
for device in devices_to_drop:
    BPM_positions.pop(device, None)

BPM_list = list(BPM_positions.keys())
dist_data = list(BPM_positions.values())

# Accelerator component labels
cavnames = ['Buncher', 'Tank 1', 'Tank 2', 'Tank 3', 'Tank 4', 'Tank 5', 'RFQ']
cavs = ['L:RFBPAH', 'L:V1QSET', 'L:V2QSET', 'L:V3QSET', 'L:V4QSET', 'L:V5QSET', 'L:RFQPAH']
cavs_read = ['L:V%iSQ' % n for n in range(1, 6)]  # Readback signals
basis_choice = (0, 1, 2, 3, 4, 5, 6)  # Basis vectors selection

def plot_fit_traj(cavs, target_trajectory, basis, response_matrix, main_response_matrix, targetlbl=None):
    """Entry point for model training and trajectory plotting"""
    target_tensor = torch.tensor(target_trajectory, dtype=torch.float32)
    tensor_b_vecs = torch.tensor(response_matrix, dtype=torch.float32)
    # Initiate model training
    coefs = linear_fit_to_basis(tensor_b_vecs, target_tensor, main_response_matrix)
    return None

class DeepNN(nn.Module):
    """Feedforward neural network for accelerator parameter prediction"""
    def __init__(self, input_dim, output_dim):
        super(DeepNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),  # Input: BPM readings (27)
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)  # Output: cavity parameters (7)
        )

    def forward(self, x):
        return self.encoder(x)

def save_checkpoint(model, optimizer, epoch, loss, path="checkpoint_epoch_{}.pth"):
    """Save training state for recovery"""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, path.format(epoch))

def energy_consistent_loss(preds, labels, strength=1e-2):
    """Encourage parameter clustering by class (intra-class compactness)"""
    loss_energy = 0.0
    for cls in labels.unique():
        idx = labels == cls
        cls_preds = preds[idx]
        if cls_preds.size(0) > 1:
            # Calculate mean pairwise distance within class
            pairwise_dists = torch.pdist(cls_preds, p=2)
            loss_energy += pairwise_dists.pow(2).mean()
    return strength * loss_energy

def temporal_smoothness_loss(preds, sequence_indices, strength=1e-2):
    """Encourage temporal consistency in parameter predictions"""
    loss_temp = 0.0
    count = 0
    for seq_id in sequence_indices.unique():
        idx = sequence_indices == seq_id
        seq_preds = preds[idx]
        if seq_preds.size(0) > 1:
            # Penalize large differences between consecutive steps
            diffs = seq_preds[1:] - seq_preds[:-1]
            loss_temp += diffs.pow(2).mean()
            count += 1
    return strength * (loss_temp / max(count, 1))

def linear_fit_to_basis(X_tensor, Y_tensor, rresponse_matrix, sequence_length=20, ts=800,
                        batch_size=500, epochs=1000, lr=0.0005, print_every=100):
    """Core training loop"""
    # Convert response matrix to tensor if needed
    if not isinstance(rresponse_matrix, torch.Tensor):
        rresponse_matrix = torch.tensor(rresponse_matrix, dtype=torch.float32)

    # Normalize inputs and targets
    X_tensor = (X_tensor - X_tensor.mean(dim=0)) / (X_tensor.std(dim=0) + 1e-8)
    Y_tensor = (Y_tensor - Y_tensor.mean(dim=0)) / (Y_tensor.std(dim=0) + 1e-8)

    # Create class labels for energy consistency loss
    unique_classes, class_labels = torch.unique(Y_tensor, dim=0, return_inverse=True)

    # Split into training/test sets
    X_train, X_test, Y_train, Y_test, labels_train, labels_test = train_test_split(
        X_tensor, Y_tensor, class_labels, test_size=ts)

    # Create data loaders
    train_loader = DataLoader(TensorDataset(X_train, Y_train, labels_train), 
                             batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(TensorDataset(X_test, Y_test, labels_test), 
                            batch_size=batch_size, shuffle=False)

    # Initialize model and optimizer
    model = DeepNN(input_dim=27, output_dim=7)  # 27 BPMs → 7 cavity params
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.95, 0.999))

    # Loss components and weighting scheme
    regression_loss_fn = nn.MSELoss()
    classification_loss_fn = nn.CrossEntropyLoss()
    weights = [0.9, 0.0025, 0.8, 0.9, 0.015]  # Weight for each loss component

    train_losses, test_losses = [], []

    for epoch in range(epochs):
        # Training phase
        model.train()
        running_train_loss = 0.0
        for x_batch, y_batch, lbl_batch in train_loader:
            optimizer.zero_grad()
            out = model(x_batch)  # Predicted cavity parameters

            # 1. Main regression loss (cavity params)
            loss_reg = regression_loss_fn(out, y_batch)
            
            # 2. Classification via distance to class
            dists = torch.cdist(out, unique_classes)
            logits = -dists  # Convert distances to "scores"
            loss_class = classification_loss_fn(logits, lbl_batch)
            
            # 3. Energy consistency regularization
            loss_energy = energy_consistent_loss(out, lbl_batch)
            
            # 4. Temporal smoothness regularization
            seq_ids = (torch.arange(len(lbl_batch)) // sequence_length).to(out.device)
            loss_temp = temporal_smoothness_loss(out, seq_ids)
            
            # 5. Tragectory Fit-based loss: Reconstruct BPM readings
            predicted_trajectory = torch.matmul(out, rresponse_matrix[:, 7:])
            fit_loss = regression_loss_fn(predicted_trajectory, x_batch)

            # Combined loss with weighting
            loss = sum(w * l for w, l in zip(weights, 
                    [loss_reg, loss_class, loss_energy, loss_temp, fit_loss]))
            
            loss.backward()
            optimizer.step()
            running_train_loss += loss.item() * x_batch.size(0)
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_losses.append(epoch_train_loss)
        # Validation phase
        model.eval()
        running_test_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch, lbl_batch in test_loader:
                out = model(x_batch)

                # Recompute all losses for validation
                loss_reg = regression_loss_fn(out, y_batch)

                dists = torch.cdist(out, unique_classes)
                logits = -dists
                loss_class = classification_loss_fn(logits, lbl_batch)

                # energy and temporal
                loss_energy = energy_consistent_loss(out, lbl_batch)
                seq_ids = (torch.arange(len(lbl_batch)) // sequence_length).to(out.device)
                loss_temp = temporal_smoothness_loss(out, seq_ids)

                predicted_trajectory = torch.matmul(out, rresponse_matrix[:, 7:])
                fit_loss_test = regression_loss_fn(predicted_trajectory, x_batch)
                loss = weights[0] * loss_reg+weights[1] * loss_class+weights[2] * loss_energy+weights[3] * loss_temp+weights[4] * fit_loss_test  # Only regression loss
                running_test_loss += loss.item() * x_batch.size(0)
        epoch_test_loss = running_test_loss / len(test_loader.dataset)
        test_losses.append(epoch_test_loss)
                
        # Periodic checkpointing
        if epoch % print_every == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch:4d} ▶ train_loss: {epoch_train_loss:.6f}  test_loss: {epoch_test_loss:.6f}")

            save_checkpoint(
                model, optimizer, epoch, epoch_test_loss,
                path=f"checkpoint-epoch-{{}}.pth"
            )

    # Plot loss curves
    plt.plot(train_losses[1:], label="Train Loss")
    plt.plot(test_losses[1:], label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    return model, train_losses, test_losses

# Start training pipeline
print("\nStarting execution of model...........\n")
plot_fit_traj(cavs, output, basis_choice, input, response_matrix)
print("\nCompleted Model Training\n")