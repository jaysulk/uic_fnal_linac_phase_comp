import os
import json
import itertools
#import imageio  # Commented out but could be used for GIF creation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Machine learning utilities
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, classification_report, accuracy_score

# Signal processing
import scipy.fft as spft

# PyTorch deep learning framework
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import TensorDataset, DataLoader

# Load accelerator component data
json_file = open('sensor_positions.json')  # BPM locations in the accelerator
df = pd.read_csv('response_matrix.csv')  # Physics model of accelerator behavior
response_matrix = df.values.tolist()  # Convert to Python list for processing

def fetch_data(file, datacols, cuts, setdevs):
    """Load and preprocess CSV data from accelerator diagnostics"""
    try:
        # Attempt to read CSV normally
        dataset = pd.read_csv(file)
    except pd.errors.ParserError as e:
        # Handle problematic CSV formatting
        print(f"ParserError when reading {file}: {e}")
        print("Diagnosing CSV formatting issues...")
        
        # Inspect file line-by-line to identify problems
        with open(file, 'r') as f:
            for i, line in enumerate(f):
                # Focus on problematic regions
                if i < 10 or (620 < i < 640):  
                    fields = line.strip().split(',')
                    print(f"Line {i+1}: {len(fields)} fields")
                    if i+1 == 629:  # Specific problematic line
                        print(f"Line 629 content: {line.strip()}")
        raise e

    # Clean column names: replace special characters
    dataset.columns = dataset.columns.str.replace("[()]", "_", regex=True)
    
    # Filter relevant columns using regex patterns
    cols = list(dataset.filter(regex='|'.join(datacols)))
    
    # Remove setpoint devices (keep readbacks)
    setdevs = ['L:%s_' % d for d in setdevs]
    cols = [col for col in cols if col not in setdevs]

    # Process column names
    subset = dataset.loc[:, cols]
    subset.columns = subset.columns.str.replace("_R_|_S_", "", regex=True)
    
    # Remove unwanted columns
    subset.drop(list(subset.filter(regex=r'\.1|Time|step|iter|stamp')), axis=1, inplace=True)

    # Apply data quality filters if provided
    if len(cuts) > 0:
        subset.query(cuts, inplace=True)

    # Clean missing values
    subset.dropna(inplace=True)
    return subset

def load_BPMphase_data_multi(cavs, files, dropdevs, scan=True):
    """Load beam position monitor (BPM) phase data from multiple files"""
    dfs = []
    for i, file in enumerate(files):
        # Fetch relevant columns (cavities, BPMs, beam flags)
        if scan:
            df = fetch_data(file, cavs + ['BF','BPM','SQ'], '', ['%s_S'%cav[2:] for cav in cavs])
        else:
            df = fetch_data(file, cavs + ['BF','BPM','SQ'], '', [])
        
        try:
            # Remove unwanted devices and diagnostic columns
            df = df.drop(list(df.filter(regex=r'20|B:|SS|SQT')), axis=1)
            df = df.drop(list(df.filter(regex=r'|'.join(dropdevs))), axis=1)
        except:
            continue
            
        # Phase unwrapping for circular measurements (±180° boundary)
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
devices_to_drop = ['L:BPM2OF','L:BPM3IF','L:BPM3OF','L:BPM5IF',"L:BPM4IF","L:BPM5OF","L:D44BF"]
for device in devices_to_drop:
    BPM_positions.pop(device, None)  # Remove problematic devices

BPM_list = list(BPM_positions.keys())  # Valid BPM names
dist_data = list(BPM_positions.values())  # Physical positions along beamline

# Accelerator component labels
cavnames = ['Buncher','Tank 1','Tank 2','Tank 3','Tank 4','Tank 5','RFQ']
cavs = ['L:RFBPAH', 'L:V1QSET', 'L:V2QSET', 'L:V3QSET', 'L:V4QSET', 'L:V5QSET','L:RFQPAH']  # Control knobs
cavs_read = ['L:V%iSQ'%n for n in range(1,6)]  # Sensor readbacks
cavs_both = cavs + cavs_read  # Combined list
basis_choice = (0,1,2,3,4,5,6)  # Basis vectors for physics model


def trajectory_fit(target_trajectory, coefs, response_matrix, save_path, targetlbl=None):
    """Plot predicted vs actual beam trajectory"""
    basis = [0,1,2,3,4,5,6]  # 7 basis vectors for accelerator physics
    
    if targetlbl:
        target = str(targetlbl)
    else:
        target='Target trajectory'
    idx = len(response_matrix)
    plt.plot(dist_data, target_trajectory, label=target)
    
    # Calculate physics-based prediction
    fit_line = np.zeros_like(dist_data)
    for i in range(len(basis)):
        # Convert the slice of response_matrix to a tensor before multiplication
        response_slice_tensor = torch.tensor(response_matrix[basis[i]][idx:], dtype=torch.float32)
        # Add contribution from each basis vector
        fit_line += coefs[i].detach().numpy() * response_slice_tensor.detach().numpy()

    # Plot prediction
    plt.plot(dist_data, fit_line, label=f"Fit")
    plt.title(save_path)
    plt.ylabel(r"$ \Delta \phi_{BPM}$ (deg)")
    plt.xlabel("Distance, m")
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.tight_layout()
    # Save figure to file
    plt.savefig(save_path, dpi=300)
    print(f"Trajectory plot saved to {save_path}")
    plt.show()

# Neural Network Definition
class DeepNN(nn.Module):
    """Feedforward network mapping BPM readings to cavity parameters"""
    def __init__(self, input_dim, output_dim):
        super(DeepNN, self).__init__()
        # Encoder architecture: 27 BPMs → 7 cavity parameters
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256), nn.ReLU(),  # Input: 27 BPM readings
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim)  # Output: 7 cavity parameters
        )

    def forward(self, x):
        return self.encoder(x)

def predict_single_input(input_1x27, checkpoint_path="checkpoint-epoch-999.pth"):
    """Make prediction using trained model"""
    # Convert input to tensor
    if isinstance(input_1x27, list) or isinstance(input_1x27, np.ndarray):
        input_tensor = torch.tensor(input_1x27, dtype=torch.float32).view(1, -1)
    elif isinstance(input_1x27, torch.Tensor):
        if input_1x27.ndim == 1:
            input_tensor = input_1x27.unsqueeze(0)  # Add batch dimension
        else:
            input_tensor = input_1x27
    else:
        raise ValueError("Input must be list, numpy array, or torch tensor")

    # Initialize and load trained model
    model = DeepNN(input_dim=27, output_dim=7)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()  # Set to evaluation mode

    # Make prediction
    with torch.no_grad():
        output = model(input_tensor)
        return output.squeeze(), input_tensor.squeeze()  # Remove batch dimension

# MAIN EXECUTION FLOW
if __name__ == "__main__":
    # Load and process test data
    path_final_test = "Test_Data.csv"
    files = [path_final_test]
    
    # Load BPM phase data and drop noisy devices
    dfs = load_BPMphase_data_multi(cavs_both, files, devices_to_drop, scan=False)
    
    # Normalize BPM data: (value - mean) / standard deviation
    tip_dfs = [(df[BPM_list] - df[BPM_list].mean()) / (df[BPM_list].std() + 1e-8) for df in dfs]
    
    # Convert to PyTorch tensors
    tip = [torch.tensor(df.values, dtype=torch.float32) for df in tip_dfs]
    
    # Physics model tensor
    response_matrix_tensor = torch.tensor(response_matrix, dtype=torch.float32)
    
    # Evaluate model on test data
    losses = []
    results = []
    
    for file_idx, test_data_tensor in enumerate(tip):
        for sample_idx in range(test_data_tensor.shape[0]):
            # Process single sample
            test_input_sample = test_data_tensor[sample_idx, :]
            
            # Predict cavity parameters from BPM readings
            predicted_output, input_tensor = predict_single_input(test_input_sample)
            
            # Reconstruct BPM readings using physics model
            # response_matrix_tensor[:, 7:] extracts the BPM response components
            predicted_trajectory = torch.matmul(predicted_output, response_matrix_tensor[:, 7:])
            
            # Calculate reconstruction error
            mse_loss = F.mse_loss(predicted_trajectory, input_tensor)
            relative_loss = mse_loss / (torch.norm(input_tensor) + 1e-8)
            
            # Store results
            losses.append(relative_loss.item())
            results.append((
                (file_idx, sample_idx), 
                relative_loss.item(),
                predicted_output,
                input_tensor
            ))
    
    # Find best and worst predictions
    results.sort(key=lambda x: x[1])
    best_original_idx, _, best_output, best_input = results[0]
    worst_original_idx, _, worst_output, worst_input = results[-1]
    
    # Visualize results
    print(f"Best Fit - Sample {best_original_idx[1]}")
    print(f"Output - {best_output.tolist()}")
    trajectory_fit(
        best_input.numpy(), 
        best_output, 
        response_matrix,
        save_path="Best_Sample_Fit.png"
    )
    
    print(f"Worst Fit - Sample {worst_original_idx[1]}")
    print(f"Output - {worst_output.tolist()}")
    trajectory_fit(
        worst_input.numpy(), 
        worst_output, 
        response_matrix,
        save_path="Worst_Sample_Fit.png"
    )