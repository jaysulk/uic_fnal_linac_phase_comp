
# Automated RF Phase Adjustment

## Project Structure

```
├── Train.py                 # Main training and fitting script
├── Test.py                  # Optional testing script
├── input.csv                # Input feature dataset (BPM Values)
├── output.csv               # Output labels (RF cavity settings)
├── response_matrix.csv      # System response matrix
├── sensor_positions.json    # BPM device position mapping
├── Test_Data.csv            # Optional testing data
├── environment.yml          # Conda environment configuration
├── Run.bat                  # Windows batch file for automated execution

```

## How to Run This Project

### Option 1: Using `Run.bat` (Windows Users Only)

1. **Set Conda Path**:
   - Edit `Run.bat` file.
   - Replace `CONDA_PATH` with the absolute path to your Anaconda installation (e.g., `C:\Users\<your-username>\anaconda3`).
   - Also ensure this path is added to your system **Environment Variables**.

2. **Double-click `Run.bat`**:
   - It checks for the environment and installs packages if needed.
   - It then executes `Train.py` and later shows options to either Retrain or Test or Exit/Stop execution.

---

### Option 2: Manual Setup via Anaconda Prompt

1. **Create and activate environment on Anaconda Prompt**:
   ```bash
   conda env create -f environment.yml
   conda activate dpienv
   ```
2. **Change the directory**:
   ```bash
   cd C:\(your-path-to-project-folder)\DPI_Automated
   ```
3. **Run the main training script**:
   ```bash
   python Train.py
   python Test.py
   ```

---

## Testing Custom Data

To test your own Test data:

- Replace the file `Test_Data.csv` with your own dataset using the same column format.
- Rerun the training or just testing script to apply the model to your dataset.

---

## Features

- Feedforward neural network with:
  - Regression & Classification Loss
  - Energy consistency loss
  - Temporal smoothness loss
  - Fit-loss based on predicted vs actual beam trajectory

---

## Outputs

- **Model Checkpoints**: Saved periodically in the format `checkpoint-epoch-{epoch}.pth`
- **Loss Curves**: Visualization of train/test losses over time
- **Best and Worst Fit Plots**: Actual beam trajectory vs predicted trajectory

---

## Requirements

- Python 3.10
- Torch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- SciPy

All packages will be installed automatically when using either the `environment.yml` or the `.bat` setup.