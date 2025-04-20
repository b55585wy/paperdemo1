# CNNMamba Sleep Stage Classification

This project implements a sleep stage classification model using CNNMamba, a hybrid CNN-Mamba architecture. The model is designed to classify sleep into 5 stages (Wake, N1, N2, N3, REM) using EEG signals.

## Features

- Hybrid CNN-Mamba architecture for time series classification
- 1D adaptation of the CNNMamba for EEG data processing
- Bidirectional Mamba for capturing dependencies in both directions
- Channel and spatial attention mechanisms for feature enhancement
- Supports K-fold cross-validation for robust evaluation

## Requirements

- Python 3.8+
- PyTorch 2.0+
- mamba-ssm 1.0+
- Additional dependencies in `requirements.txt`

## Installation

```bash
# Clone the repository
git clone https://github.com/username/cnnmamba-sleep-classification.git
cd cnnmamba-sleep-classification

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Data Preparation

Place your sleep EEG data in the `data/` directory. The data should be in `.npz` format with the following structure:
- `x`: EEG signal data with shape [N, W, C] (segments, time steps, channels)
- `y`: Sleep stage labels (0: Wake, 1: N1, 2: N2, 3: N3, 4: REM)
- `fs`: Sampling frequency

### Configuration

Configure the model and training parameters in `hyperparameters.yaml`:

```yaml
data:
  data_dir: "./data/"

preprocess:
  sequence_epochs: 20  # Number of consecutive 30-second segments per sample
  enhance_window_stride: 5  # Stride for data augmentation
  normalize: true

model:
  time_steps: 3000  # Fixed time steps
  hidden_dim: 64  # Hidden dimension
  num_classes: 5  # Five-class classification
  num_blocks: 2  # Number of MambaSS1D blocks

training:
  batch_size: 8
  epochs: 50
  lr: 0.001
  patience: 10  # Early stopping patience
  k_folds: 10  # Number of folds for cross-validation
```

### Training

Run the training script:

```bash
python train.py
```

This will train the model using K-fold cross-validation and save the best model for each fold in the `models/` directory.

## Model Architecture

The model consists of:
1. A CNN-based feature extractor for time-domain features
2. Positional encoding for sequence awareness
3. Multiple MambaSS1D blocks for temporal modeling
4. Classification head for sleep stage prediction

The MambaSS1D module is a 1D adaptation of the SS2D architecture from CNNMamba, optimized for EEG signal processing.

## License

[MIT License](LICENSE) 