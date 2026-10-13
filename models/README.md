# Trained Models

This directory stores trained model checkpoints for the Human Activity Recognition system.

## Checkpoint Files
- `har_cnn.pth`: PyTorch checkpoint of the trained 1D Convolutional Neural Network (CNN).

## Checkpoint Contents
The `.pth` checkpoint dictionary contains:
- `model_state_dict`: Learned network weights and biases.
- `class_names`: List of 6 target activity labels:
  `["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]`
- `input_channels`: `9` (3 body acc + 3 total acc + 3 body gyro)
- `sequence_length`: `128` time steps
- `best_epoch`: Best validation epoch index saved during training.

## Loading the Model in Python
```python
import torch
from src.predict import load_model

model, class_names = load_model("models/har_cnn.pth")
```
