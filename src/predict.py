import os
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import torch

from .model import HAR_CNN

DEFAULT_CLASS_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]


def load_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None
) -> tuple[HAR_CNN, list[str]]:
    """
    Load the trained HAR_CNN model from a checkpoint file.
    """
    if checkpoint_path is None:
        # Default to models/har_cnn.pth relative to repo root
        root_dir = Path(__file__).resolve().parent.parent
        checkpoint_path = str(root_dir / "models" / "har_cnn.pth")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    
    model = HAR_CNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, class_names


def predict_activity(
    sensor_window: np.ndarray,
    model: HAR_CNN,
    class_names: list[str],
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """
    Predict activity class from a single 9x128 sensor window.

    Args:
        sensor_window: NumPy array with shape (9, 128)
        model: Trained HAR_CNN instance
        class_names: List of activity labels
        device: Torch device

    Returns:
        Dictionary containing predicted class, confidence, and full probabilities.
    """
    sensor_window = np.asarray(sensor_window, dtype=np.float32)

    if sensor_window.shape != (9, 128):
        raise ValueError(f"Expected shape (9, 128), got {sensor_window.shape}")

    if not np.isfinite(sensor_window).all():
        raise ValueError("Input contains NaN or infinite values.")

    if device is None:
        device = next(model.parameters()).device

    model.eval()
    x = torch.from_numpy(sensor_window).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    order = np.argsort(probs)[::-1]
    predicted_id = int(order[0])

    return {
        "predicted_activity": class_names[predicted_id],
        "predicted_class_id": predicted_id,
        "confidence": float(probs[predicted_id]),
        "second_best_activity": class_names[int(order[1])],
        "second_best_confidence": float(probs[order[1]]),
        "class_probabilities": {
            class_names[i]: float(probs[i])
            for i in range(len(class_names))
        }
    }


if __name__ == "__main__":
    print("Loading HAR_CNN checkpoint...")
    model, classes = load_model()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully! Trainable parameters: {total_params:,}")
    print(f"Target activity classes: {classes}")

    # Generate a dummy test window (9 inertial sensor channels x 128 time steps)
    sample_window = np.random.randn(9, 128).astype(np.float32)
    output = predict_activity(sample_window, model, classes)

    print("\n--- Test Sample Inference Output ---")
    print(f"Predicted Activity: {output['predicted_activity']}")
    print(f"Confidence:         {output['confidence']:.2%}")
    print(f"Second Best:        {output['second_best_activity']} ({output['second_best_confidence']:.2%})")
    print("\nFull Class Probabilities:")
    for activity, prob in output["class_probabilities"].items():
        print(f"  {activity:<22}: {prob:.4f}")

