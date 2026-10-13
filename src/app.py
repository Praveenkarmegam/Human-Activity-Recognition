"""
Standalone Gradio Web Application for Human Activity Recognition.
Loads the trained PyTorch 1D CNN checkpoint and allows interactive activity prediction
and optional Gemini API explanations.
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import gradio as gr

from src.predict import load_model, predict_activity

# Project paths
ROOT_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT_DIR / "models" / "har_cnn.pth"

# Attempt to load Gemini if API key is present
GEMINI_AVAILABLE = False
try:
    from google import genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        gemini_client = genai.Client(api_key=api_key)
        GEMINI_AVAILABLE = True
except Exception:
    GEMINI_AVAILABLE = False


def generate_synthetic_sample(activity_hint: str = "WALKING") -> np.ndarray:
    """Generate a realistic test sensor signal window (9 channels, 128 time steps)."""
    t = np.linspace(0, 2.56, 128)
    signals = np.zeros((9, 128), dtype=np.float32)

    # Base frequencies and amplitudes per activity type
    if "WALK" in activity_hint:
        freq = 1.8
        signals[0] = 0.4 * np.sin(2 * np.pi * freq * t) + np.random.normal(0, 0.05, 128)
        signals[1] = 0.6 * np.cos(2 * np.pi * freq * t) + np.random.normal(0, 0.05, 128)
        signals[2] = 0.3 * np.sin(4 * np.pi * freq * t) + np.random.normal(0, 0.05, 128)
    elif "SIT" in activity_hint or "STAND" in activity_hint:
        signals[0] = np.random.normal(0.02, 0.02, 128)
        signals[1] = np.random.normal(0.01, 0.02, 128)
        signals[2] = np.random.normal(0.98, 0.02, 128)
    elif "LAY" in activity_hint:
        signals[0] = np.random.normal(-0.8, 0.02, 128)
        signals[1] = np.random.normal(0.0, 0.02, 128)
        signals[2] = np.random.normal(0.0, 0.02, 128)

    # Gyroscope & total acceleration channels
    for ch in range(3, 9):
        signals[ch] = 0.2 * np.sin(2 * np.pi * 1.5 * t + ch) + np.random.normal(0, 0.04, 128)

    return signals


def explain_with_gemini(prediction: dict, signal_info: str) -> str:
    """Explain prediction using Gemini API if configured."""
    if not GEMINI_AVAILABLE:
        return (
            "Gemini explanations unavailable. To enable:\n"
            "Set the GEMINI_API_KEY environment variable before running the application."
        )

    prompt = f"""
You are an AI explanation assistant for an educational Human Activity Recognition system.
Explain the following 1D CNN prediction output:
Prediction: {prediction['predicted_activity']} (Confidence: {prediction['confidence']:.2%})
Class Probabilities: {json.dumps(prediction['class_probabilities'], indent=2)}
Signal Info: {signal_info}

Provide:
1. Activity summary
2. Confidence analysis
3. Sensor interpretation
4. Educational disclaimer (non-safety-critical prototype)
"""
    try:
        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=prompt
        )
        return response.text or "No explanation generated."
    except Exception as e:
        return f"Gemini request failed: {e}"


def launch_app():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model checkpoint not found at: {MODEL_PATH}")

    model, class_names = load_model(str(MODEL_PATH))

    def run_prediction(activity_preset: str, use_gemini: bool):
        sensor_window = generate_synthetic_sample(activity_preset)
        result = predict_activity(sensor_window, model, class_names)

        # Plot 3-axis body acceleration
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(sensor_window[0], label="X-axis (body_acc_x)", color="#2563eb")
        ax.plot(sensor_window[1], label="Y-axis (body_acc_y)", color="#16a34a")
        ax.plot(sensor_window[2], label="Z-axis (body_acc_z)", color="#dc2626")
        ax.set_title(f"3-Axis Body Acceleration ({activity_preset})")
        ax.set_xlabel("Time Step (128 samples / 2.56s)")
        ax.set_ylabel("Normalized Acceleration")
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()

        summary = (
            f"Predicted Activity: {result['predicted_activity']}\n"
            f"Confidence: {result['confidence']:.2%}\n"
            f"Second Best: {result['second_best_activity']} ({result['second_best_confidence']:.2%})"
        )

        explanation = "Gemini explanation not requested."
        if use_gemini:
            explanation = explain_with_gemini(result, f"Preset: {activity_preset}")

        prob_output = {k: round(v, 4) for k, v in result["class_probabilities"].items()}
        return prob_output, summary, fig, explanation

    demo = gr.Interface(
        fn=run_prediction,
        inputs=[
            gr.Dropdown(
                choices=class_names,
                value="WALKING",
                label="Sample Activity Simulation Pattern"
            ),
            gr.Checkbox(
                value=False,
                label="Generate Gemini AI Explanation (requires GEMINI_API_KEY)"
            )
        ],
        outputs=[
            gr.Label(num_top_classes=6, label="Activity Probabilities"),
            gr.Textbox(label="Prediction Summary", lines=3),
            gr.Plot(label="Sensor Signal Visualization"),
            gr.Textbox(label="Gemini Explanation", lines=8)
        ],
        title="Human Activity Recognition (1D CNN)",
        description=(
            "Deep Learning system for classifying human activities from smartphone "
            "inertial sensor data (accelerometer & gyroscope)."
        )
    )

    return demo


if __name__ == "__main__":
    demo = launch_app()
    demo.launch()
