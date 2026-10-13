# Human Activity Recognition Using Deep Learning

A deep learning system for Human Activity Recognition (HAR) that classifies six daily human activities from smartphone accelerometer and gyroscope data using a 1D Convolutional Neural Network (CNN). The project includes an interactive Gradio web application and an optional Google Gemini API integration for natural language explanation of predictions.

---

## Project Structure

```text
Human-Activity-Recognition/
├── .venv/                      # Python virtual environment (local)
├── data/                       # Dataset directory & documentation
│   └── README.md               # Dataset source & acquisition notes
├── models/                     # Trained model checkpoints
│   ├── har_cnn.pth             # Trained 1D CNN PyTorch weights
│   └── README.md               # Model checkpoint details
├── notebooks/                  # Interactive experimentation & training
│   └── Human_Activity_Recognition.ipynb
├── results/                    # Quantitative evaluation artifacts
│   ├── classification_report.csv
│   ├── confusion_matrix.csv
│   ├── metrics.json
│   └── README.md               # Evaluation performance summary
├── src/                        # Modular Python source code
│   ├── __init__.py             # Package init
│   ├── model.py                # PyTorch 1D CNN architecture definition
│   ├── predict.py              # Inference pipeline & checkpoint loading
│   └── app.py                  # Standalone Gradio web application
├── .gitignore                  # Git ignore rules
├── requirements.txt            # Python project dependencies
└── README.md                   # Project documentation
```

---

## Activities Classified

1. **Walking**
2. **Walking Upstairs**
3. **Walking Downstairs**
4. **Sitting**
5. **Standing**
6. **Laying Down**

---

## Tech Stack

- **Deep Learning**: PyTorch (`torch`, `torch.nn`)
- **Data & Scientific Computing**: NumPy, Pandas, Scikit-learn
- **Visualization**: Matplotlib, Seaborn
- **Interactive UI**: Gradio
- **LLM Reasoning**: Google GenAI SDK (Gemini API)
- **Environment**: Python 3.10+ / Virtualenv

---

## Dataset

- **UCI Human Activity Recognition Using Smartphones Dataset**
- **Link**: [UCI Machine Learning Repository #240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Sensor Input**: 9 inertial sensor channels (`body_acc_x/y/z`, `total_acc_x/y/z`, `body_gyro_x/y/z`) across 128 time steps per window (sampled at 50 Hz over 2.56 seconds).

---

## Model Architecture

1. **Input**: Tensor shape `(batch_size, 9, 128)`
2. **Conv Block 1**: 1D Conv (32 filters, kernel 7, padding 3) + ReLU + MaxPool1d (kernel 2)
3. **Conv Block 2**: 1D Conv (64 filters, kernel 5, padding 2) + ReLU + MaxPool1d (kernel 2)
4. **Global Pooling**: Adaptive Average Pooling `(AdaptiveAvgPool1d(1))`
5. **Classifier**: Dropout ($p=0.3$) + Fully Connected layer $(64 \to 6)$

---

## Evaluation Benchmark

| Metric | Result |
| :--- | :--- |
| **Test Accuracy** | **90.74%** |
| **Macro F1-Score** | **0.9075** |
| **Weighted F1-Score** | **0.9066** |
| **Baseline Accuracy (Majority Class)** | **18.22%** |
| **Net Accuracy Gain** | **+72.52%** |

Full per-class precision, recall, and confusion matrix are saved in [`results/`](results/).

---

## Getting Started

### 1. Set Up the Python Virtual Environment

A dedicated virtual environment has been created at `.venv`. To activate it:

**On Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
```

**On Windows (Command Prompt):**
```cmd
.\.venv\Scripts\activate.bat
```

**On Linux / macOS:**
```bash
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Interactive Web Application

Launch the standalone Gradio UI:
```bash
python -m src.app
```
*(Optional: set `GEMINI_API_KEY` in your environment to enable AI explanations).*

### 4. Run or Train via Notebook

Open the training and evaluation notebook in VS Code or Jupyter:
```bash
jupyter notebook notebooks/Human_Activity_Recognition.ipynb
```
Or open [`notebooks/Human_Activity_Recognition.ipynb`](notebooks/Human_Activity_Recognition.ipynb) in Google Colab.

---

## Reproducibility & Notes

- Checkpoint weights are saved under [`models/har_cnn.pth`](models/har_cnn.pth).
- Large dataset archives and virtual environments are excluded from Git via [`.gitignore`](.gitignore).
