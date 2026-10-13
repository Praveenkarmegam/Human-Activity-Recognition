# Evaluation Results & Metrics

This directory contains evaluation outputs and benchmark results for the trained 1D CNN model on the held-out test split of the UCI HAR dataset.

## Summary Metrics
- **Test Accuracy**: 90.74%
- **Macro F1-Score**: 0.9075
- **Weighted F1-Score**: 0.9066
- **Majority-Class Baseline**: 18.22% (Absolute improvement: +72.52%)

## Files in this Directory
1. `metrics.json`: Machine-readable JSON summary of core test metrics.
2. `classification_report.csv`: Per-class precision, recall, F1-score, and support counts.
3. `confusion_matrix.csv`: Full 6x6 confusion matrix across all test activity windows.

## Per-Class Performance Summary
| Activity | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **WALKING** | 98.36% | 96.98% | 0.9766 | 496 |
| **WALKING_UPSTAIRS** | 89.29% | 99.15% | 0.9396 | 471 |
| **WALKING_DOWNSTAIRS** | 98.96% | 90.95% | 0.9479 | 420 |
| **SITTING** | 83.61% | 70.67% | 0.7660 | 491 |
| **STANDING** | 77.09% | 86.65% | 0.8159 | 532 |
| **LAYING** | 100.00% | 99.81% | 0.9991 | 537 |
