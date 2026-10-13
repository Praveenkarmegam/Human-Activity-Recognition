# Dataset Directory

This directory is designated for the **UCI Human Activity Recognition Using Smartphones Dataset**.

## Dataset Source
- **Origin**: UCI Machine Learning Repository
- **URL**: [https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- **Direct Download**: `https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip`

## Signal Structure
The dataset contains 3-axial linear acceleration and 3-axial angular velocity captured at 50Hz by an embedded accelerometer and gyroscope in a smartphone worn on the waist:
- **Time steps per window**: 128 (sampled over 2.56 seconds with 50% overlap)
- **Channels (9 total)**:
  - `body_acc_x`, `body_acc_y`, `body_acc_z`
  - `total_acc_x`, `total_acc_y`, `total_acc_z`
  - `body_gyro_x`, `body_gyro_y`, `body_gyro_z`
- **Classes (6 activities)**:
  1. WALKING
  2. WALKING_UPSTAIRS
  3. WALKING_DOWNSTAIRS
  4. SITTING
  5. STANDING
  6. LAYING

## Automatic Download
When executing the training notebook in `notebooks/Human_Activity_Recognition.ipynb`, the dataset is downloaded and extracted automatically if not present locally.

*Note: Large `.zip` archives and extracted dataset folders are excluded from Git version control via `.gitignore`.*
