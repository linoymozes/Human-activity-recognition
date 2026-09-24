# Human Activity Recognition Using Smartphone Accelerometer Data

## Overview

This project develops a machine learning pipeline for **Human Activity Recognition (HAR)** using smartphone accelerometer data.

The goal is to identify different types of human movement from raw acceleration signals recorded along the X, Y, and Z axes.

The project covers the complete machine learning workflow, including data exploration, signal preprocessing, feature engineering, model training, evaluation, and experimentation with unknown activity detection.

---

## Activities

The dataset contains recordings representing several human activities, including:

- No movement
- Walking
- Running
- Stair movement

The recordings also include different movement patterns and intensities, such as steady and faster activity variations.

---

## Project Pipeline

The project follows an end-to-end machine learning workflow:

**Accelerometer Recordings → Data Quality Analysis → EDA → Preprocessing → Window Segmentation → Feature Engineering → Model Training → Evaluation**

### 1. Exploratory Data Analysis

The accelerometer recordings were analyzed to understand their structure and signal characteristics.

The analysis included:

- Missing-value and structural validation
- Timestamp consistency checks
- Duplicate recording detection
- Recording-duration analysis
- Sampling-rate analysis
- Time-series visualization
- Distribution and outlier analysis
- Correlation analysis
- Frequency-domain and cadence exploration
- Activity-level signal comparison

Exact duplicate recordings were identified using file hashing and excluded from the cleaned analysis dataset.

---

### 2. Signal Preprocessing

The recordings were transformed into fixed-duration time windows to enable consistent analysis of time-series signals.

An important challenge was the presence of recordings with different sampling rates. The preprocessing pipeline therefore considers the temporal duration of the signals rather than assuming that every recording contains the same number of samples per second.

For models requiring fixed-length input, the signals were resampled to a consistent representation.

---

### 3. Feature Engineering

A broad set of time-domain and frequency-domain features was extracted from the accelerometer windows.

Examples include:

- Mean, standard deviation, median, minimum and maximum
- RMS and signal magnitude
- Median Absolute Deviation (MAD)
- Interquartile range
- Jerk-based features
- Axis correlations
- Signal entropy
- Peak-related features
- Autocorrelation
- Dominant frequency
- Frequency-band power
- Spectral centroid and spectral spread

Highly correlated features were also analyzed to reduce redundant information.

---

## Machine Learning Models

Several machine learning approaches were explored and compared:

### Random Forest

A Random Forest classifier was trained using the engineered features.

The workflow included model evaluation, feature-importance analysis, and hyperparameter optimization.

### XGBoost

An XGBoost classifier was also trained on the engineered feature representation.

Hyperparameter optimization was performed using **Optuna**.

### MiniRocket

MiniRocket was used as an alternative time-series classification approach, allowing the model to learn representations directly from the segmented accelerometer signals.

This provided a useful comparison between traditional feature-based machine learning and a dedicated time-series classification approach.

---

## Unknown Activity Detection

The project also explores methods for detecting activity patterns that do not confidently belong to one of the known activity classes.

Two approaches were investigated:

- Prediction-confidence based rejection
- Isolation Forest anomaly detection

This component is treated as an experimental extension of the main HAR classification pipeline.

---

## Acceleration and Deceleration Analysis

An additional analysis was developed to identify periods of:

- Acceleration
- Stable movement
- Deceleration

The approach was also evaluated on interval-running accelerometer data to examine changes in movement intensity over time.

---

## Technologies

The project was implemented in Python using tools and libraries including:

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- XGBoost
- MiniRocket
- Optuna
- Matplotlib
- Seaborn
- Jupyter / Google Colab

---

## Key Challenges

Several practical time-series machine learning challenges were encountered during the project:

- Duplicate sensor recordings
- Different recording durations
- Multiple sampling rates
- Class imbalance
- High-dimensional feature engineering
- Time-series segmentation
- Distinguishing similar movement patterns
- Detecting potentially unknown activity patterns

---

## Results

The experiments demonstrated that accelerometer signals contain meaningful patterns that can be used to distinguish between different human activities.

The models achieved strong classification performance for several activities, particularly clearly distinguishable movement patterns such as running and stationary behavior.

Some closely related activities remained more challenging to distinguish, highlighting the difficulty of fine-grained Human Activity Recognition from accelerometer signals alone.

---

## Limitations and Future Work

Potential extensions of the project include:

- Recording-level grouped train/test splitting for stricter generalization evaluation
- Evaluation on additional participants and devices
- Collection of dedicated unknown-activity recordings
- Further sampling-rate normalization
- Additional sensor modalities such as gyroscope data
- Deep-learning approaches for end-to-end time-series classification

---

## Notebook

The complete analysis, preprocessing pipeline, model development, and experimental results are available in:

`HAR_BEST_TEAM_NOTEBOOK_FINALE.ipynb`

---

## Authors

Developed as part of a Human Activity Recognition machine learning project.
