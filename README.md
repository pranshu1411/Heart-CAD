# Cardiovascular Disease Prediction using Machine Learning & ANFIS

## Overview

This project develops and compares multiple machine learning models for **Cardiovascular Disease (CAD) prediction**. It implements both classical machine learning baselines and an advanced **FA-OPT-SC-ANFIS (Firefly Algorithm Optimized Self-Constructing ANFIS)** approach to accurately predict the presence of cardiovascular disease across multiple datasets.

## Features

- **Multiple Datasets**: Cleveland, Statlog, and Indian Cardiovascular Disease datasets
- **Baseline Models**:
  - Logistic Regression
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - XGBoost
- **Advanced Model**: FA-OPT-SC-ANFIS (Firefly Algorithm optimized ANFIS)
- **Comprehensive Evaluation**: Accuracy, Precision, Recall, F1-score, ROC-AUC
- **Statistical Testing**: McNEmar's test for model comparison
- **Data Preprocessing**: Normalization and missing value handling
- **Visualization**: Feature importance heatmaps, ROC curves, confusion matrices

## Project Structure

```
CAD-SC/
├── original-data/                 # Raw datasets
│   ├── Cardiovascular_Disease_Dataset.csv
│   ├── cleveland.csv
│   └── statlog.csv
├── processed-data/                # Preprocessed and normalized datasets
│   ├── cleveland_processed.csv
│   ├── cleveland_normalised.csv
│   ├── indian_processed.csv
│   ├── indian_normalised.csv
│   ├── statlog_processed.csv
│   └── statlog_normalised.csv
├── results/                       # Model predictions and results
│   ├── anfis_predictions.csv      # ANFIS model predictions
│   ├── final_model_workspace.mat  # MATLAB workspace with ANFIS model
│   └── baseline_v1/               # Baseline model results
│       ├── *_predictions.csv
│       └── *_results.csv
├── analysis_images/               # Generated visualizations
│   ├── anfis_mfs/
│   ├── baseline_v1/
│   └── comparison/
├── scripts/
│   ├── processing.py              # Data preprocessing and normalization
│   ├── model_training.py          # Training baseline ML models
│   ├── statistical_tests.py       # McNEmar's test for model comparison
│   ├── heatmap_feature_importance.py
│   ├── distribution_analysis.py
│   ├── plot_all_csvs.py
│   ├── fa_hybrid_anfis.m          # ANFIS model implementation (MATLAB)
│   ├── run_fa_opt_anfis.m         # FA optimization script (MATLAB)
│   ├── plot_mfs.m                 # Plot membership functions (MATLAB)
│   └── extract_mat_to_csv.py      # Convert MATLAB results to CSV
├── LICENSE                        # MIT License
└── README.md                      # This file
```

## Installation & Setup

### Requirements

- **Python 3.8+**
- **MATLAB** (for ANFIS-based approaches)

### Python Dependencies

```bash
pip install -r requirements.txt
```

Key packages:
- pandas
- scikit-learn
- xgboost
- matplotlib
- statsmodels

### Environment Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd CAD-SC
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install pandas scikit-learn xgboost matplotlib statsmodels
```

## Usage

### 1. Data Preprocessing

Process and normalize the raw datasets:

```bash
cd scripts
python processing.py
```

This script:
- Loads Cleveland, Statlog, and Indian datasets
- Handles missing values
- Standardizes feature names across datasets
- Applies StandardScaler normalization
- Saves processed data to `../processed-data/`

### 2. Training Baseline Models

Train all baseline machine learning models:

```bash
python model_training.py
```

This script:
- Loads processed data
- Trains 5 baseline models (LR, SVM, RF, KNN, XGBoost)
- Uses Stratified K-Fold cross-validation
- Generates predictions and evaluation metrics
- Saves results to `../results/baseline_v1/`

### 3. ANFIS Model (MATLAB)

Run the Firefly Algorithm optimized ANFIS:

```matlab
run_fa_opt_anfis.m
```

This MATLAB script:
- Implements FA-OPT-SC-ANFIS
- Optimizes membership functions using Firefly Algorithm
- Generates predictions
- Saves results to `../results/anfis_predictions.csv`

### 4. Extract ANFIS Results

Convert MATLAB workspace to CSV:

```bash
python extract_mat_to_csv.py
```

### 5. Statistical Comparison

Compare ANFIS performance with baseline models using McNEmar's test:

```bash
python statistical_tests.py
```

This script:
- Performs McNEmar's test between ANFIS and each baseline model
- Tests if models have significantly different error rates
- Outputs statistical significance at p < 0.05 level

### 6. Visualization

Generate analysis plots and heatmaps:

```bash
python heatmap_feature_importance.py
python distribution_analysis.py
python plot_all_csvs.py
```

## Key Methodologies

### Data Preprocessing
- **Standardization**: Features normalized using StandardScaler
- **Missing Value Handling**: Removal of records with missing values
- **Feature Harmonization**: Consistent naming across three datasets

### Baseline Models
All models use:
- **Stratified K-Fold Cross-Validation**
- **Pipeline Architecture**: StandardScaler → Classifier
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

### FA-OPT-SC-ANFIS
- **Firefly Algorithm**: Meta-heuristic optimization for membership functions
- **Self-Constructing ANFIS**: Automatic fuzzy rule generation
- **Optimization**: Minimizes prediction error using FA

### Statistical Testing
- **McNEmar's Test**: Non-parametric test comparing two classifiers
- **Significance Level**: α = 0.05

## Results

Results are organized in `results/` directory:
- **Baseline Models**: Individual prediction CSVs and performance metrics
- **ANFIS Model**: Predictions and optimal workspace
- **Comparisons**: Images showing feature importance and ROC curves

## Features Description

The models use the following cardiovascular disease indicators:

| Feature | Description |
|---------|-------------|
| age | Patient age in years |
| sex | Gender (0=female, 1=male) |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure (mmHg) |
| chol | Serum cholesterol (mg/dl) |
| fbs | Fasting blood sugar > 120 mg/dl (0=no, 1=yes) |
| restecg | Resting electrocardiographic results (0-2) |
| thalach | Maximum heart rate achieved (bpm) |
| exang | Exercise induced angina (0=no, 1=yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of ST segment (0-2) |
| ca | Number of major vessels (0-4) |

## Citation

If you use this work in your research, please cite:

```bibtex
@project{CAD-SC,
  title={Cardiovascular Disease Prediction using Machine Learning & ANFIS},
  author={Pranshu Gupta},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request or open an Issue for bugs and feature suggestions.

## Contact

For questions or inquiries, please reach out through the project repository.

## Acknowledgments

- Cleveland Clinic Foundation for the Cleveland Heart Disease dataset
- University of California, Irvine Machine Learning Repository for Statlog dataset
- Open-source machine learning community (scikit-learn, XGBoost, etc.)
