# SMS Spam Detection Using Text Classification

## Overview

This repository contains the code and analysis for Problem Set #2 (Using Text as Data). The project investigates whether NLP-based text classification can accurately distinguish spam from legitimate SMS messages, comparing a Support Vector Machine with a Gradient Boosting classifier — both trained on TF-IDF features and tuned via GridSearchCV with 5-fold stratified cross-validation.

## Research Question

**Can natural language processing methods accurately classify SMS messages as spam or legitimate (ham), and does a Gradient Boosting ensemble model outperform a linear Support Vector Machine on this task?**

## Dataset

**UCI SMS Spam Collection** (Almeida et al., 2011)

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Alternative**: [Kaggle — SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size**: 5,574 SMS messages (4,827 ham + 747 spam)
- **File needed**: `spam.csv` (downloaded from Kaggle as a ZIP, unzip to get the CSV)

## Repository Structure

```
├── analysis.py             # Main analysis script (preprocessing, modelling, figures)
├── data/
│   └── spam.csv                # Dataset file (download separately — see below)
├── outputs/                # Generated figures and results (created by script)
│   ├── fig1_class_distribution.png
│   ├── fig2_length_distribution.png
│   ├── fig3_roc_curves.png
│   ├── fig4_confusion_matrices.png
│   ├── fig5_feature_importance.png
│   ├── fig6_cv_boxplot.png
│   └── table1_performance.csv
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## How to Reproduce

### 1. Clone the repository

```bash
git clone https://github.com/[saaransh46-glitch]/sms-spam-detection.git
cd sms-spam-detection
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download the SMS Spam Collection from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) or [UCI](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection). Unzip the download and place `spam.csv` inside a `data/` folder:

```
data/
└── spam.csv
```

### 4. Run the analysis

```bash
python analysis.py
```

The script will:
1. Load and preprocess the SMS data
2. Train a TF-IDF + Linear SVM model with GridSearchCV hyperparameter tuning
3. Train a TF-IDF + Gradient Boosting model with GridSearchCV hyperparameter tuning
4. Evaluate both models using 5-fold stratified cross-validation
5. Generate all figures and a performance comparison table

All outputs are saved to the `outputs/` directory. Runtime is approximately 5–10 minutes on a standard laptop (no GPU required).

## Methods Summary

| Component | Detail |
|-----------|--------|
| Text representation | TF-IDF (unigrams and bigrams) |
| Model 1 | Linear Support Vector Machine |
| Model 2 | Gradient Boosting Classifier |
| Hyperparameter tuning | GridSearchCV (exhaustive grid search) |
| Cross-validation | 5-fold stratified CV |
| Evaluation metrics | Accuracy, Precision, Recall, F1, AUC-ROC |

## Dependencies

- Python 3.9+
- scikit-learn >= 1.3
- pandas >= 2.0
- numpy >= 1.24
- matplotlib >= 3.7
- seaborn >= 0.12

## Reference

Almeida, T. A., Gómez Hidalgo, J. M., and Yamakami, A. (2011). Contributions to the study of SMS spam filtering: new collection and results. *Proceedings of the 2011 ACM Symposium on Document Engineering*, pp. 259–262.
