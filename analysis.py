"""
SMS Spam Detection Using Text Classification:
Comparing Support Vector Machines and Gradient Boosting

=======================================================
This script implements a complete NLP pipeline to classify SMS messages
as spam or ham (legitimate) using the UCI SMS Spam Collection dataset.
It compares two supervised learning approaches:
  1. TF-IDF + Support Vector Machine (SVM)
  2. TF-IDF + Gradient Boosting

Both models are tuned using GridSearchCV with 5-fold stratified
cross-validation to ensure robust and fair comparison.

Dataset: UCI SMS Spam Collection
  - Source: https://archive.ics.uci.edu/ml/datasets/sms+spam+collection
  - Also available: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
  - Download and unzip to get 'spam.csv', then place it in a /data folder

Author: [Your Name]
Date: March 2026
"""

# =============================================================================
# IMPORTS
# =============================================================================
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn imports for modelling and evaluation
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    RocCurveDisplay
)
from sklearn.pipeline import Pipeline

# Suppress convergence warnings for cleaner output
warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# =============================================================================
# STEP 1: LOAD AND EXPLORE THE DATA
# =============================================================================
print("=" * 65)
print("STEP 1: Loading and Exploring Data")
print("=" * 65)

# The Kaggle download provides spam.csv with columns v1 (label) and v2 (message).
# Extra unnamed columns are artefacts of commas in message text — we drop them.
df = pd.read_csv(
    os.path.join(DATA_DIR, "spam.csv"),
    encoding='latin-1'
)

# Rename columns and drop any junk columns from the CSV parsing
df = df.rename(columns={'v1': 'label', 'v2': 'message'})
df = df[['label', 'message']]

# Encode labels: ham = 0, spam = 1
df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

print(f"Total messages loaded: {len(df)}")
print(f"  Ham (legitimate): {(df['label_num'] == 0).sum()}")
print(f"  Spam:             {(df['label_num'] == 1).sum()}")
print(f"  Spam ratio:       {df['label_num'].mean():.2%}")

# Add word count feature for exploratory analysis
df['word_count'] = df['message'].apply(lambda x: len(str(x).split()))

print(f"\nAverage message length (words):")
print(f"  Ham:  {df[df['label_num']==0]['word_count'].mean():.1f}")
print(f"  Spam: {df[df['label_num']==1]['word_count'].mean():.1f}")


# =============================================================================
# FIGURE 1: Class Distribution
# =============================================================================
fig, ax = plt.subplots(figsize=(5.5, 4))
counts = df['label_num'].value_counts().sort_index()
labels_fig = ['Ham\n(Legitimate)', 'Spam']
colours = ['#3498db', '#e74c3c']
bars = ax.bar(labels_fig, counts.values, color=colours,
              edgecolor='black', linewidth=0.7, width=0.55)

# Add count labels above each bar
for bar, count in zip(bars, counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
            f'{count:,}', ha='center', va='bottom', fontweight='bold', fontsize=11)

ax.set_ylabel('Number of Messages', fontsize=11)
ax.set_title('Figure 1: Class Distribution of SMS Messages',
             fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
ax.set_ylim(0, counts.max() * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig1_class_distribution.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("\n[Saved] Figure 1: Class distribution")


# =============================================================================
# FIGURE 2: Message Length Distribution by Class
# =============================================================================
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(df[df['label_num'] == 0]['word_count'], bins=40, alpha=0.65,
        label='Ham', color='#3498db', edgecolor='black', linewidth=0.4)
ax.hist(df[df['label_num'] == 1]['word_count'], bins=40, alpha=0.65,
        label='Spam', color='#e74c3c', edgecolor='black', linewidth=0.4)

ax.set_xlabel('Word Count', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Figure 2: Message Length Distribution by Class',
             fontsize=12, fontweight='bold')
ax.legend(fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig2_length_distribution.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("[Saved] Figure 2: Message length distribution")


# =============================================================================
# STEP 2: TEXT PREPROCESSING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 2: Text Preprocessing")
print("=" * 65)

def clean_text(text):
    """
    Clean an SMS message by:
    - Converting to lowercase
    - Removing URLs
    - Removing non-alphabetic characters (keeps spaces)
    - Collapsing multiple spaces into one
    """
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', '', text)   # Remove URLs
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)        # Remove numbers/symbols
    text = re.sub(r'\s+', ' ', text).strip()         # Collapse whitespace
    return text

df['cleaned'] = df['message'].apply(clean_text)

# Show a few examples
print("Sample cleaned messages:")
for i in range(3):
    print(f"  [{df.iloc[i]['label']}] {df.iloc[i]['cleaned'][:80]}...")


# =============================================================================
# STEP 3: TRAIN-TEST SPLIT
# =============================================================================
print("\n" + "=" * 65)
print("STEP 3: Stratified Train-Test Split (80/20)")
print("=" * 65)

X = df['cleaned']
y = df['label_num']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {len(X_train)} messages")
print(f"Test set:     {len(X_test)} messages")
print(f"Train spam ratio: {y_train.mean():.2%}")
print(f"Test spam ratio:  {y_test.mean():.2%}")


# =============================================================================
# STEP 4: MODEL TRAINING WITH HYPERPARAMETER TUNING
# =============================================================================
print("\n" + "=" * 65)
print("STEP 4: Model Training with GridSearchCV (5-Fold Stratified CV)")
print("=" * 65)

# Define the cross-validation strategy used for ALL models
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# -------------------------------------------------------------------------
# MODEL 1: TF-IDF + Support Vector Machine (Linear SVM)
# -------------------------------------------------------------------------
print("\n--- Model 1: TF-IDF + Linear SVM ---")

svm_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', LinearSVC(random_state=42, max_iter=5000))
])

# Define a hyperparameter grid to search over
svm_param_grid = {
    'tfidf__max_df': [0.90, 0.95],           # Ignore terms in >90% or >95% of docs
    'tfidf__ngram_range': [(1, 1), (1, 2)],   # Unigrams vs unigrams+bigrams
    'tfidf__max_features': [5000, 10000],      # Vocabulary size cap
    'clf__C': [0.1, 1.0, 10.0],               # Regularisation strength
}

svm_grid = GridSearchCV(
    svm_pipeline, svm_param_grid,
    cv=cv, scoring='f1', n_jobs=-1, verbose=1, refit=True
)
svm_grid.fit(X_train, y_train)

print(f"\nBest hyperparameters: {svm_grid.best_params_}")
print(f"Best CV F1 score:     {svm_grid.best_score_:.4f}")

# Report 5-fold CV scores using the best estimator
svm_cv_scores = cross_val_score(
    svm_grid.best_estimator_, X_train, y_train, cv=cv, scoring='f1'
)
print(f"5-fold CV F1 scores:  {np.round(svm_cv_scores, 4)}")
print(f"Mean CV F1:           {svm_cv_scores.mean():.4f} "
      f"(\u00b1{svm_cv_scores.std():.4f})")

# Evaluate on the held-out test set
svm_pred = svm_grid.predict(X_test)

# SVM does not output probabilities natively; calibrate for ROC-AUC
calibrated_svm = CalibratedClassifierCV(
    svm_grid.best_estimator_, cv=3, method='sigmoid'
)
calibrated_svm.fit(X_train, y_train)
svm_proba = calibrated_svm.predict_proba(X_test)[:, 1]

svm_accuracy  = accuracy_score(y_test, svm_pred)
svm_precision = precision_score(y_test, svm_pred)
svm_recall    = recall_score(y_test, svm_pred)
svm_f1        = f1_score(y_test, svm_pred)
svm_auc       = roc_auc_score(y_test, svm_proba)

print(f"\n--- SVM Test Set Results ---")
print(f"  Accuracy:  {svm_accuracy:.4f}")
print(f"  Precision: {svm_precision:.4f}")
print(f"  Recall:    {svm_recall:.4f}")
print(f"  F1 Score:  {svm_f1:.4f}")
print(f"  AUC-ROC:   {svm_auc:.4f}")
print(f"\n{classification_report(y_test, svm_pred, target_names=['Ham', 'Spam'])}")


# -------------------------------------------------------------------------
# MODEL 2: TF-IDF + Gradient Boosting
# -------------------------------------------------------------------------
print("\n--- Model 2: TF-IDF + Gradient Boosting ---")

gb_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', GradientBoostingClassifier(random_state=42))
])

# Hyperparameter grid for Gradient Boosting
gb_param_grid = {
    'tfidf__max_df': [0.90, 0.95],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'tfidf__max_features': [5000, 10000],
    'clf__n_estimators': [100, 200],        # Number of boosting stages
    'clf__learning_rate': [0.05, 0.1],      # Step size shrinkage
    'clf__max_depth': [3, 5],               # Depth of each tree
}

gb_grid = GridSearchCV(
    gb_pipeline, gb_param_grid,
    cv=cv, scoring='f1', n_jobs=-1, verbose=1, refit=True
)
gb_grid.fit(X_train, y_train)

print(f"\nBest hyperparameters: {gb_grid.best_params_}")
print(f"Best CV F1 score:     {gb_grid.best_score_:.4f}")

# Report 5-fold CV scores
gb_cv_scores = cross_val_score(
    gb_grid.best_estimator_, X_train, y_train, cv=cv, scoring='f1'
)
print(f"5-fold CV F1 scores:  {np.round(gb_cv_scores, 4)}")
print(f"Mean CV F1:           {gb_cv_scores.mean():.4f} "
      f"(\u00b1{gb_cv_scores.std():.4f})")

# Evaluate on the held-out test set
gb_pred = gb_grid.predict(X_test)
gb_proba = gb_grid.predict_proba(X_test)[:, 1]

gb_accuracy  = accuracy_score(y_test, gb_pred)
gb_precision = precision_score(y_test, gb_pred)
gb_recall    = recall_score(y_test, gb_pred)
gb_f1        = f1_score(y_test, gb_pred)
gb_auc       = roc_auc_score(y_test, gb_proba)

print(f"\n--- Gradient Boosting Test Set Results ---")
print(f"  Accuracy:  {gb_accuracy:.4f}")
print(f"  Precision: {gb_precision:.4f}")
print(f"  Recall:    {gb_recall:.4f}")
print(f"  F1 Score:  {gb_f1:.4f}")
print(f"  AUC-ROC:   {gb_auc:.4f}")
print(f"\n{classification_report(y_test, gb_pred, target_names=['Ham', 'Spam'])}")


# =============================================================================
# STEP 5: MODEL COMPARISON AND VISUALISATIONS
# =============================================================================
print("\n" + "=" * 65)
print("STEP 5: Model Comparison and Visualisations")
print("=" * 65)

# -------------------------------------------------------------------------
# TABLE 1: Performance Summary
# -------------------------------------------------------------------------
results = pd.DataFrame({
    'Model': ['TF-IDF + Linear SVM', 'TF-IDF + Gradient Boosting'],
    'Accuracy': [svm_accuracy, gb_accuracy],
    'Precision': [svm_precision, gb_precision],
    'Recall': [svm_recall, gb_recall],
    'F1 Score': [svm_f1, gb_f1],
    'AUC-ROC': [svm_auc, gb_auc],
    'CV F1 Mean (±SD)': [
        f"{svm_cv_scores.mean():.4f} (\u00b1{svm_cv_scores.std():.4f})",
        f"{gb_cv_scores.mean():.4f} (\u00b1{gb_cv_scores.std():.4f})"
    ]
})
print("\nTable 1: Model Performance Comparison")
print(results.to_string(index=False))
results.to_csv(os.path.join(OUTPUT_DIR, 'table1_performance.csv'), index=False)
print("[Saved] Table 1: table1_performance.csv")


# -------------------------------------------------------------------------
# FIGURE 3: ROC Curves
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6.5, 5.5))

RocCurveDisplay.from_predictions(
    y_test, svm_proba,
    name=f"Linear SVM (AUC = {svm_auc:.3f})",
    ax=ax, color='#2980b9', linestyle='--', linewidth=2
)
RocCurveDisplay.from_predictions(
    y_test, gb_proba,
    name=f"Gradient Boosting (AUC = {gb_auc:.3f})",
    ax=ax, color='#e67e22', linestyle='-', linewidth=2
)

ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random Chance')
ax.set_title('Figure 3: ROC Curves \u2014 Model Comparison',
             fontsize=12, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig3_roc_curves.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("[Saved] Figure 3: ROC curves")


# -------------------------------------------------------------------------
# FIGURE 4: Confusion Matrices (side by side)
# -------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

for ax, preds, title in zip(
    axes,
    [svm_pred, gb_pred],
    ['Linear SVM', 'Gradient Boosting']
):
    cm = confusion_matrix(y_test, preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                cbar=False, annot_kws={'size': 14})
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)

fig.suptitle('Figure 4: Confusion Matrices', fontsize=13,
             fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig4_confusion_matrices.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("[Saved] Figure 4: Confusion matrices")


# -------------------------------------------------------------------------
# FIGURE 5: Top 20 Most Informative Features (from SVM coefficients)
# -------------------------------------------------------------------------
# Extract the TF-IDF vectoriser and SVM model from the best pipeline
best_svm = svm_grid.best_estimator_
tfidf_vec = best_svm.named_steps['tfidf']
svm_model = best_svm.named_steps['clf']

feature_names = np.array(tfidf_vec.get_feature_names_out())
coefficients = svm_model.coef_[0]

# Top 10 words most indicative of SPAM (highest positive coefficients)
top_spam_idx = np.argsort(coefficients)[-10:]
# Top 10 words most indicative of HAM (most negative coefficients)
top_ham_idx = np.argsort(coefficients)[:10]

top_idx = np.concatenate([top_ham_idx, top_spam_idx])
top_words = feature_names[top_idx]
top_coefs = coefficients[top_idx]

fig, ax = plt.subplots(figsize=(8, 5.5))
bar_colours = ['#3498db' if c < 0 else '#e74c3c' for c in top_coefs]
ax.barh(range(len(top_words)), top_coefs, color=bar_colours,
        edgecolor='black', linewidth=0.4)
ax.set_yticks(range(len(top_words)))
ax.set_yticklabels(top_words, fontsize=10)
ax.set_xlabel('SVM Coefficient Weight', fontsize=11)
ax.set_title('Figure 5: Top 20 Most Informative Words\n'
             '(Blue = Ham, Red = Spam)',
             fontsize=12, fontweight='bold')
ax.axvline(x=0, color='black', linewidth=0.8)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig5_feature_importance.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("[Saved] Figure 5: Feature importance (SVM coefficients)")


# -------------------------------------------------------------------------
# FIGURE 6: 5-Fold Cross-Validation F1 Score Comparison
# -------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(6, 4))

cv_data = [svm_cv_scores, gb_cv_scores]
bp = ax.boxplot(cv_data,
                labels=['Linear SVM', 'Gradient Boosting'],
                patch_artist=True, widths=0.45)
bp['boxes'][0].set_facecolor('#2980b9')
bp['boxes'][1].set_facecolor('#e67e22')
for box in bp['boxes']:
    box.set_alpha(0.7)
    box.set_edgecolor('black')

ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('Figure 6: 5-Fold Cross-Validation F1 Scores',
             fontsize=12, fontweight='bold')
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig6_cv_boxplot.png'),
            dpi=200, bbox_inches='tight')
plt.close()
print("[Saved] Figure 6: Cross-validation boxplot")


# =============================================================================
# STEP 6: FINAL SUMMARY
# =============================================================================
print("\n" + "=" * 65)
print("FINAL SUMMARY")
print("=" * 65)

# Determine best model
if svm_f1 >= gb_f1:
    best_name = "Linear SVM"
    best_f1 = svm_f1
    best_auc = svm_auc
else:
    best_name = "Gradient Boosting"
    best_f1 = gb_f1
    best_auc = gb_auc

print(f"\nBest performing model: {best_name}")
print(f"  F1 Score: {best_f1:.4f}")
print(f"  AUC-ROC:  {best_auc:.4f}")

print(f"\nSVM Best Hyperparameters:")
for k, v in svm_grid.best_params_.items():
    print(f"  {k}: {v}")

print(f"\nGradient Boosting Best Hyperparameters:")
for k, v in gb_grid.best_params_.items():
    print(f"  {k}: {v}")

print(f"\nAll figures saved to: {OUTPUT_DIR}/")
print("=" * 65)
print("DONE. Ready for submission.")
