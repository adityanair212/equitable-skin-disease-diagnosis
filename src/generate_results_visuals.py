import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve
import os

# Paths (update these if your paths are different)
CONFIDENCE_CSV = "confidence_summary.csv"
PREDICTIONS_CSV = "ensemble_predictions.csv"
RESULTS_DIR = "results_visuals"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Load data
confidence_df = pd.read_csv(CONFIDENCE_CSV)
preds_df = pd.read_csv(PREDICTIONS_CSV)

# Cleaned data: remove unknowns
valid_preds = confidence_df[confidence_df["predicted"] != "unknown"]
y_true = valid_preds["true_label"]
y_pred = valid_preds["predicted"]

# Class labels
labels = sorted(y_true.unique())

# 1. Confusion Matrix
cm = confusion_matrix(y_true, y_pred, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix.png"))
plt.close()

# 2. Classification Report
report = classification_report(y_true, y_pred, labels=labels, output_dict=True)
report_df = pd.DataFrame(report).transpose()
report_df.to_csv(os.path.join(RESULTS_DIR, "classification_report.csv"))
print("Saved classification report.")

# 3. Rejection Stats
rejected = confidence_df[confidence_df["predicted"] == "unknown"]
rejected_count = len(rejected)
total = len(confidence_df)
rejected_pct = rejected_count / total * 100
with open(os.path.join(RESULTS_DIR, "rejection_stats.txt"), "w") as f:
    f.write(f"Rejected: {rejected_count} out of {total} ({rejected_pct:.2f}%)\n")
print("Saved rejection stats.")

# 4. Precision/Recall per class (bar chart)
fig, ax = plt.subplots(figsize=(8, 5))
report_df.loc[labels, ["precision", "recall"]].plot(kind='bar', ax=ax)
plt.title("Per-Class Precision and Recall")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "precision_recall_barplot.png"))
plt.close()

# 5. Confidence Histogram
plt.figure(figsize=(6, 4))
sns.histplot(confidence_df["confidence"], bins=30, kde=True, color="teal")
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence Score")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "confidence_distribution.png"))
plt.close()

print("All visualizations saved to:", RESULTS_DIR)
