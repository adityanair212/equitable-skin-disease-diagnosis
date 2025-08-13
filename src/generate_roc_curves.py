import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import os

# Setup
PRED_CSV = "ensemble_predictions.csv"
SOFTMAX_EFF = "effnet_softmax.npy"
SOFTMAX_DENSE = "densenet_softmax.npy"
SOFTMAX_MOBILE = "mobilenet_softmax.npy"
RESULTS_DIR = "results_visuals"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Class mapping (assumes 0-4 as in your project)
class_names = ["basal-cell-carcinoma", "eczema", "melanocytic-nevi", "seborrheic-keratosis", "verruca-vulgaris"]
n_classes = len(class_names)

# Load files
df_preds = pd.read_csv(PRED_CSV)
y_true = df_preds["true_label"].values

# Load softmax outputs and ensemble average
softmax_eff = np.load(SOFTMAX_EFF)
softmax_dense = np.load(SOFTMAX_DENSE)
softmax_mobile = np.load(SOFTMAX_MOBILE)
ensemble_probs = (softmax_eff + softmax_dense + softmax_mobile) / 3

# Binarize labels
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

# Calculate ROC curves and AUCs
fpr, tpr, roc_auc = {}, {}, {}
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], ensemble_probs[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Macro average
fpr["macro"], tpr["macro"], _ = roc_curve(y_true_bin.ravel(), ensemble_probs.ravel())
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot ROC curves
plt.figure(figsize=(10, 7))
colors = sns.color_palette("husl", n_classes)
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], lw=2, label=f"{class_names[i]} (AUC = {roc_auc[i]:.2f})", color=color)

plt.plot(fpr["macro"], tpr["macro"], linestyle="--", lw=2, label=f"Macro Avg (AUC = {roc_auc['macro']:.2f})", color="black")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves for Ensemble Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "roc_curves.png"))
print("✅ ROC curves saved to:", os.path.join(RESULTS_DIR, "roc_curves.png"))
