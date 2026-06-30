"""
============================================================
 Binary LSTM Graphs — DB-HLSTM Framework
 Real Electrophysiology Data — 200 Cells

 Input  : /kaggle/working/binary_real_log.csv
          /kaggle/working/lstm_real_binary.keras
          /kaggle/input/datasets/sajidkhan1214/boldrealdata/Neural_BOLD_Data.csv

 Output : /kaggle/working/binary_graphs/
            Graph1_accuracy.png
            Graph2_loss.png
            Graph3_auc.png
            Graph4_precision_recall.png
            Graph5_confusion_matrix.png
            Graph6_roc_curve.png
            Graph7_summary.png
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             precision_recall_curve)
import tensorflow as tf
import seaborn as sns
import gc
import os
from tqdm.notebook import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
LOG_CSV    = '/kaggle/working/binary_real_log.csv'
MODEL_PATH = '/kaggle/working/lstm_real_binary.keras'
CSV_PATH   = '/kaggle/input/datasets/sajidkhan1214/boldrealdata/Neural_BOLD_Data.csv'
OUT_DIR    = '/kaggle/working/binary_graphs'

os.makedirs(OUT_DIR, exist_ok=True)
print(f"  Output folder: {OUT_DIR}")

SEQ_LEN     = 600
WINDOW      = 100
STRIDE      = 10
N_CLASSES   = 20
MIN_PER_CLS = 500
MAX_PER_CLS = 500
BATCH_SIZE  = 256

# ============================================================
# STEP 1 — LOAD HISTORY
# ============================================================
print("\n  Loading training history ...")
log = pd.read_csv(LOG_CSV)
print(f"  Epochs trained   : {len(log)}")
print(f"  Best Val AUC     : {log['val_auc'].max():.4f}")
print(f"  Best Val Accuracy: {log['val_accuracy'].max()*100:.2f}%")
print(f"  Best Epoch       : {log['val_auc'].idxmax()}")

# ============================================================
# STEP 2 — REBUILD TEST DATA
# ============================================================
print("\n  Rebuilding test data ...")
df = pd.read_csv(CSV_PATH)

# Binary label — same as training
df['Binary'] = (df['Class'] != 0).astype(int)

# KMeans for balancing — same as training
cvs           = df['I_val'].values.reshape(-1, 1)
kmeans        = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=10)
df['Class20'] = kmeans.fit_predict(cvs)

dfs = []
for cls in range(N_CLASSES):
    cls_df = df[df['Class20'] == cls]
    if len(cls_df) < MIN_PER_CLS:
        cls_df = cls_df.sample(MIN_PER_CLS, replace=True,  random_state=42)
    else:
        cls_df = cls_df.sample(MAX_PER_CLS, replace=False, random_state=42)
    dfs.append(cls_df)

df_bal = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
del df; gc.collect()

bold_cols   = [f'bold_{i}'   for i in range(SEQ_LEN)]
hrf_c_cols  = [f'hrf_c_{i}'  for i in range(SEQ_LEN)]
hrf_td_cols = [f'hrf_td_{i}' for i in range(SEQ_LEN)]
hrf_dd_cols = [f'hrf_dd_{i}' for i in range(SEQ_LEN)]
v_cols      = [f'v_{i}'      for i in range(SEQ_LEN)]

n_windows = (SEQ_LEN - WINDOW) // STRIDE + 1
n_samples = len(df_bal) * n_windows

X_bold   = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_c  = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_td = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_hrf_dd = np.zeros((n_samples, WINDOW), dtype=np.float32)
X_volt   = np.zeros((n_samples, WINDOW), dtype=np.float32)
y_bin    = np.zeros(n_samples, dtype=np.int32)

idx = 0
for _, row in tqdm(df_bal.iterrows(), total=len(df_bal), desc="Windowing"):
    bold   = row[bold_cols].values.astype(np.float32)
    hrf_c  = row[hrf_c_cols].values.astype(np.float32)
    hrf_td = row[hrf_td_cols].values.astype(np.float32)
    hrf_dd = row[hrf_dd_cols].values.astype(np.float32)
    volt   = row[v_cols].values.astype(np.float32)
    lb     = int(row['Binary'])

    for start in range(0, SEQ_LEN - WINDOW + 1, STRIDE):
        end = start + WINDOW
        X_bold[idx]   = bold[start:end]
        X_hrf_c[idx]  = hrf_c[start:end]
        X_hrf_td[idx] = hrf_td[start:end]
        X_hrf_dd[idx] = hrf_dd[start:end]
        X_volt[idx]   = volt[start:end]
        y_bin[idx]    = lb
        idx += 1

del df_bal; gc.collect()
print(f"  Windowing complete: {idx:,} samples")

# Normalize
X_ts = np.stack([X_bold, X_hrf_c, X_hrf_td, X_hrf_dd, X_volt], axis=2)
del X_bold, X_hrf_c, X_hrf_td, X_hrf_dd; gc.collect()

X_norm = np.zeros_like(X_ts, dtype=np.float32)
for fi in range(5):
    feat = X_ts[:,:,fi]
    X_norm[:,:,fi] = (feat - feat.mean(axis=1, keepdims=True)) / \
                     (feat.std(axis=1, keepdims=True) + 1e-8)
del X_ts; gc.collect()

sc = np.column_stack([
    X_volt.mean(axis=1),
    X_volt.std(axis=1),
    X_volt.min(axis=1),
    X_volt.max(axis=1),
    (X_volt > 0).sum(axis=1),
    X_volt.max(axis=1) - X_volt.min(axis=1),
    np.abs(X_volt).mean(axis=1),
    X_volt.std(axis=1) / (X_volt.mean(axis=1) + 1e-8),
]).astype(np.float32)

X_sc = StandardScaler().fit_transform(sc).astype(np.float32)
del X_volt, sc; gc.collect()

# Test split — same as training
n      = idx
val_n  = int(n * 0.15)
test_n = int(n * 0.10)
tr_n   = n - val_n - test_n

X_te    = X_norm[tr_n + val_n:]
X_sc_te = X_sc[tr_n + val_n:]
y_te    = y_bin[tr_n + val_n:]
del X_norm, X_sc, y_bin; gc.collect()

print(f"  Test samples: {len(y_te):,}")
print(f"  Regular  (0): {(y_te==0).sum():,}")
print(f"  Irregular(1): {(y_te==1).sum():,}")

# ============================================================
# STEP 3 — LOAD MODEL + PREDICT
# ============================================================
print("\n  Loading model and predicting ...")
model = tf.keras.models.load_model(MODEL_PATH)

AUTOTUNE = tf.data.AUTOTUNE
te_ds = (tf.data.Dataset
         .from_tensor_slices({'ts_input': X_te, 'scalar_input': X_sc_te})
         .batch(BATCH_SIZE).prefetch(AUTOTUNE))

y_prob = model.predict(te_ds, verbose=1).flatten()
y_pred = (y_prob >= 0.5).astype(int)
del X_te, X_sc_te; gc.collect()

overall_acc = (y_pred == y_te).mean() * 100
best_ep     = log['val_auc'].idxmax()

print(f"  Test Accuracy: {overall_acc:.2f}%")

# ============================================================
# GRAPH 1 — Accuracy per Epoch
# ============================================================
print("\n  Saving Graph 1 — Accuracy per Epoch ...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.plot(log['epoch'], log['accuracy'],     label='Train', color='#1F4E79', linewidth=2)
ax.plot(log['epoch'], log['val_accuracy'], label='Val',   color='#FF8C00', linewidth=2, linestyle='--')
ax.axvline(x=best_ep, color='green', linestyle=':', linewidth=1.5, label=f'Best Epoch={best_ep}')
ax.annotate(f"Best: {log['val_accuracy'].max()*100:.2f}%",
            xy=(best_ep, log['val_accuracy'].max()),
            xytext=(best_ep + 2, log['val_accuracy'].max() - 0.05),
            fontsize=10, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='green'))
ax.set_title('Accuracy per Epoch — Binary LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Accuracy', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_facecolor('#F0F6FF')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph1_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph1_accuracy.png")

# ============================================================
# GRAPH 2 — Loss per Epoch
# ============================================================
print("  Saving Graph 2 — Loss per Epoch ...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.plot(log['epoch'], log['loss'],     label='Train Loss', color='#1F4E79', linewidth=2)
ax.plot(log['epoch'], log['val_loss'], label='Val Loss',   color='#FF8C00', linewidth=2, linestyle='--')
ax.axvline(x=best_ep, color='green', linestyle=':', linewidth=1.5, label=f'Best Epoch={best_ep}')
ax.set_title('Loss per Epoch — Binary LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Loss', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_facecolor('#F0F6FF')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph2_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph2_loss.png")

# ============================================================
# GRAPH 3 — AUC per Epoch
# ============================================================
print("  Saving Graph 3 — AUC per Epoch ...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.plot(log['epoch'], log['auc'],     label='Train AUC', color='#1F4E79', linewidth=2)
ax.plot(log['epoch'], log['val_auc'], label='Val AUC',   color='#FF8C00', linewidth=2, linestyle='--')
ax.axvline(x=best_ep, color='green', linestyle=':', linewidth=1.5, label=f'Best Epoch={best_ep}')
ax.annotate(f"Best AUC: {log['val_auc'].max():.4f}",
            xy=(best_ep, log['val_auc'].max()),
            xytext=(best_ep + 2, log['val_auc'].max() - 0.03),
            fontsize=10, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='green'))
ax.set_title('AUC per Epoch — Binary LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('AUC', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3)
ax.set_facecolor('#F0F6FF')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph3_auc.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph3_auc.png")

# ============================================================
# GRAPH 4 — Precision & Recall per Epoch
# ============================================================
print("  Saving Graph 4 — Precision & Recall per Epoch ...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.plot(log['epoch'], log['precision'],     label='Train Precision', color='#1F4E79', linewidth=2)
ax.plot(log['epoch'], log['val_precision'], label='Val Precision',   color='#375623', linewidth=2, linestyle='--')
ax.plot(log['epoch'], log['recall'],        label='Train Recall',    color='#C00000', linewidth=2)
ax.plot(log['epoch'], log['val_recall'],    label='Val Recall',      color='#FF8C00', linewidth=2, linestyle='--')
ax.axvline(x=best_ep, color='green', linestyle=':', linewidth=1.5, label=f'Best Epoch={best_ep}')
ax.set_title('Precision & Recall per Epoch — Binary LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Score', fontsize=11)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
ax.set_facecolor('#F0F6FF')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph4_precision_recall.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph4_precision_recall.png")

# ============================================================
# GRAPH 5 — Confusion Matrix
# ============================================================
print("  Saving Graph 5 — Confusion Matrix ...")
cm      = confusion_matrix(y_te, y_pred)
cm_pct  = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#F8FBFF')
sns.heatmap(cm, annot=False, ax=ax, cmap='Blues', fmt='d')
for i in range(2):
    for j in range(2):
        val   = int(cm[i, j])
        color = 'white' if cm_pct[i, j] > 50 else '#0D47A1'
        ax.text(j + 0.5, i + 0.5,
                f'{val}\n({cm_pct[i,j]:.1f}%)',
                ha='center', va='center',
                fontsize=14, fontweight='bold', color=color)
ax.set_xticklabels(['Regular', 'Irregular'], fontsize=12, fontweight='bold')
ax.set_yticklabels(['Regular', 'Irregular'], fontsize=12, fontweight='bold')
ax.set_xlabel('Predicted', fontsize=12, fontweight='bold')
ax.set_ylabel('Actual',    fontsize=12, fontweight='bold')
ax.set_title(f'Confusion Matrix — Binary LSTM\nAccuracy: {overall_acc:.2f}%',
             fontweight='bold', color='#1F4E79', fontsize=13)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph5_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph5_confusion_matrix.png")

# ============================================================
# GRAPH 6 — ROC Curve
# ============================================================
print("  Saving Graph 6 — ROC Curve ...")
fpr, tpr, _ = roc_curve(y_te, y_prob)
roc_auc     = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('#F8FBFF')
ax.plot(fpr, tpr, color='#1F4E79', lw=2.5, label=f'ROC (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random (AUC = 0.50)')
ax.fill_between(fpr, tpr, alpha=0.12, color='#1F4E79')
ax.annotate(f'AUC = {roc_auc:.4f}',
            xy=(0.6, 0.3), fontsize=14, fontweight='bold', color='#1F4E79',
            bbox=dict(boxstyle='round,pad=0.4', fc='#EBF3FB', ec='#1F4E79'))
ax.set_title('ROC Curve — Binary LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('False Positive Rate', fontsize=11)
ax.set_ylabel('True Positive Rate',  fontsize=11)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.set_facecolor('#F0F6FF')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph6_roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph6_roc_curve.png")

# ============================================================
# GRAPH 7 — Summary Stats Table
# ============================================================
print("  Saving Graph 7 — Summary Stats ...")

# Metrics calculate
tn, fp, fn, tp = cm.ravel()
precision_val  = tp / (tp + fp + 1e-8) * 100
recall_val     = tp / (tp + fn + 1e-8) * 100
f1_val         = 2 * precision_val * recall_val / (precision_val + recall_val + 1e-8)

fig, ax = plt.subplots(figsize=(8, 7))
fig.patch.set_facecolor('#F8FBFF')
ax.axis('off')

summary = [
    ['Metric',            'Value'],
    ['Test Accuracy',     f'{overall_acc:.2f}%'],
    ['Best Val AUC',      f'{log["val_auc"].max():.4f}'],
    ['ROC AUC',           f'{roc_auc:.4f}'],
    ['Best Val Accuracy', f'{log["val_accuracy"].max()*100:.2f}%'],
    ['Precision',         f'{precision_val:.2f}%'],
    ['Recall',            f'{recall_val:.2f}%'],
    ['F1 Score',          f'{f1_val:.2f}%'],
    ['Best Epoch',        f'{int(best_ep)}'],
    ['Epochs Trained',    f'{len(log)}'],
    ['True Positives',    f'{tp:,}'],
    ['True Negatives',    f'{tn:,}'],
    ['False Positives',   f'{fp:,}'],
    ['False Negatives',   f'{fn:,}'],
]

table = ax.table(cellText=summary[1:], colLabels=summary[0],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.5, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1F4E79')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#EBF3FB')
    else:
        cell.set_facecolor('#FFFFFF')
    cell.set_edgecolor('#CCCCCC')

ax.set_title('Summary Statistics — Binary LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13, pad=20)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph7_summary.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph7_summary.png")

# ============================================================
# DONE
# ============================================================
print()
print("=" * 60)
print("  ALL BINARY GRAPHS COMPLETE ✅")
print("=" * 60)
print(f"  Test Accuracy : {overall_acc:.2f}%")
print(f"  ROC AUC       : {roc_auc:.4f}")
print(f"  Best Val AUC  : {log['val_auc'].max():.4f}")
print(f"  Precision     : {precision_val:.2f}%")
print(f"  Recall        : {recall_val:.2f}%")
print(f"  F1 Score      : {f1_val:.2f}%")
print()
print(f"  Saved in: {OUT_DIR}/")
print(f"    Graph1_accuracy.png")
print(f"    Graph2_loss.png")
print(f"    Graph3_auc.png")
print(f"    Graph4_precision_recall.png")
print(f"    Graph5_confusion_matrix.png")
print(f"    Graph6_roc_curve.png")
print(f"    Graph7_summary.png")
print("=" * 60)
