"""
============================================================
 20-Class LSTM Graphs — DB-HLSTM Framework
 Real Electrophysiology Data — 200 Cells

 Input  : /kaggle/working/multiclass_real_log.csv
          /kaggle/working/lstm_real_multiclass.keras
          /kaggle/input/datasets/sajidkhan1214/boldrealdata/Neural_BOLD_Data.csv

 Output : /kaggle/working/graphs/
            Graph1_multiclass_accuracy.png
            Graph2_multiclass_loss.png
            Graph3_perclass_accuracy.png
            Graph4_confusion_matrix.png
            Graph5_class_distribution.png
            Graph6_top3_accuracy.png
            Graph7_summary_stats.png
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
import tensorflow as tf
import seaborn as sns
import gc
import os
from tqdm.notebook import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
LOG_CSV    = '/kaggle/working/multiclass_real_log.csv'
MODEL_PATH = '/kaggle/working/lstm_real_multiclass.keras'
CSV_PATH   = '/kaggle/input/datasets/sajidkhan1214/boldrealdata/Neural_BOLD_Data.csv'
OUT_DIR    = '/kaggle/working/multiclass_graphs'

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
print(f"  Best Val Accuracy: {log['val_accuracy'].max()*100:.2f}%")
print(f"  Best Epoch       : {log['val_accuracy'].idxmax()}")

# ============================================================
# STEP 2 — REBUILD TEST DATA
# ============================================================
print("\n  Rebuilding test data ...")
df = pd.read_csv(CSV_PATH)

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

df_bal = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
del df; gc.collect()

bold_cols   = [f'bold_{i}'   for i in range(SEQ_LEN)]
hrf_c_cols  = [f'hrf_c_{i}'  for i in range(SEQ_LEN)]
hrf_td_cols = [f'hrf_td_{i}' for i in range(SEQ_LEN)]
hrf_dd_cols = [f'hrf_dd_{i}' for i in range(SEQ_LEN)]
v_cols      = [f'v_{i}'      for i in range(SEQ_LEN)]

n_windows = (SEQ_LEN - WINDOW) // STRIDE + 1
n_samples = len(df_bal) * n_windows

X_ts = np.zeros((n_samples, WINDOW, 5), dtype=np.float32)
X_sc = np.zeros((n_samples, 8),         dtype=np.float32)
y_20 = np.zeros(n_samples,              dtype=np.int32)

idx = 0
for _, row in tqdm(df_bal.iterrows(), total=len(df_bal), desc="Windowing"):
    bold   = row[bold_cols].values.astype(np.float32)
    hrf_c  = row[hrf_c_cols].values.astype(np.float32)
    hrf_td = row[hrf_td_cols].values.astype(np.float32)
    hrf_dd = row[hrf_dd_cols].values.astype(np.float32)
    volt   = row[v_cols].values.astype(np.float32)
    l20    = int(row['Class20'])

    for start in range(0, SEQ_LEN - WINDOW + 1, STRIDE):
        end  = start + WINDOW
        b    = bold[start:end];   b   = (b   - b.mean())   / (b.std()   + 1e-8)
        c    = hrf_c[start:end];  c   = (c   - c.mean())   / (c.std()   + 1e-8)
        t    = hrf_td[start:end]; t   = (t   - t.mean())   / (t.std()   + 1e-8)
        d    = hrf_dd[start:end]; d   = (d   - d.mean())   / (d.std()   + 1e-8)
        v    = volt[start:end];   v_n = (v   - v.mean())   / (v.std()   + 1e-8)

        X_ts[idx, :, 0] = b
        X_ts[idx, :, 1] = c
        X_ts[idx, :, 2] = t
        X_ts[idx, :, 3] = d
        X_ts[idx, :, 4] = v_n

        X_sc[idx] = [
            volt[start:end].mean(),
            volt[start:end].std(),
            volt[start:end].min(),
            volt[start:end].max(),
            (volt[start:end] > 0).sum(),
            volt[start:end].max() - volt[start:end].min(),
            np.abs(volt[start:end]).mean(),
            volt[start:end].std() / (volt[start:end].mean() + 1e-8),
        ]
        y_20[idx] = l20
        idx += 1

del df_bal; gc.collect()

X_sc = StandardScaler().fit_transform(X_sc).astype(np.float32)

# Test split — same as training
n      = idx
val_n  = int(n * 0.15)
test_n = int(n * 0.10)
tr_n   = n - val_n - test_n

X_te    = X_ts[tr_n + val_n:]
X_sc_te = X_sc[tr_n + val_n:]
y_te    = y_20[tr_n + val_n:]
del X_ts, X_sc, y_20; gc.collect()

print(f"  Test samples: {len(y_te):,}")

# ============================================================
# STEP 3 — LOAD MODEL + PREDICT
# ============================================================
print("\n  Loading model and predicting ...")
model = tf.keras.models.load_model(MODEL_PATH)

AUTOTUNE = tf.data.AUTOTUNE
te_ds = (tf.data.Dataset
         .from_tensor_slices({'ts_input': X_te, 'scalar_input': X_sc_te})
         .batch(BATCH_SIZE).prefetch(AUTOTUNE))

y_pred_prob = model.predict(te_ds, verbose=1)
y_pred      = np.argmax(y_pred_prob, axis=1)
del X_te, X_sc_te; gc.collect()

overall_acc  = (y_pred == y_te).mean() * 100
best_val_acc = log['val_accuracy'].max() * 100
best_ep      = log['val_accuracy'].idxmax()
best_val_loss= log.loc[best_ep, 'val_loss']

print(f"  Test Accuracy: {overall_acc:.2f}%")

# Per class accuracy
per_class_acc = []
for cls in range(N_CLASSES):
    mask = y_te == cls
    acc  = (y_pred[mask] == cls).mean() * 100 if mask.sum() > 0 else 0.0
    per_class_acc.append(acc)

# Top-3 accuracy
top3_acc = []
for cls in range(N_CLASSES):
    mask = y_te == cls
    if mask.sum() > 0:
        top3 = np.argsort(y_pred_prob[mask], axis=1)[:, -3:]
        acc  = np.mean([cls in row for row in top3]) * 100
    else:
        acc = 0.0
    top3_acc.append(acc)

# ============================================================
# GRAPH 1 — Accuracy per Epoch
# ============================================================
print("\n  Saving Graph 1 — Accuracy per Epoch ...")
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.plot(log['epoch'], log['accuracy'],     label='Train', color='#1F4E79', linewidth=2)
ax.plot(log['epoch'], log['val_accuracy'], label='Val',   color='#FF8C00', linewidth=2, linestyle='--')
ax.axvline(x=best_ep, color='green', linestyle=':', linewidth=1.5, label=f'Best Epoch={best_ep}')
ax.annotate(f"Best: {best_val_acc:.2f}%",
            xy=(best_ep, log['val_accuracy'].max()),
            xytext=(best_ep + 2, log['val_accuracy'].max() - 0.05),
            fontsize=10, color='green',
            bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='green'))
ax.set_title('Accuracy per Epoch — 20-Class LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('Accuracy', fontsize=11)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
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
ax.set_title('Loss per Epoch — 20-Class LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Epoch', fontsize=11); ax.set_ylabel('Loss', fontsize=11)
ax.legend(fontsize=10); ax.grid(alpha=0.3)
ax.set_facecolor('#F0F6FF')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph2_loss.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph2_loss.png")

# ============================================================
# GRAPH 3 — Per-Class Accuracy
# ============================================================
print("  Saving Graph 3 — Per-Class Accuracy ...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#F8FBFF')
colors_bar = ['#C00000' if a < 50 else '#2E75B6' if a < 70 else '#375623'
              for a in per_class_acc]
bars = ax.bar(range(N_CLASSES), per_class_acc, color=colors_bar,
              edgecolor='white', linewidth=0.5, width=0.7)
ax.axhline(y=np.mean(per_class_acc), color='#FF8C00', linestyle='--',
           linewidth=2, label=f'Mean = {np.mean(per_class_acc):.1f}%')
for bar, acc in zip(bars, per_class_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('Per-Class Accuracy — 20-Class LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Class', fontsize=11); ax.set_ylabel('Accuracy (%)', fontsize=11)
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels([f'C{i}' for i in range(N_CLASSES)], fontsize=9)
ax.set_ylim(0, 115)
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
ax.set_facecolor('#F0F6FF')
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph3_perclass_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph3_perclass_accuracy.png")

# ============================================================
# GRAPH 4 — Confusion Matrix
# ============================================================
print("  Saving Graph 4 — Confusion Matrix ...")
cm  = confusion_matrix(y_te, y_pred)
fig, ax = plt.subplots(figsize=(14, 12))
fig.patch.set_facecolor('#F8FBFF')
sns.heatmap(cm, annot=True, fmt='d', ax=ax,
            xticklabels=[f'C{i}' for i in range(N_CLASSES)],
            yticklabels=[f'C{i}' for i in range(N_CLASSES)],
            cmap='Blues', annot_kws={'size': 8})
ax.set_title(f'Confusion Matrix — 20-Class LSTM\nAccuracy: {overall_acc:.2f}%',
             fontweight='bold', color='#1F4E79', fontsize=13, pad=15)
ax.set_xlabel('Predicted Class', fontsize=11)
ax.set_ylabel('Actual Class',    fontsize=11)
ax.tick_params(labelsize=9)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph4_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph4_confusion_matrix.png")

# ============================================================
# GRAPH 5 — Class Distribution
# ============================================================
print("  Saving Graph 5 — Class Distribution ...")
unique, counts = np.unique(y_te, return_counts=True)
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor('#F8FBFF')
bars = ax.bar(unique, counts, color='#2E75B6', edgecolor='white', width=0.7)
for bar, cnt in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            str(cnt), ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Test Set Class Distribution — 20-Class LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Class', fontsize=11); ax.set_ylabel('Samples', fontsize=11)
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels([f'C{i}' for i in range(N_CLASSES)], fontsize=9)
ax.set_facecolor('#F0F6FF')
ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph5_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph5_class_distribution.png")

# ============================================================
# GRAPH 6 — Top-3 Accuracy
# ============================================================
print("  Saving Graph 6 — Top-3 Accuracy ...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#F8FBFF')
bars = ax.bar(range(N_CLASSES), top3_acc, color='#375623',
              edgecolor='white', linewidth=0.5, width=0.7)
ax.axhline(y=np.mean(top3_acc), color='#FF8C00', linestyle='--',
           linewidth=2, label=f'Mean = {np.mean(top3_acc):.1f}%')
for bar, acc in zip(bars, top3_acc):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{acc:.0f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
ax.set_title('Top-3 Accuracy per Class — 20-Class LSTM\nDB-HLSTM Framework',
             fontweight='bold', color='#1F4E79', fontsize=13)
ax.set_xlabel('Class', fontsize=11); ax.set_ylabel('Top-3 Accuracy (%)', fontsize=11)
ax.set_xticks(range(N_CLASSES))
ax.set_xticklabels([f'C{i}' for i in range(N_CLASSES)], fontsize=9)
ax.set_ylim(0, 115)
ax.set_facecolor('#F0F6FF')
ax.legend(fontsize=10); ax.grid(axis='y', alpha=0.3)
fig.tight_layout()
fig.savefig(f'{OUT_DIR}/Graph6_top3_accuracy.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ Graph6_top3_accuracy.png")

# ============================================================
# GRAPH 7 — Summary Stats Table
# ============================================================
print("  Saving Graph 7 — Summary Stats ...")
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('#F8FBFF')
ax.axis('off')

summary = [
    ['Metric',            'Value'],
    ['Test Accuracy',     f'{overall_acc:.2f}%'],
    ['Best Val Accuracy', f'{best_val_acc:.2f}%'],
    ['Best Epoch',        f'{int(best_ep)}'],
    ['Best Val Loss',     f'{best_val_loss:.4f}'],
    ['Mean Class Acc',    f'{np.mean(per_class_acc):.2f}%'],
    ['Mean Top-3 Acc',    f'{np.mean(top3_acc):.2f}%'],
    ['Total Classes',     '20'],
    ['Random Chance',     '5.00%'],
    ['Improvement',       f'{overall_acc/5:.1f}x over random'],
    ['Epochs Trained',    f'{len(log)}'],
]

table = ax.table(cellText=summary[1:], colLabels=summary[0],
                 loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.5, 2.0)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_facecolor('#1F4E79')
        cell.set_text_props(color='white', fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor('#EBF3FB')
    else:
        cell.set_facecolor('#FFFFFF')
    cell.set_edgecolor('#CCCCCC')

ax.set_title('Summary Statistics — DB-HLSTM\n20-Class Real Electrophysiology Data',
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
print("  ALL GRAPHS COMPLETE ✅")
print("=" * 60)
print(f"  Test Accuracy    : {overall_acc:.2f}%")
print(f"  Best Val Accuracy: {best_val_acc:.2f}%")
print(f"  Mean Class Acc   : {np.mean(per_class_acc):.2f}%")
print(f"  Mean Top-3 Acc   : {np.mean(top3_acc):.2f}%")
print(f"  Random Chance    : 5.00%")
print(f"  Improvement      : {overall_acc/5:.1f}x over random")
print()
print(f"  Saved in: {OUT_DIR}/")
print(f"    Graph1_accuracy.png")
print(f"    Graph2_loss.png")
print(f"    Graph3_perclass_accuracy.png")
print(f"    Graph4_confusion_matrix.png")
print(f"    Graph5_class_distribution.png")
print(f"    Graph6_top3_accuracy.png")
print(f"    Graph7_summary.png")
print("=" * 60)
