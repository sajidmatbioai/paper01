"""
============================================================
 LSTM Graphs — Binary + Multi-Class  [KAGGLE VERSION]
 DB-HLSTM PARAMETERS — Step 0 Fixed (I,L,O,P)

 Generates 10 graphs:
   Graph 1  — Binary training history
   Graph 2  — Binary ROC curve
   Graph 3  — Binary confusion matrix
   Graph 4  — Multi-class training history
   Graph 5  — Multi-class confusion matrix (20×20)
   Graph 6  — Per-label accuracy bar chart
   Graph 7  — Binary precision-recall curve
   Graph 8  — Per-class F1 score bar chart (Multi-Class)
   Graph 9  — Class distribution (Binary + Multi-Class)
   Graph 10 — Class weight visualization (Binary + Multi-Class)

 Run AFTER lstm_binary AND lstm_multiclass complete
 Skips graphs that already exist (resume support)

 FIXED PARAMETERS (Step 0):
   I: a=0.04,  b=0.18,  c=-63, d=7,  xi=12.0
   L: a=0.025, b=-0.15, c=-58, d=5,  xi=38.0
   O: a=0.04,  b=0.30,  c=-58, d=2,  xi=7.0
   P: a=0.12,  b=0.28,  c=-58, d=1,  xi=0.5
============================================================
"""

import os, gc
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (confusion_matrix, roc_curve,
                             auc, precision_recall_curve,
                             classification_report)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     BatchNormalization, Concatenate)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# ── PATHS ───────────────────────────────────────────────────
CSV_PATH    = "/kaggle/working/Neural_BOLD_Data.csv"
WORK_DIR    = "/kaggle/working"
MODEL_BIN   = os.path.join(WORK_DIR, "lstm_binary_hybrid.keras")
MODEL_MULTI = os.path.join(WORK_DIR, "lstm_hybrid_v3.keras")
HIST_BIN    = os.path.join(WORK_DIR, "binary_history.npy")
HIST_MULTI  = os.path.join(WORK_DIR, "hybrid_v3_history.npy")       # actual file name from training code

# ── ALSO CHECK CSV LOG (alternate history source) ────────────
HIST_MULTI_CSV = os.path.join(WORK_DIR, "hybrid_v3_epoch_log.csv")  # actual CSV name from training code

OUT_DIR     = os.path.join(WORK_DIR, "LSTM_Graphs_Fixed")

# ── DELETE all old graphs before saving new ones ─────────────
import shutil
if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
    print(f"  Deleted old graphs folder: {OUT_DIR}")
os.makedirs(OUT_DIR, exist_ok=True)
print(f"  Fresh output folder created: {OUT_DIR}")

SEQ_LEN = 600
LABELS  = list("ABCDEFGHIJKLMNOPQRST")
FIXED   = ['I','L','O','P']

GRAPH_FILES = {
    1:  "Graph1_binary_history.png",
    2:  "Graph2_binary_ROC.png",
    3:  "Graph3_binary_confusion.png",
    4:  "Graph4_multiclass_history.png",
    5:  "Graph5_multiclass_confusion.png",
    6:  "Graph6_per_label_accuracy.png",
    7:  "Graph7_binary_precision_recall.png",
    8:  "Graph8_multiclass_f1_score.png",
    9:  "Graph9_class_distribution.png",
    10: "Graph10_class_weight_visualization.png",
}

# ── MODEL BUILDERS — must match training code exactly ────────
def build_binary():
    REG = l2(1e-4)
    ts_in = Input(shape=(SEQ_LEN,5), name='ts_input')
    x = LSTM(128,return_sequences=True,dropout=0.2,recurrent_dropout=0.0,kernel_regularizer=REG)(ts_in)
    x = BatchNormalization()(x)
    x = LSTM(64,return_sequences=False,dropout=0.2,recurrent_dropout=0.0,kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dense(64,activation='relu',kernel_regularizer=REG)(x)
    x = Dropout(0.2)(x)
    sc_in = Input(shape=(8,),name='scalar_input')
    y_ = Dense(32,activation='relu',kernel_regularizer=REG)(sc_in)
    y_ = BatchNormalization()(y_)
    y_ = Dense(16,activation='relu',kernel_regularizer=REG)(y_)
    merged = Concatenate()([x,y_])
    z = Dense(32,activation='relu',kernel_regularizer=REG)(merged)
    z = Dropout(0.2)(z)
    out = Dense(1,activation='sigmoid',dtype='float32')(z)
    m = Model(inputs=[ts_in,sc_in],outputs=out)
    m.compile(optimizer=Adam(0.001),loss='binary_crossentropy',
              metrics=['accuracy',tf.keras.metrics.AUC(name='auc')])
    return m

def build_multi():
    REG = l2(1e-4)
    ts_in = Input(shape=(SEQ_LEN,5),name='ts_input')
    x = LSTM(128,return_sequences=True,dropout=0.2,recurrent_dropout=0.0,kernel_regularizer=REG)(ts_in)
    x = BatchNormalization()(x)
    x = LSTM(64,return_sequences=False,dropout=0.2,recurrent_dropout=0.0,kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dense(64,activation='relu',kernel_regularizer=REG)(x)
    x = Dropout(0.2)(x)
    sc_in = Input(shape=(8,),name='scalar_input')
    y_ = Dense(32,activation='relu',kernel_regularizer=REG)(sc_in)
    y_ = BatchNormalization()(y_)
    y_ = Dense(16,activation='relu',kernel_regularizer=REG)(y_)
    merged = Concatenate()([x,y_])
    z = Dense(64,activation='relu',kernel_regularizer=REG)(merged)
    z = Dropout(0.2)(z)
    out = Dense(20,activation='softmax',dtype='float32',name='output')(z)
    m = Model(inputs=[ts_in,sc_in],outputs=out)
    m.compile(optimizer=Adam(0.001),loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

# ── LOAD HISTORY — with CSV fallback ─────────────────────────
def load_history_multi():
    """
    Priority:
    1. hybrid_v3_history.npy   (saved by kaggle_lstm_multiclass.py)
    2. hybrid_v3_epoch_log.csv (CSVLogger from training callbacks)
    Returns dict with keys: accuracy, val_accuracy, loss, val_loss
    or empty dict if nothing found.
    """
    # Option 1: .npy  — training code saves this automatically after model.fit()
    if os.path.exists(HIST_MULTI):
        h = np.load(HIST_MULTI, allow_pickle=True).item()
        if h and any(len(v) > 0 for v in h.values()):
            print("  ✅  Loaded history from hybrid_v3_history.npy")
            return h
        else:
            print("  ⚠️  hybrid_v3_history.npy exists but is EMPTY")

    # Option 2: CSV log — written by CSVLogger callback during training
    if os.path.exists(HIST_MULTI_CSV):
        try:
            df_log = pd.read_csv(HIST_MULTI_CSV)
            print(f"  ✅  Loaded history from hybrid_v3_epoch_log.csv  "
                  f"({len(df_log)} epochs)")
            print(f"      Columns: {df_log.columns.tolist()}")
            h = {}
            col_map = {
                'accuracy':     ['accuracy', 'acc', 'train_accuracy', 'train_acc'],
                'val_accuracy': ['val_accuracy', 'val_acc'],
                'loss':         ['loss', 'train_loss'],
                'val_loss':     ['val_loss'],
            }
            for key, candidates in col_map.items():
                for c in candidates:
                    if c in df_log.columns:
                        h[key] = df_log[c].tolist()
                        break
            if h:
                return h
            else:
                print("  ⚠️  CSV found but no recognised columns")
        except Exception as e:
            print(f"  ⚠️  Could not read CSV: {e}")

    print("  ❌  No history found for multi-class model.")
    print("      Expected files in /kaggle/working/:")
    print("        hybrid_v3_history.npy   (auto-saved by training code)")
    print("        hybrid_v3_epoch_log.csv (auto-saved by CSVLogger callback)")
    print("      Run kaggle_lstm_multiclass.py first, then run this file.")
    return {}

# ── STATUS CHECK ─────────────────────────────────────────────
print("="*60)
print("  LSTM Graphs  [KAGGLE — DB-HLSTM]")
print("="*60)
status = {}
for n, fname in GRAPH_FILES.items():
    path = os.path.join(OUT_DIR, fname)
    if os.path.exists(path):
        status[n] = "done"; print(f"  Graph {n}  — exists ✅ skip")
    elif n in [2,3,7] and not os.path.exists(MODEL_BIN):
        status[n] = "no_model"; print(f"  Graph {n}  — waiting: run lstm_binary first ⏳")
    elif n in [5,6,8] and not os.path.exists(MODEL_MULTI):
        status[n] = "no_model"; print(f"  Graph {n}  — waiting: run lstm_multiclass first ⏳")
    else:
        status[n] = "todo"; print(f"  Graph {n}  — will generate ▶")

to_do = [n for n, s in status.items() if s == "todo"]
if not to_do:
    print("\n  ⚠️  All graphs already exist — deleting and regenerating all...")
    for fname in GRAPH_FILES.values():
        fpath = os.path.join(OUT_DIR, fname)
        if os.path.exists(fpath):
            os.remove(fpath)
    to_do = list(GRAPH_FILES.keys())
    print(f"  ✅ Cleared. Will regenerate: {to_do}\n")

# ── LOAD DATA ────────────────────────────────────────────────
print(f"\n  Loading CSV ...")
needed = (['Label','Mode','Class','s_t_count'] +
          [f"v_{i}"      for i in range(SEQ_LEN)] +
          [f"s_{i}"      for i in range(SEQ_LEN)] +
          [f"hrf_c_{i}"  for i in range(SEQ_LEN)] +
          [f"hrf_td_{i}" for i in range(SEQ_LEN)] +
          [f"hrf_dd_{i}" for i in range(SEQ_LEN)] +
          [f"bold_{i}"   for i in range(SEQ_LEN)])
header = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
df     = pd.read_csv(CSV_PATH, usecols=[c for c in needed if c in header])
print(f"  Rows: {len(df):,}")

# Build arrays
hrf_c  = df[[f"hrf_c_{i}"  for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_td = df[[f"hrf_td_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_dd = df[[f"hrf_dd_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
bold   = df[[f"bold_{i}"   for i in range(SEQ_LEN)]].values.astype(np.float32)
v_ts   = df[[f"v_{i}"      for i in range(SEQ_LEN)]].values.astype(np.float32) \
         if "v_0" in df.columns else np.zeros_like(bold)
X      = np.stack([hrf_c, hrf_td, hrf_dd, bold, v_ts], axis=2)
X_norm = np.zeros_like(X, dtype=np.float32)
for fi in range(5):
    feat = X[:,:,fi]
    X_norm[:,:,fi] = (feat - feat.mean(axis=1, keepdims=True)) / (feat.std(axis=1, keepdims=True) + 1e-8)
del X, hrf_c, hrf_td, hrf_dd, bold, v_ts; gc.collect()

# Scalars
s_cols = [f"s_{i}" for i in range(SEQ_LEN)]
v_cols = [f"v_{i}" for i in range(SEQ_LEN)]
scalars = []
for idx in range(len(df)):
    row = df.iloc[idx]; sc = float(row.get('s_t_count', 0))
    if "v_0" in df.columns:
        v = row[v_cols].values.astype(float); vm, vs, vn, vx = v.mean(), v.std(), v.min(), v.max()
    else:
        vm = vs = vn = vx = 0.0
    if "s_0" in df.columns:
        s = row[s_cols].values.astype(float); spk = np.where(s == 1)[0]; sr = len(spk) / SEQ_LEN
        isi = np.diff(spk); im, ist = (isi.mean(), isi.std()) if len(spk) > 1 else (0.0, 0.0)
    else:
        sr = sc / SEQ_LEN; im = ist = 0.0
    scalars.append([sc, vm, vs, vn, vx, sr, im, ist])
X_sc   = np.array(scalars, dtype=np.float32)
sc     = StandardScaler()
X_sc_n = sc.fit_transform(X_sc).astype(np.float32)

y_bin = df["Class"].values
le    = LabelEncoder(); y_int = le.fit_transform(df["Label"].values)
label_mode = df.groupby('Label')['Mode'].first().to_dict()

# ── Save full-dataset arrays BEFORE del df — needed for Graph 9 & 10 ──
y_bin_all = y_bin.copy()
y_int_all = y_int.copy()

del df, scalars, X_sc; gc.collect()

_, X_ts_te, _, X_sc_te, _, y_te_b  = train_test_split(X_norm, X_sc_n, y_bin, test_size=0.20, random_state=42, stratify=y_bin)
_, X_ts_m,  _, X_sc_m,  _, y_te_mi = train_test_split(X_norm, X_sc_n, y_int, test_size=0.20, random_state=42, stratify=y_int)

hist_b = np.load(HIST_BIN,   allow_pickle=True).item() if os.path.exists(HIST_BIN) else {}
hist_m = load_history_multi()   # ← uses smart loader with CSV fallback

def predict_binary(Xts, Xsc):
    m = build_binary()
    m([np.zeros((1, SEQ_LEN, 5), dtype=np.float32), np.zeros((1, 8), dtype=np.float32)], training=False)
    m.load_weights(MODEL_BIN)
    return m.predict({'ts_input': Xts, 'scalar_input': Xsc}, verbose=0).flatten()

def predict_multi(Xts, Xsc):
    m = build_multi()
    m([np.zeros((1, SEQ_LEN, 5), dtype=np.float32), np.zeros((1, 8), dtype=np.float32)], training=False)
    m.load_weights(MODEL_MULTI)
    return np.argmax(m.predict({'ts_input': Xts, 'scalar_input': Xsc}, verbose=0), axis=1)

# ── GRAPH 1 — Binary History ─────────────────────────────────
if 1 in to_do:
    print("\n  Graph 1: Binary History ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Binary — Training History  [DB-HLSTM]\nRegular vs Irregular",
                 fontsize=14, fontweight='bold', color='#1F4E79')
    for ax, (key, title) in zip(axes, [('accuracy','Accuracy'), ('loss','Loss')]):
        train_vals = hist_b.get(key, [])
        val_vals   = hist_b.get(f'val_{key}', [])
        if not train_vals:
            ax.text(0.5, 0.5, f'No binary history data\nRun lstm_binary first',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=12, color='gray',
                    bbox=dict(boxstyle='round', fc='#FFF9C4', ec='orange'))
        else:
            ax.plot(train_vals, color='#2E75B6', lw=2, label='Train')
            ax.plot(val_vals,   color='#C00000', lw=2, linestyle='--', label='Val')
            if key == 'accuracy':
                ax.set_ylim(0, 1.05)
                if val_vals:
                    best = max(val_vals)
                    ax.axhline(best, color='green', lw=1, linestyle=':')
                    ax.annotate(f"Best:{best:.3f}", xy=(0.65, best),
                                xycoords=('axes fraction','data'), fontsize=10, color='green',
                                bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='green'))
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.legend()
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[1]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[1]}")

# ── GRAPH 2 — Binary ROC ─────────────────────────────────────
if 2 in to_do:
    print("  Graph 2: Binary ROC ...")
    y_prob = predict_binary(X_ts_te, X_sc_te)
    fpr, tpr, _ = roc_curve(y_te_b, y_prob); roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#C00000', lw=2.5, label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0,1], [0,1], color='gray', lw=1.5, linestyle='--', label='Random')
    ax.fill_between(fpr, tpr, alpha=0.15, color='#C00000')
    ax.set_title("ROC Curve — Binary LSTM  [DB-HLSTM]", fontweight='bold', fontsize=13, color='#1F4E79')
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.annotate(f"AUC = {roc_auc:.3f}", xy=(0.6, 0.3), fontsize=14, fontweight='bold', color='#C00000',
                bbox=dict(boxstyle='round,pad=0.4', fc='#FFEBEE', ec='#C00000'))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[2]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[2]}")

# ── GRAPH 3 — Binary Confusion (manual imshow — no white line bug) ──
if 3 in to_do:
    print("  Graph 3: Binary Confusion ...")
    if 'y_prob' not in dir():
        y_prob = predict_binary(X_ts_te, X_sc_te)
    y_pred_b = (y_prob >= 0.5).astype(int)
    acc      = (y_pred_b == y_te_b).mean() * 100
    cm_b     = confusion_matrix(y_te_b, y_pred_b)
    cmap_b   = LinearSegmentedColormap.from_list(
                   'b2', ['#FFFFFF','#C9D9F0','#0D47A1'], N=256)
    cm_b_pct = cm_b.astype(float) / (cm_b.sum(axis=1, keepdims=True) + 1e-8) * 100
    fig, ax  = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_b_pct, cmap=cmap_b, vmin=0, vmax=100, aspect='equal')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).ax.tick_params(labelsize=9)
    for i in range(2):
        for j in range(2):
            val   = int(cm_b[i, j])
            color = 'white' if cm_b_pct[i, j] > 50 else '#0D47A1'
            ax.text(j, i, f"{val}\n({cm_b_pct[i,j]:.1f}%)",
                    ha='center', va='center', fontsize=13,
                    fontweight='bold', color=color)
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Regular','Irregular'], fontsize=11, fontweight='bold')
    ax.set_yticklabels(['Regular','Irregular'], fontsize=11, fontweight='bold')
    ax.tick_params(length=0)
    ax.set_xlabel('Predicted', fontsize=11, fontweight='bold', labelpad=8)
    ax.set_ylabel('True',      fontsize=11, fontweight='bold', labelpad=8)
    ax.set_title(f"Confusion Matrix — Binary  [DB-HLSTM]\nAccuracy: {acc:.1f}%",
                 fontweight='bold', fontsize=12, color='#1F4E79')
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[3]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[3]}")

# ── GRAPH 4 — Multi History ───────────────────────────────────
if 4 in to_do:
    print("\n  Graph 4: Multi-class History ...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("LSTM Multi-Class — Training History  [DB-HLSTM]\n20 Labels (A to T)",
                 fontsize=14, fontweight='bold', color='#1F4E79')

    history_found = bool(hist_m) and any(len(v) > 0 for v in hist_m.values())

    for ax, (key, title) in zip(axes, [('accuracy','Accuracy'), ('loss','Loss')]):
        train_vals = hist_m.get(key, [])
        val_vals   = hist_m.get(f'val_{key}', [])

        # ── FIX: show message if no data instead of blank graph ──
        if not train_vals:
            ax.text(0.5, 0.5,
                    'No multi-class history data found.\n\n'
                    'Fix options:\n'
                    '1. Run lstm_multiclass training first\n'
                    '2. It auto-saves as hybrid_v3_history.npy\n'
                    '3. Or CSV: hybrid_v3_epoch_log.csv\n\n'
                    f'Expected path:\n{WORK_DIR}',
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='#333333',
                    bbox=dict(boxstyle='round', fc='#FFF9C4', ec='orange', alpha=0.9))
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            ax.axis('off')
        else:
            epochs = list(range(1, len(train_vals) + 1))
            ax.plot(epochs, train_vals, color='#2E75B6', lw=2, label='Train')
            if val_vals:
                ax.plot(epochs, val_vals, color='#C00000', lw=2, linestyle='--', label='Val')
            if key == 'accuracy':
                ax.set_ylim(0, 1.05)
                if val_vals:
                    best = max(val_vals)
                    ax.axhline(best, color='green', lw=1, linestyle=':')
                    ax.annotate(f"Best:{best:.3f}", xy=(0.65, best),
                                xycoords=('axes fraction','data'), fontsize=10, color='green',
                                bbox=dict(boxstyle='round,pad=0.3', fc='#E8F5E9', ec='green'))
            ax.set_xlabel("Epoch")
            ax.legend()
            ax.grid(True, alpha=0.3)

        ax.set_title(title, fontweight='bold')

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[4]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[4]}")
    if not history_found:
        print("  ⚠️  Graph 4 saved with placeholder — no actual history data found.")
        print(f"      Save your Keras history as: {HIST_MULTI}")
        print("      Training code saves it automatically — just run kaggle_lstm_multiclass.py")

# ── GRAPH 5 — Multi Confusion (manual imshow — no white line bug) ──
if 5 in to_do:
    print("  Graph 5: Multi Confusion ...")
    y_pred_m  = predict_multi(X_ts_m, X_sc_m)
    acc_m     = (y_pred_m == y_te_mi).mean() * 100
    cm_m      = confusion_matrix(y_te_mi, y_pred_m)
    n         = len(LABELS)
    cmap_m    = LinearSegmentedColormap.from_list(
                    'lb', ['#FFFFFF','#C9D9F0','#4A7FC1','#0D47A1'], N=256)
    cm_m_pct  = cm_m.astype(float) / (cm_m.sum(axis=1, keepdims=True) + 1e-8) * 100
    fig, ax   = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_m_pct, cmap=cmap_m, vmin=0, vmax=100, aspect='equal')
    fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02).ax.tick_params(labelsize=9)
    for i in range(n):
        for j in range(n):
            val   = int(cm_m[i, j])
            color = 'white' if cm_m_pct[i, j] > 50 else '#0D47A1'
            ax.text(j, i, str(val), ha='center', va='center',
                    fontsize=9, fontweight='bold', color=color)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(LABELS, fontsize=10, fontweight='bold')
    ax.set_yticklabels(LABELS, fontsize=10, fontweight='bold')
    ax.tick_params(length=0)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label',      fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(f"Confusion Matrix — Multi-Class  [DB-HLSTM]\nAccuracy: {acc_m:.1f}%",
                 fontweight='bold', fontsize=13, color='#1F4E79')
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[5]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[5]}")

# ── GRAPH 6 — Per-Label Bar ───────────────────────────────────
if 6 in to_do:
    print("  Graph 6: Per-label Accuracy ...")
    if 'y_pred_m' not in dir():
        y_pred_m = predict_multi(X_ts_m, X_sc_m)
    per_acc = [(y_pred_m[y_te_mi==i]==i).mean()*100 if (y_te_mi==i).sum()>0 else 0 for i in range(20)]
    colors  = ['#C00000' if label_mode.get(l,'')=='Irregular' else '#2E75B6' for l in LABELS]
    fig, ax = plt.subplots(figsize=(16, 6))
    bars = ax.bar(LABELS, per_acc, color=colors, width=0.65, edgecolor='white')
    for bar, a in zip(bars, per_acc):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{a:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(np.mean(per_acc), color='green', lw=2, linestyle='--')
    for i, lbl in enumerate(LABELS):
        if lbl in FIXED:
            ax.text(i, 3, "⚡", ha='center', fontsize=12, color='orange')
    ax.set_title("Per-Label Accuracy — Multi-Class  [DB-HLSTM]\nBlue=Regular  Red=Irregular  ⚡=Fixed labels",
                 fontweight='bold', fontsize=13, color='#1F4E79')
    ax.set_xlabel("Label (A–T)"); ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 118)
    ax.legend(handles=[
        Patch(color='#2E75B6', label='Regular'),
        Patch(color='#C00000', label='Irregular'),
        plt.Line2D([0],[0], color='green', lw=2, linestyle='--',
                   label=f'Mean: {np.mean(per_acc):.1f}%')
    ], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[6]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[6]}")

# ── GRAPH 7 — Binary Precision-Recall Curve ──────────────────
if 7 in to_do:
    print("  Graph 7: Binary Precision-Recall Curve ...")
    if 'y_prob' not in dir():
        y_prob = predict_binary(X_ts_te, X_sc_te)
    precision, recall, _ = precision_recall_curve(y_te_b, y_prob)
    pr_auc   = auc(recall, precision)
    no_skill = y_te_b.sum() / len(y_te_b)
    fig, ax  = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color='#2E75B6', lw=2.5,
            label=f'DB-HLSTM (PR-AUC={pr_auc:.3f})')
    ax.axhline(no_skill, color='gray', lw=1.5, linestyle='--',
               label=f'Baseline ({no_skill:.3f})')
    ax.fill_between(recall, precision, alpha=0.12, color='#2E75B6')
    ax.set_title("Precision-Recall Curve — Binary LSTM  [DB-HLSTM]",
                 fontweight='bold', fontsize=13, color='#1F4E79')
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11); ax.grid(True, alpha=0.3)
    ax.annotate(f"PR-AUC = {pr_auc:.3f}", xy=(0.35, 0.15),
                fontsize=14, fontweight='bold', color='#2E75B6',
                bbox=dict(boxstyle='round,pad=0.4', fc='#E3F2FD', ec='#2E75B6'))
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[7]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[7]}")

# ── GRAPH 8 — Multi-Class Per-Class F1 Score ─────────────────
if 8 in to_do:
    print("  Graph 8: Per-Class F1 Score ...")
    if 'y_pred_m' not in dir():
        y_pred_m = predict_multi(X_ts_m, X_sc_m)
    report    = classification_report(y_te_mi, y_pred_m,
                                      target_names=LABELS,
                                      output_dict=True,
                                      zero_division=0)
    f1_scores  = [report[lbl]['f1-score'] * 100 for lbl in LABELS]
    colors_f1  = ['#C00000' if label_mode.get(l,'')=='Irregular'
                  else '#2E75B6' for l in LABELS]
    mean_f1    = np.mean(f1_scores)
    fig, ax    = plt.subplots(figsize=(16, 6))
    bars = ax.bar(LABELS, f1_scores, color=colors_f1, width=0.65, edgecolor='white')
    for bar, f in zip(bars, f1_scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                f"{f:.0f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax.axhline(mean_f1, color='green', lw=2, linestyle='--')
    for i, lbl in enumerate(LABELS):
        if lbl in FIXED:
            ax.text(i, 3, "⚡", ha='center', fontsize=12, color='orange')
    ax.set_title("Per-Class F1 Score — Multi-Class  [DB-HLSTM]\nBlue=Regular  Red=Irregular  ⚡=Fixed labels",
                 fontweight='bold', fontsize=13, color='#1F4E79')
    ax.set_xlabel("Label (A–T)"); ax.set_ylabel("F1 Score (%)"); ax.set_ylim(0, 118)
    ax.legend(handles=[
        Patch(color='#2E75B6', label='Regular'),
        Patch(color='#C00000', label='Irregular'),
        plt.Line2D([0],[0], color='green', lw=2, linestyle='--',
                   label=f'Mean F1: {mean_f1:.1f}%')
    ], fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[8]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[8]}")

# ── GRAPH 9 — Class Distribution ─────────────────────────────
if 9 in to_do:
    print("  Graph 9: Class Distribution ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Class Distribution — DB-HLSTM Dataset\nBinary (Left)  |  Multi-Class 20 Labels (Right)",
                 fontsize=14, fontweight='bold', color='#1F4E79')

    # Left — Binary
    ax_b       = axes[0]
    bin_counts = [int((y_bin_all == 0).sum()), int((y_bin_all == 1).sum())]
    bin_labels = ['Regular', 'Irregular']
    bin_colors = ['#2E75B6', '#C00000']
    total_b    = sum(bin_counts)
    b_bars = ax_b.bar(bin_labels, bin_counts, color=bin_colors,
                      width=0.45, edgecolor='white')
    for bar, cnt in zip(b_bars, bin_counts):
        ax_b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + total_b * 0.01,
                  f"{cnt:,}\n({cnt/total_b*100:.1f}%)",
                  ha='center', va='bottom', fontsize=12, fontweight='bold')
    ax_b.set_title("Binary — Regular vs Irregular", fontweight='bold', fontsize=12)
    ax_b.set_ylabel("Sample Count", fontsize=11, fontweight='bold')
    ax_b.set_ylim(0, max(bin_counts) * 1.20)
    ax_b.grid(True, alpha=0.3, axis='y')
    ax_b.tick_params(axis='x', labelsize=12)
    for sp in ['top','right']: ax_b.spines[sp].set_visible(False)

    # Right — Multi-Class
    ax_m         = axes[1]
    multi_counts = [int((y_int_all == i).sum()) for i in range(20)]
    mc_colors    = ['#C00000' if label_mode.get(l,'')=='Irregular'
                    else '#2E75B6' for l in LABELS]
    m_bars = ax_m.bar(LABELS, multi_counts, color=mc_colors,
                      width=0.65, edgecolor='white')
    for bar, cnt in zip(m_bars, multi_counts):
        ax_m.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + max(multi_counts) * 0.01,
                  str(cnt), ha='center', va='bottom',
                  fontsize=8, fontweight='bold')
    for i, lbl in enumerate(LABELS):
        if lbl in FIXED:
            ax_m.text(i, 3, "⚡", ha='center', fontsize=11, color='orange')
    ax_m.axhline(np.mean(multi_counts), color='green', lw=2, linestyle='--',
                 label=f'Mean: {np.mean(multi_counts):.0f}')
    ax_m.set_title("Multi-Class — 20 Izhikevich Neuron Types",
                   fontweight='bold', fontsize=12)
    ax_m.set_xlabel("Label (A–T)"); ax_m.set_ylabel("Sample Count")
    ax_m.set_ylim(0, max(multi_counts) * 1.20)
    ax_m.legend(handles=[
        Patch(color='#2E75B6', label='Regular'),
        Patch(color='#C00000', label='Irregular'),
        plt.Line2D([0],[0], color='green', lw=2, linestyle='--',
                   label=f'Mean: {np.mean(multi_counts):.0f}')
    ], fontsize=9)
    ax_m.grid(True, alpha=0.3, axis='y')
    for sp in ['top','right']: ax_m.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[9]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[9]}")

# ── GRAPH 10 — Class Weight Visualization ────────────────────
if 10 in to_do:
    print("  Graph 10: Class Weight Visualization ...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Class Weight Visualization — DB-HLSTM  [compute_class_weight]\n"
                 "Binary (Left)  |  Multi-Class 20 Labels (Right)",
                 fontsize=14, fontweight='bold', color='#1F4E79')

    # Left — Binary class weights
    ax_b      = axes[0]
    cw_bin    = compute_class_weight('balanced',
                                     classes=np.array([0, 1]),
                                     y=y_bin_all)
    bin_labels = ['Regular', 'Irregular']
    bin_colors = ['#2E75B6', '#C00000']
    b_bars = ax_b.bar(bin_labels, cw_bin, color=bin_colors,
                      width=0.45, edgecolor='white')
    for bar, w in zip(b_bars, cw_bin):
        ax_b.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + max(cw_bin) * 0.02,
                  f"{w:.4f}", ha='center', va='bottom',
                  fontsize=13, fontweight='bold')
    ax_b.axhline(1.0, color='gray', lw=1.5, linestyle='--', label='Balanced = 1.0')
    ax_b.set_title("Binary — Class Weights", fontweight='bold', fontsize=12)
    ax_b.set_ylabel("Class Weight", fontsize=11, fontweight='bold')
    ax_b.set_ylim(0, max(cw_bin) * 1.25)
    ax_b.legend(fontsize=10)
    ax_b.grid(True, alpha=0.3, axis='y')
    ax_b.tick_params(axis='x', labelsize=12)
    for sp in ['top','right']: ax_b.spines[sp].set_visible(False)

    # Right — Multi-class weights (20 labels)
    ax_m    = axes[1]
    cw_multi = compute_class_weight('balanced',
                                    classes=np.arange(20),
                                    y=y_int_all)
    mc_colors = ['#C00000' if label_mode.get(l,'')=='Irregular'
                 else '#2E75B6' for l in LABELS]
    m_bars = ax_m.bar(LABELS, cw_multi, color=mc_colors,
                      width=0.65, edgecolor='white')
    for bar, w in zip(m_bars, cw_multi):
        ax_m.text(bar.get_x() + bar.get_width()/2,
                  bar.get_height() + max(cw_multi) * 0.01,
                  f"{w:.2f}", ha='center', va='bottom',
                  fontsize=8, fontweight='bold')
    for i, lbl in enumerate(LABELS):
        if lbl in FIXED:
            ax_m.text(i, min(cw_multi) * 0.5, "⚡",
                      ha='center', fontsize=11, color='orange')
    ax_m.axhline(1.0, color='gray', lw=1.5, linestyle='--', label='Balanced = 1.0')
    ax_m.set_title("Multi-Class — Per-Label Class Weights",
                   fontweight='bold', fontsize=12)
    ax_m.set_xlabel("Label (A–T)"); ax_m.set_ylabel("Class Weight")
    ax_m.set_ylim(0, max(cw_multi) * 1.20)
    ax_m.legend(handles=[
        Patch(color='#2E75B6', label='Regular'),
        Patch(color='#C00000', label='Irregular'),
        plt.Line2D([0],[0], color='gray', lw=1.5, linestyle='--',
                   label='Balanced = 1.0')
    ], fontsize=9)
    ax_m.grid(True, alpha=0.3, axis='y')
    for sp in ['top','right']: ax_m.spines[sp].set_visible(False)

    fig.tight_layout()
    fig.savefig(f"{OUT_DIR}/{GRAPH_FILES[10]}", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✅ {GRAPH_FILES[10]}")

print()
print("="*60)
waiting = [n for n, s in status.items() if s == 'no_model']
print(f"  Generated : {to_do}")
if waiting:
    print(f"  Waiting   : {waiting}")
print(f"\n  Folder: {OUT_DIR}")
print("="*60)
