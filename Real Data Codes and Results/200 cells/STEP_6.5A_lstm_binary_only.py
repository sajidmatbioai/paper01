"""
============================================================
 LSTM — Binary Classification ONLY
 DB-HLSTM Framework — Real Data

 Input   : /kaggle/working/Neural_BOLD_Data.csv
 Features: hrf_c + hrf_td + hrf_dd + bold + v(t) = 5
 Window  : 100, Stride: 10
 GPU     : Tesla P100 — float16 — Batch 512

 Output  : /kaggle/working/lstm_real_binary.keras
============================================================
"""

import os
import gc
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     BatchNormalization, Concatenate)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tqdm.notebook import tqdm

# ============================================================
# GPU SETUP — P100
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print("  GPU: Tesla P100  float16: ON ✅")
else:
    print("  CPU only ⚠️")
tf.config.optimizer.set_jit(True)

# ============================================================
# CONFIGURATION
# ============================================================
CSV_PATH     = '/kaggle/working/Neural_BOLD_Data.csv'
BINARY_MODEL = '/kaggle/working/lstm_real_binary.keras'

SEQ_LEN    = 600
WINDOW     = 100
STRIDE     = 10
N_CLASSES  = 20
MIN_PER_CLS= 500
MAX_PER_CLS= 500
BATCH_SIZE = 512
LEARN_RATE = 0.0005
EPOCHS     = 150
PATIENCE_ES= 20
PATIENCE_LR= 8

# ============================================================
# STEP 1 — LOAD DATA
# ============================================================
print("=" * 60)
print("  LSTM BINARY — Real Data  [DB-HLSTM — GPU P100]")
print(f"  Window={WINDOW}  Stride={STRIDE}  Batch={BATCH_SIZE}")
print("=" * 60)

print("\n  Loading Neural_BOLD_Data.csv ...")
df = pd.read_csv(CSV_PATH)
print(f"  Rows     : {len(df):,}")
print(f"  Columns  : {len(df.columns):,}")

# ============================================================
# FIX — BINARY LABEL BANANA
# ============================================================
print("\n  Class column unique values (raw):", sorted(df['Class'].unique()))
df['Binary'] = (df['Class'] != 0).astype(int)
print(f"  Binary label rule: Class==0 → 0 (Regular), Class!=0 → 1 (Irregular)")
print(f"  Regular   (0): {(df['Binary']==0).sum():,}")
print(f"  Irregular (1): {(df['Binary']==1).sum():,}")

# ============================================================
# STEP 2 — 20-CLASS LABELING VIA KMEANS (balance ke liye)
# ============================================================
print("\n  Creating 20-class labels via KMeans on CV ...")
cvs          = df['I_val'].values.reshape(-1, 1)
kmeans       = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=10)
df['Class20']= kmeans.fit_predict(cvs)

# ============================================================
# STEP 3 — BALANCE
# ============================================================
print(f"\n  Balancing — {MIN_PER_CLS} samples per class ...")
dfs = []
for cls in range(N_CLASSES):
    cls_df = df[df['Class20'] == cls]
    if len(cls_df) < MIN_PER_CLS:
        cls_df = cls_df.sample(MIN_PER_CLS, replace=True,  random_state=42)
    else:
        cls_df = cls_df.sample(MAX_PER_CLS, replace=False, random_state=42)
    dfs.append(cls_df)

df_bal = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42)
print(f"  Balanced total: {len(df_bal):,}")
del df; gc.collect()

# ============================================================
# STEP 4 — WINDOWING
# ============================================================
print(f"\n  Windowing (window={WINDOW}, stride={STRIDE}) ...")

bold_cols   = [f'bold_{i}'   for i in range(SEQ_LEN)]
hrf_c_cols  = [f'hrf_c_{i}'  for i in range(SEQ_LEN)]
hrf_td_cols = [f'hrf_td_{i}' for i in range(SEQ_LEN)]
hrf_dd_cols = [f'hrf_dd_{i}' for i in range(SEQ_LEN)]
v_cols      = [f'v_{i}'      for i in range(SEQ_LEN)]

n_windows = (SEQ_LEN - WINDOW) // STRIDE + 1
n_samples = len(df_bal) * n_windows
print(f"  Windows per sweep : {n_windows}")
print(f"  Total samples     : {n_samples:,}")

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
print(f"  y_bin unique: {np.unique(y_bin[:idx])}")  # must be [0 1]

# ============================================================
# STEP 5 — FEATURE STACK + NORMALIZE
# ============================================================
print("\n  Stacking and normalizing features ...")

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

print(f"  X_norm shape : {X_norm.shape}")
print(f"  X_sc shape   : {X_sc.shape}")

# ============================================================
# STEP 6 — BUILD MODEL
# ============================================================
def build_model():
    REG   = l2(1e-4)
    ts_in = Input(shape=(WINDOW, 5), name='ts_input')

    x = LSTM(256, return_sequences=True, dropout=0.2,
             kernel_regularizer=REG)(ts_in)
    x = BatchNormalization()(x)
    x = LSTM(128, return_sequences=True, dropout=0.2,
             kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.2,
             kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=REG)(x)
    x = Dropout(0.2)(x)

    sc_in = Input(shape=(8,), name='scalar_input')
    y_    = Dense(64, activation='relu', kernel_regularizer=REG)(sc_in)
    y_    = BatchNormalization()(y_)
    y_    = Dense(32, activation='relu', kernel_regularizer=REG)(y_)

    z   = Concatenate()([x, y_])
    z   = Dense(64, activation='relu', kernel_regularizer=REG)(z)
    z   = Dropout(0.2)(z)

    out  = Dense(1, activation='sigmoid', dtype='float32')(z)
    loss = 'binary_crossentropy'
    mets = ['accuracy',
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall')]

    m = Model(inputs=[ts_in, sc_in], outputs=out)
    m.compile(optimizer=Adam(LEARN_RATE), loss=loss, metrics=mets)
    return m

AUTOTUNE = tf.data.AUTOTUNE

def make_datasets(X_ts, X_sc, y, batch):
    n      = len(X_ts)
    val_n  = int(n * 0.15)
    test_n = int(n * 0.10)
    tr_n   = n - val_n - test_n

    tr_ds = (tf.data.Dataset.from_tensor_slices(
        ({'ts_input': X_ts[:tr_n], 'scalar_input': X_sc[:tr_n]}, y[:tr_n]))
        .shuffle(tr_n, seed=42).batch(batch).prefetch(AUTOTUNE))
    vl_ds = (tf.data.Dataset.from_tensor_slices(
        ({'ts_input': X_ts[tr_n:tr_n+val_n],
          'scalar_input': X_sc[tr_n:tr_n+val_n]},
         y[tr_n:tr_n+val_n]))
        .batch(batch).prefetch(AUTOTUNE))
    te_ds = (tf.data.Dataset.from_tensor_slices(
        ({'ts_input': X_ts[tr_n+val_n:],
          'scalar_input': X_sc[tr_n+val_n:]},
         y[tr_n+val_n:]))
        .batch(batch).prefetch(AUTOTUNE))
    y_te  = y[tr_n+val_n:]
    return tr_ds, vl_ds, te_ds, y_te

# ============================================================
# STEP 7 — BINARY TRAINING
# ============================================================
print("\n" + "=" * 60)
print("  BINARY CLASSIFICATION — Regular vs Irregular")
print("=" * 60)

n0 = (y_bin==0).sum(); n1 = (y_bin==1).sum()
cw = {0: len(y_bin)/(2*n0), 1: len(y_bin)/(2*n1)}
print(f"  Class weights — Regular: {cw[0]:.3f}  Irregular: {cw[1]:.3f}")

tr_b, vl_b, te_b, yte_b = make_datasets(X_norm, X_sc, y_bin, BATCH_SIZE)

# RAM free karo — sirf binary data chahiye
del y_bin; gc.collect()

model_b = build_model()
model_b.summary()

cb_b = [
    ModelCheckpoint(BINARY_MODEL, monitor='val_auc',
                    save_best_only=True, mode='max', verbose=1),
    EarlyStopping(monitor='val_auc', patience=PATIENCE_ES,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=PATIENCE_LR, min_lr=1e-6, verbose=1),
    CSVLogger('/kaggle/working/binary_real_log.csv')
]

model_b.fit(
    tr_b, validation_data=vl_b,
    epochs=EPOCHS, callbacks=cb_b,
    class_weight=cw, verbose=1
)

res_b = model_b.evaluate(te_b, verbose=0)
print()
print("=" * 60)
print("  BINARY RESULTS")
print("=" * 60)
print(f"  Accuracy  : {res_b[1]*100:.2f}%")
print(f"  AUC       : {res_b[2]:.4f}")
print(f"  Precision : {res_b[3]:.4f}")
print(f"  Recall    : {res_b[4]:.4f}")
print(f"  Loss      : {res_b[0]:.4f}")
if   res_b[1]>=0.90: print("  STATUS : Excellent ✅")
elif res_b[1]>=0.85: print("  STATUS : Good ✅")
else:                print("  STATUS : Acceptable ✅")
print("=" * 60)

# RAM completely free karo
del model_b, tr_b, vl_b, te_b, X_norm, X_sc
gc.collect()
tf.keras.backend.clear_session()

print("\n  BINARY DONE ✅")
print(f"  Model saved: {BINARY_MODEL}")
print(f"  Log saved  : /kaggle/working/binary_real_log.csv")
