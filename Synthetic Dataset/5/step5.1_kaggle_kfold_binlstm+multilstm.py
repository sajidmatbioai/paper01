"""
============================================================
 K-FOLD CROSS VALIDATION — DB-HLSTM
 Binary + Multi-Class
 
 Output (only these 2 numbers):
   Binary:      XX.XX% ± X.XX%
   Multi-Class: XX.XX% ± X.XX%

 Run on Kaggle — same CSV will be used:
   /kaggle/working/Neural_BOLD_Data.csv
============================================================
"""

import os, gc
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     BatchNormalization, Concatenate)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

# ── GPU ─────────────────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"  GPU: {len(gpus)}  float16: ON ✅")
else:
    print("  CPU mode ⚠️")
tf.config.optimizer.set_jit(True)

# ── CONFIG ──────────────────────────────────────────────────
CSV_PATH   = "/kaggle/working/Neural_BOLD_Data.csv"
SEQ_LEN    = 600
N_FOLDS    = 5        # 5-Fold CV
N_CLASSES  = 20
BATCH_SIZE = 128
LEARN_RATE = 0.001
EPOCHS     = 100      # Max epochs per fold
PATIENCE   = 15       # Early stopping
LABELS     = list("ABCDEFGHIJKLMNOPQRST")

print("=" * 60)
print("  K-FOLD CV — DB-HLSTM  [Binary + Multi-Class]")
print(f"  Folds: {N_FOLDS}  Epochs/fold: {EPOCHS}  Batch: {BATCH_SIZE}")
print("=" * 60)

# ── LOAD DATA ───────────────────────────────────────────────
print("\n  Loading data ...")
needed_cols = (
    ['Label', 'Mode', 'Class', 'a', 'b', 'c', 'd', 'I_val', 's_t_count'] +
    [f"v_{i}"      for i in range(SEQ_LEN)] +
    [f"s_{i}"      for i in range(SEQ_LEN)] +
    [f"hrf_c_{i}"  for i in range(SEQ_LEN)] +
    [f"hrf_td_{i}" for i in range(SEQ_LEN)] +
    [f"hrf_dd_{i}" for i in range(SEQ_LEN)] +
    [f"bold_{i}"   for i in range(SEQ_LEN)]
)
header      = pd.read_csv(CSV_PATH, nrows=0).columns.tolist()
cols_to_use = [c for c in needed_cols if c in header]
df          = pd.read_csv(CSV_PATH, usecols=cols_to_use)
print(f"  Rows: {len(df):,}")

has_v = "v_0" in df.columns
has_s = "s_0" in df.columns

# ── SCALAR FEATURES ─────────────────────────────────────────
def compute_scalars(dataframe):
    scalars = []
    for idx in range(len(dataframe)):
        row     = dataframe.iloc[idx]
        s_count = float(row.get('s_t_count', 0))
        if has_v:
            v       = row[[f"v_{i}" for i in range(SEQ_LEN)]].values.astype(float)
            v_mean, v_std, v_min, v_max = v.mean(), v.std(), v.min(), v.max()
        else:
            v_mean = v_std = v_min = v_max = 0.0
        if has_s:
            s    = row[[f"s_{i}" for i in range(SEQ_LEN)]].values.astype(float)
            spk  = np.where(s == 1)[0]
            spike_rate = len(spk) / SEQ_LEN
            if len(spk) > 1:
                isi = np.diff(spk)
                isi_mean, isi_std = isi.mean(), isi.std()
            else:
                isi_mean = isi_std = 0.0
        else:
            spike_rate = s_count / SEQ_LEN
            isi_mean   = isi_std = 0.0
        scalars.append([s_count, v_mean, v_std, v_min, v_max,
                        spike_rate, isi_mean, isi_std])
    return np.array(scalars, dtype=np.float32)

# ── BUILD FEATURES ───────────────────────────────────────────
print("  Building features ...")
hrf_c  = df[[f"hrf_c_{i}"  for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_td = df[[f"hrf_td_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_dd = df[[f"hrf_dd_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
bold   = df[[f"bold_{i}"   for i in range(SEQ_LEN)]].values.astype(np.float32)
v_ts   = df[[f"v_{i}"      for i in range(SEQ_LEN)]].values.astype(np.float32) \
         if has_v else np.zeros_like(bold)

X_ts  = np.stack([hrf_c, hrf_td, hrf_dd, bold, v_ts], axis=2)
X_sc  = compute_scalars(df)

# Labels
y_binary = df["Class"].values                          # 0/1
le       = LabelEncoder()
y_multi  = le.fit_transform(df["Label"].values)        # 0-19

del df, hrf_c, hrf_td, hrf_dd, bold, v_ts; gc.collect()
print(f"  X_ts: {X_ts.shape}  X_sc: {X_sc.shape}")

# ── NORMALIZE TIME SERIES ────────────────────────────────────
X_ts_norm = np.zeros_like(X_ts, dtype=np.float32)
for fi in range(X_ts.shape[2]):
    feat = X_ts[:, :, fi]
    X_ts_norm[:, :, fi] = ((feat - feat.mean(axis=1, keepdims=True)) /
                            (feat.std(axis=1, keepdims=True) + 1e-8))
del X_ts; gc.collect()

# ── MODEL BUILDERS ───────────────────────────────────────────
def build_binary_model():
    REG    = l2(1e-4)
    ts_in  = Input(shape=(SEQ_LEN, 5), name='ts_input')
    x = LSTM(128, return_sequences=True, dropout=0.2,
             kernel_regularizer=REG)(ts_in)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.2,
             kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=REG)(x)
    x = Dropout(0.2)(x)

    sc_in = Input(shape=(8,), name='scalar_input')
    y_ = Dense(32, activation='relu', kernel_regularizer=REG)(sc_in)
    y_ = BatchNormalization()(y_)
    y_ = Dense(16, activation='relu', kernel_regularizer=REG)(y_)

    merged = Concatenate()([x, y_])
    z   = Dense(32, activation='relu', kernel_regularizer=REG)(merged)
    z   = Dropout(0.2)(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(z)

    m = Model(inputs=[ts_in, sc_in], outputs=out)
    m.compile(optimizer=Adam(LEARN_RATE),
              loss='binary_crossentropy',
              metrics=['accuracy'])
    return m

def build_multi_model():
    REG    = l2(1e-4)
    ts_in  = Input(shape=(SEQ_LEN, 5), name='ts_input')
    x = LSTM(128, return_sequences=True, dropout=0.2,
             kernel_regularizer=REG)(ts_in)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.2,
             kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=REG)(x)
    x = Dropout(0.2)(x)

    sc_in = Input(shape=(8,), name='scalar_input')
    y_ = Dense(32, activation='relu', kernel_regularizer=REG)(sc_in)
    y_ = BatchNormalization()(y_)
    y_ = Dense(16, activation='relu', kernel_regularizer=REG)(y_)

    merged = Concatenate()([x, y_])
    z   = Dense(64, activation='relu', kernel_regularizer=REG)(merged)
    z   = Dropout(0.2)(z)
    out = Dense(N_CLASSES, activation='softmax', dtype='float32')(z)

    m = Model(inputs=[ts_in, sc_in], outputs=out)
    m.compile(optimizer=Adam(LEARN_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

# ── K-FOLD FUNCTION ──────────────────────────────────────────
def run_kfold(task='binary'):
    """
    task = 'binary'  → binary classification
    task = 'multi'   → 20-class classification
    """
    print(f"\n{'='*60}")
    print(f"  {N_FOLDS}-FOLD CV — {'BINARY' if task=='binary' else 'MULTI-CLASS'}")
    print(f"{'='*60}")

    y_labels = y_binary if task == 'binary' else y_multi
    skf      = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_accs = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X_ts_norm, y_labels)):
        print(f"\n  ── Fold {fold+1}/{N_FOLDS} ──")

        # Split
        Xts_tr, Xts_te = X_ts_norm[train_idx], X_ts_norm[test_idx]
        Xsc_tr, Xsc_te = X_sc[train_idx],      X_sc[test_idx]
        y_tr,   y_te   = y_labels[train_idx],   y_labels[test_idx]

        # Normalize scalars — fit on train only
        sc_scaler = StandardScaler()
        Xsc_tr_n  = sc_scaler.fit_transform(Xsc_tr).astype(np.float32)
        Xsc_te_n  = sc_scaler.transform(Xsc_te).astype(np.float32)

        # Prepare labels
        if task == 'binary':
            y_tr_in = y_tr
            y_te_in = y_te
        else:
            y_tr_in = to_categorical(y_tr, N_CLASSES)
            y_te_in = to_categorical(y_te, N_CLASSES)

        # Add noise to training data only
        noise = np.random.normal(0, 0.02, Xts_tr.shape).astype(np.float32)
        Xts_tr_noisy = Xts_tr + noise

        # Build fresh model each fold
        tf.keras.backend.clear_session()
        model = build_binary_model() if task == 'binary' else build_multi_model()

        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=PATIENCE,
                          restore_best_weights=True, mode='max', verbose=0),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=6, min_lr=1e-6, verbose=0)
        ]

        # Train
        model.fit(
            {'ts_input': Xts_tr_noisy, 'scalar_input': Xsc_tr_n},
            y_tr_in,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=0     # quiet — only final result will be shown
        )

        # Evaluate
        res = model.evaluate(
            {'ts_input': Xts_te, 'scalar_input': Xsc_te_n},
            y_te_in,
            verbose=0
        )
        acc = res[1] * 100
        fold_accs.append(acc)
        print(f"  Fold {fold+1} Accuracy: {acc:.2f}%")

        del model, Xts_tr, Xts_te, Xsc_tr, Xsc_te
        del Xts_tr_noisy, noise
        gc.collect()
        tf.keras.backend.clear_session()

    # Final Result
    mean_acc = np.mean(fold_accs)
    std_acc  = np.std(fold_accs)
    print(f"\n  ── {'Binary' if task=='binary' else 'Multi-Class'} K-Fold Result ──")
    for i, a in enumerate(fold_accs):
        print(f"     Fold {i+1}: {a:.2f}%")
    print(f"  ─────────────────────────────")
    print(f"  Mean ± Std: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Min: {min(fold_accs):.2f}%   Max: {max(fold_accs):.2f}%")
    return mean_acc, std_acc, fold_accs

# ══════════════════════════════════════════════════════════════
#  MAIN — RUN BOTH
# ══════════════════════════════════════════════════════════════

# 1) Binary K-Fold
bin_mean, bin_std, bin_folds = run_kfold(task='binary')

# 2) Multi-Class K-Fold
multi_mean, multi_std, multi_folds = run_kfold(task='multi')

# ── FINAL SUMMARY (To be included in paper) ──────────────────
print("\n")
print("=" * 60)
print("  FINAL K-FOLD RESULTS — TO BE INCLUDED IN PAPER")
print("=" * 60)
print(f"  Binary Classification:")
print(f"    {N_FOLDS}-Fold Mean Accuracy : {bin_mean:.2f}% ± {bin_std:.2f}%")
print(f"")
print(f"  Multi-Class Classification (20 labels):")
print(f"    {N_FOLDS}-Fold Mean Accuracy : {multi_mean:.2f}% ± {multi_std:.2f}%")
print("=" * 60)
print("\n  Add the following sentence to the paper:")
print(f"  'Five-fold cross-validation yielded {bin_mean:.1f}% ± {bin_std:.1f}%")
print(f"   binary accuracy and {multi_mean:.1f}% ± {multi_std:.1f}%")
print(f"   multi-class accuracy, confirming result stability.'")
print("=" * 60)

# Save results to CSV
results_df = pd.DataFrame({
    'Fold':       list(range(1, N_FOLDS+1)),
    'Binary_Acc': bin_folds,
    'Multi_Acc':  multi_folds
})
results_df.loc[len(results_df)] = ['Mean ± Std',
                                    f"{bin_mean:.2f}±{bin_std:.2f}",
                                    f"{multi_mean:.2f}±{multi_std:.2f}"]
results_df.to_csv("/kaggle/working/kfold_results.csv", index=False)
print("\n  Results saved: /kaggle/working/kfold_results.csv ✅")