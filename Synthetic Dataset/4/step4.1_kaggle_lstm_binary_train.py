"""
============================================================
 LSTM — Binary Classification  [KAGGLE VERSION]
 DB-HLSTM PARAMETERS — Step 0 Fixed (I,L,O,P)
 STEP 1 IMPROVEMENTS APPLIED

 Regular (Class 0) vs Irregular (Class 1)
 Input   : /kaggle/working/Neural_BOLD_Data.csv
 Features: hrf_c + hrf_td + hrf_dd + bold + v(t) = 5 features
 Scalar  : v_mean, v_std, v_min, v_max,
           spike_count, spike_rate, ISI_mean, ISI_std = 8 scalars
 Output  : /kaggle/working/lstm_binary_hybrid.keras

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
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     BatchNormalization, Concatenate)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

# ── PATHS ───────────────────────────────────────────────────
CSV_PATH   = "/kaggle/working/Neural_BOLD_Data.csv"
WORK_DIR   = "/kaggle/working"
MODEL_PATH = os.path.join(WORK_DIR, "lstm_binary_hybrid.keras")
LOG_CSV    = os.path.join(WORK_DIR, "binary_epoch_log.csv")
HIST_PATH  = os.path.join(WORK_DIR, "binary_history.npy")

# ── GPU CONFIGURATION ────────────────────────────────────────
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"  GPU: {len(gpus)}  float16: ON ✅")
else:
    print("  CPU ⚠️  Settings → Accelerator → GPU T4 x2")
tf.config.optimizer.set_jit(True)

# ── CONFIGURATION — STEP 1 ───────────────────────────────────
SEQ_LEN      = 600
TOTAL_EPOCHS = 80
BATCH_SIZE   = 128
LEARN_RATE   = 0.001
PATIENCE_ES  = 15
PATIENCE_LR  = 6
LABELS       = list("ABCDEFGHIJKLMNOPQRST")

# ── STEP 0 FIXED PARAMETERS ──────────────────────────────────
FIXED = {
    'I': {'a':0.04,  'b':0.18,  'c':-63, 'd':7, 'xi':12.0},
    'L': {'a':0.025, 'b':-0.15, 'c':-58, 'd':5, 'xi':38.0},
    'O': {'a':0.04,  'b':0.30,  'c':-58, 'd':2, 'xi':7.0 },
    'P': {'a':0.12,  'b':0.28,  'c':-58, 'd':1, 'xi':0.5 },
}

# ── RESUME TRAINING ──────────────────────────────────────────
def get_initial_epoch():
    if os.path.exists(LOG_CSV):
        try:
            log_df = pd.read_csv(LOG_CSV)
            if len(log_df) > 0:
                done = int(log_df['epoch'].max()) + 1
                print(f"\n  RESUME — {done} epochs completed, continuing from epoch {done+1}")
                return done
        except: pass
    print("\n  FRESH START")
    return 0

# ── SCALAR FEATURE EXTRACTION ────────────────────────────────
def compute_scalars(df):
    s_cols = [f"s_{i}" for i in range(SEQ_LEN)]
    v_cols = [f"v_{i}" for i in range(SEQ_LEN)]
    has_v  = "v_0" in df.columns
    has_s  = "s_0" in df.columns
    scalars = []
    for idx in range(len(df)):
        row = df.iloc[idx]
        s_count = float(row.get('s_t_count', 0))
        if has_v:
            v = row[v_cols].values.astype(float)
            v_mean,v_std,v_min,v_max = v.mean(),v.std(),v.min(),v.max()
        else:
            v_mean=v_std=v_min=v_max=0.0
        if has_s:
            s = row[s_cols].values.astype(float)
            spk = np.where(s==1)[0]
            spike_rate = len(spk)/SEQ_LEN
            if len(spk)>1:
                isi=np.diff(spk); isi_mean=isi.mean(); isi_std=isi.std()
            else:
                isi_mean=isi_std=0.0
        else:
            spike_rate=s_count/SEQ_LEN; isi_mean=isi_std=0.0
        scalars.append([s_count,v_mean,v_std,v_min,v_max,
                        spike_rate,isi_mean,isi_std])
    return np.array(scalars, dtype=np.float32)

# ── LOAD DATA ────────────────────────────────────────────────
print("=" * 60)
print("  LSTM Binary — DB-HLSTM  [KAGGLE — STEP 0+1]")
print(f"  LR={LEARN_RATE}  Epochs={TOTAL_EPOCHS}  Batch={BATCH_SIZE}")
print("=" * 60)

needed_cols = (
    ['Label','Mode','Class','a','b','c','d','I_val','s_t_count'] +
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
print(f"  Rows: {len(df):,}  Cols loaded: {len(df.columns):,}")

# ── STEP 0 PARAMETER VERIFICATION ───────────────────────────
print("\n  Step 0 — Fixed labels verification:")
lp = df.groupby('Label')[['a','b','c','d','I_val']].first()
for lbl, exp in FIXED.items():
    if lbl not in lp.index: continue
    row = lp.loc[lbl]
    ok  = all(abs(float(row[k])-v)<1e-6
              for k,v in [('a',exp['a']),('b',exp['b']),
                           ('c',exp['c']),('d',exp['d']),
                           ('I_val',exp['xi'])])
    print(f"    {lbl}: {'OK ✅' if ok else 'ERROR ❌'}")

# ── FEATURE CONSTRUCTION ─────────────────────────────────────
print("\n  Building features ...")
hrf_c  = df[[f"hrf_c_{i}"  for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_td = df[[f"hrf_td_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_dd = df[[f"hrf_dd_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
bold   = df[[f"bold_{i}"   for i in range(SEQ_LEN)]].values.astype(np.float32)
v_ts   = df[[f"v_{i}"      for i in range(SEQ_LEN)]].values.astype(np.float32) \
         if "v_0" in df.columns else np.zeros_like(bold)

X_ts = np.stack([hrf_c,hrf_td,hrf_dd,bold,v_ts], axis=2)
y    = df["Class"].values
print(f"  X_ts shape: {X_ts.shape}")

X_sc = compute_scalars(df)
del df, hrf_c, hrf_td, hrf_dd, bold, v_ts; gc.collect()

# ── NORMALIZATION ────────────────────────────────────────────
X_norm = np.zeros_like(X_ts, dtype=np.float32)
for fi in range(X_ts.shape[2]):
    feat = X_ts[:,:,fi]
    X_norm[:,:,fi] = ((feat-feat.mean(axis=1,keepdims=True)) /
                      (feat.std(axis=1,keepdims=True)+1e-8))
del X_ts; gc.collect()

scaler  = StandardScaler()
X_sc_n  = scaler.fit_transform(X_sc).astype(np.float32)
del X_sc; gc.collect()

# ── TRAIN / TEST SPLIT ───────────────────────────────────────
(X_ts_tr, X_ts_te,
 X_sc_tr, X_sc_te,
 y_tr,    y_te) = train_test_split(
    X_norm, X_sc_n, y,
    test_size=0.20, random_state=42, stratify=y)
print(f"  Train: {len(X_ts_tr):,}  Test: {len(X_ts_te):,}")
del X_norm, X_sc_n; gc.collect()

# ── tf.data PIPELINE ─────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
val_size = int(len(X_ts_tr)*0.15)
train_ds = (tf.data.Dataset.from_tensor_slices((
                {'ts_input':X_ts_tr[val_size:],
                 'scalar_input':X_sc_tr[val_size:]}, y_tr[val_size:]))
            .shuffle(len(y_tr)-val_size, seed=42)
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices((
                {'ts_input':X_ts_tr[:val_size],
                 'scalar_input':X_sc_tr[:val_size]}, y_tr[:val_size]))
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))
test_ds  = (tf.data.Dataset.from_tensor_slices((
                {'ts_input':X_ts_te,
                 'scalar_input':X_sc_te}, y_te))
            .batch(BATCH_SIZE).prefetch(AUTOTUNE))

# ── MODEL DEFINITION ─────────────────────────────────────────
def build_model():
    REG = l2(1e-4)
    ts_in = Input(shape=(SEQ_LEN,5), name='ts_input')
    x = LSTM(128, return_sequences=True, dropout=0.2,
             recurrent_dropout=0.0, kernel_regularizer=REG)(ts_in)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=False, dropout=0.2,
             recurrent_dropout=0.0, kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=REG)(x)
    x = Dropout(0.2)(x)

    sc_in = Input(shape=(8,), name='scalar_input')
    y_ = Dense(32, activation='relu', kernel_regularizer=REG)(sc_in)
    y_ = BatchNormalization()(y_)
    y_ = Dense(16, activation='relu', kernel_regularizer=REG)(y_)

    merged = Concatenate()([x, y_])
    z = Dense(32, activation='relu', kernel_regularizer=REG)(merged)
    z = Dropout(0.2)(z)
    out = Dense(1, activation='sigmoid', dtype='float32')(z)

    m = Model(inputs=[ts_in, sc_in], outputs=out)
    m.compile(
        optimizer=Adam(LEARN_RATE),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )
    return m

# ── MODEL LOADING / INITIALIZATION ───────────────────────────
initial_epoch = get_initial_epoch()
if initial_epoch > 0 and os.path.exists(MODEL_PATH):
    try:
        print(f"\n  Loading saved model (epoch {initial_epoch}) ...")
        model = build_model()
        model([np.zeros((1,SEQ_LEN,5),dtype=np.float32),
               np.zeros((1,8),dtype=np.float32)], training=False)
        model.load_weights(MODEL_PATH)
        print("  Loaded ✅")
    except Exception as e:
        print(f"  Incompatible checkpoint — starting fresh")
        for f in [MODEL_PATH,LOG_CSV,HIST_PATH]:
            if os.path.exists(f): os.remove(f)
        model = build_model(); initial_epoch = 0
else:
    initial_epoch = 0
    model = build_model()

model.summary()

callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_auc',
                    save_best_only=True, mode='max', verbose=1),
    CSVLogger(LOG_CSV, append=True),
    EarlyStopping(monitor='val_auc', patience=PATIENCE_ES,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=PATIENCE_LR, min_lr=1e-6, verbose=1)
]

remaining = TOTAL_EPOCHS - initial_epoch
print(f"\n  Training: epoch {initial_epoch+1} → {TOTAL_EPOCHS}")
print("  Session timeout → resume ✅")

if remaining <= 0:
    print("  Training complete! ✅"); model.load_weights(MODEL_PATH)
else:
    history = model.fit(
        train_ds, validation_data=val_ds,
        epochs=TOTAL_EPOCHS, initial_epoch=initial_epoch,
        callbacks=callbacks, verbose=1
    )
    if os.path.exists(HIST_PATH) and initial_epoch > 0:
        old = np.load(HIST_PATH, allow_pickle=True).item()
        for k in history.history:
            old[k] = old.get(k,[]) + history.history[k]
        np.save(HIST_PATH, old)
    else:
        np.save(HIST_PATH, history.history)

# ── EVALUATION ───────────────────────────────────────────────
res = model.evaluate(test_ds, verbose=0)
print()
print("=" * 60)
print("  BINARY RESULTS  [DB-HLSTM — STEP 0+1]")
print("=" * 60)
print(f"  Test Accuracy  : {res[1]*100:.2f}%")
print(f"  Test AUC       : {res[2]:.4f}")
print(f"  Test Precision : {res[3]:.4f}")
print(f"  Test Recall    : {res[4]:.4f}")
print(f"  Test Loss      : {res[0]:.4f}")
if   res[1]>=0.93: print("  STATUS : Excellent ✅")
elif res[1]>=0.85: print("  STATUS : Good ✅")
else:              print("  STATUS : Needs improvement ⚠️")
print(f"\n  Model: {MODEL_PATH}")
print("=" * 60)