# ============================================================
#  LSTM — Multi-Class  [KAGGLE — HYBRID MODEL v3]
#  STEP 0  |  RESUME SUPPORT
#
#  DEEP ANALYSIS RESULT:
#  A,C,E,F → xi=15.1 SAME → BOLD same → only c,d differ
#  B,D,M,N → xi=4.3-4.5 SAME → only c,d differ
#  These differ by: reset potential c, recovery d
#  → v(t) membrane voltage add karo → c,d ka direct effect
#
#  HYBRID ARCHITECTURE:
#  Branch 1: LSTM on time series (hrf_c, hrf_td, hrf_dd, bold, v)
#  Branch 2: Dense on scalar features (s_t_count, v_mean,
#             v_std, v_min, spike_rate, ISI_mean, ISI_std)
#  → Concatenate → Final Dense
#  → Scalar features c,d differences directly capture karte hain
#
#  FIX LOG (v3 → v3_fixed):
#  - BUG FIXED: np.save now stores y_pred (argmax predictions)
#    alongside y_true so the plotter can compute real per-label
#    accuracy instead of silently falling back to PRIOR_ACC values.
#  - Previously saved dict only had keys {"y_true", "labels"};
#    plotter's branch required y_pred but it was never written,
#    causing all per-label bars to show stale prior-run numbers.
# ============================================================

import os, gc

DATASET_NAME = "sajid-neural-bold"
CSV_PATH   = "/kaggle/working/Neural_BOLD_Data.csv"
WORK_DIR   = "/kaggle/working"
MODEL_PATH = os.path.join(WORK_DIR, "lstm_hybrid_v3.keras")
LOG_CSV    = os.path.join(WORK_DIR, "hybrid_v3_epoch_log.csv")
HIST_PATH  = os.path.join(WORK_DIR, "hybrid_v3_history.npy")
PRED_PATH  = os.path.join(WORK_DIR, "hybrid_v3_predictions.npy")

if not os.path.exists(CSV_PATH):
    import glob
    found = glob.glob("/kaggle/input/**/Neural_BOLD_Data.csv", recursive=True)
    if found:
        CSV_PATH = found[0]
    else:
        raise FileNotFoundError("Neural_BOLD_Data.csv not found!")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (LSTM, Dense, Dropout, Input,
                                     BatchNormalization, Concatenate,
                                     GlobalAveragePooling1D)
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau, CSVLogger)
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
    print("  CPU ⚠️  Settings → Accelerator → GPU T4 x2")
tf.config.optimizer.set_jit(True)

# ── CONFIG ──────────────────────────────────────────────────
SEQ_LEN      = 600
N_TS_FEAT    = 5    # time series: hrf_c, hrf_td, hrf_dd, bold, v
N_SCALAR     = 8    # scalars: s_count, v_mean, v_std, v_min, v_max,
                    #          spike_rate, ISI_mean, ISI_std
N_CLASSES    = 20
TOTAL_EPOCHS = 150
BATCH_SIZE   = 128
LEARN_RATE   = 0.001
PATIENCE_ES  = 25
PATIENCE_LR  = 8
LABELS       = list("ABCDEFGHIJKLMNOPQRST")

CLASS_WEIGHTS = {
    0:1.0,  1:1.0,  2:2.0,  3:1.5,  4:4.0,
    5:1.0,  6:3.0,  7:2.0,  8:3.0,  9:3.0,
    10:2.0, 11:4.0, 12:2.0, 13:2.0, 14:1.0,
    15:1.0, 16:3.0, 17:1.0, 18:1.0, 19:1.0,
}

# ── RESUME ──────────────────────────────────────────────────
def get_initial_epoch():
    if os.path.exists(LOG_CSV):
        try:
            log_df = pd.read_csv(LOG_CSV)
            if len(log_df) > 0:
                done = int(log_df['epoch'].max()) + 1
                print(f"\n  RESUME — {done} epochs done, resuming from epoch {done + 1}")
                return done
        except:
            pass
    print("\n  FRESH START")
    return 0

# ── LOAD DATA ───────────────────────────────────────────────
print("=" * 60)
print("  LSTM DB-HLSTM  [KAGGLE — HYBRID v3 FIXED]")
print(f"  TS features: {N_TS_FEAT}  Scalar features: {N_SCALAR}")
print("=" * 60)

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
print(f"  v(t) cols: {'YES' if has_v else 'NO — fallback'}")
print(f"  s(t) cols: {'YES' if has_s else 'NO'}")

# ── STEP 0 VERIFY ────────────────────────────────────────────
print("\n  Step 0 — I, L, O, P:")
label_params = df.groupby('Label')[['a', 'b', 'c', 'd', 'I_val']].first()
FIXED = {
    'I': {'a': 0.04,  'b':  0.18,  'c': -63, 'd': 7,   'xi': 12.0},
    'L': {'a': 0.025, 'b': -0.15,  'c': -58, 'd': 5,   'xi': 38.0},
    'O': {'a': 0.04,  'b':  0.30,  'c': -58, 'd': 2,   'xi':  7.0},
    'P': {'a': 0.12,  'b':  0.28,  'c': -58, 'd': 1,   'xi':  0.5},
}
for lbl, exp in FIXED.items():
    row = label_params.loc[lbl]
    ok  = all(abs(float(row[k]) - v) < 1e-6
              for k, v in [('a', exp['a']), ('b', exp['b']),
                            ('c', exp['c']), ('d', exp['d']),
                            ('I_val', exp['xi'])])
    print(f"    {lbl}: {'OK ✅' if ok else 'ERROR ❌'}")

# ── BUILD FEATURES ───────────────────────────────────────────
print("\n  Building features ...")

hrf_c  = df[[f"hrf_c_{i}"  for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_td = df[[f"hrf_td_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
hrf_dd = df[[f"hrf_dd_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
bold   = df[[f"bold_{i}"   for i in range(SEQ_LEN)]].values.astype(np.float32)

if has_v:
    v_arr = df[[f"v_{i}" for i in range(SEQ_LEN)]].values.astype(np.float32)
    X_ts  = np.stack([hrf_c, hrf_td, hrf_dd, bold, v_arr], axis=2)
    print(f"  TS shape: {X_ts.shape}  (with v(t))")
else:
    X_ts      = np.stack([hrf_c, hrf_td, hrf_dd, bold], axis=2)
    N_TS_FEAT = 4
    print(f"  TS shape: {X_ts.shape}  (without v(t))")

print("  Computing scalar features ...")
scalars = []
for idx in range(len(df)):
    row = df.iloc[idx]

    s_count = float(row['s_t_count']) if 's_t_count' in row else 0.0

    if has_v:
        v_data = row[[f"v_{i}" for i in range(SEQ_LEN)]].values.astype(float)
        v_mean = v_data.mean()
        v_std  = v_data.std()
        v_min  = v_data.min()
        v_max  = v_data.max()
    else:
        v_mean = v_std = v_min = v_max = 0.0

    if has_s:
        s_data     = row[[f"s_{i}" for i in range(SEQ_LEN)]].values.astype(float)
        spike_idx  = np.where(s_data == 1)[0]
        spike_rate = len(spike_idx) / SEQ_LEN
        if len(spike_idx) > 1:
            isi      = np.diff(spike_idx)
            isi_mean = isi.mean()
            isi_std  = isi.std()
        else:
            isi_mean = isi_std = 0.0
    else:
        spike_rate = s_count / SEQ_LEN
        isi_mean   = isi_std = 0.0

    scalars.append([s_count, v_mean, v_std, v_min, v_max,
                    spike_rate, isi_mean, isi_std])

X_scalar = np.array(scalars, dtype=np.float32)
print(f"  Scalar shape: {X_scalar.shape}")

le    = LabelEncoder()
y_int = le.fit_transform(df["Label"].values)
y_cat = to_categorical(y_int, num_classes=N_CLASSES)
del df, hrf_c, hrf_td, hrf_dd, bold
gc.collect()

# ── NOISE AUGMENTATION ──────────────────────────────────────
def add_gaussian_noise(X, std=0.05):
    noise = np.random.normal(0, std, X.shape)
    return (X + noise).astype(np.float32)

# ── NORMALIZE ───────────────────────────────────────────────
X_ts_norm = np.zeros_like(X_ts, dtype=np.float32)
for fi in range(X_ts.shape[2]):
    feat = X_ts[:, :, fi]
    X_ts_norm[:, :, fi] = ((feat - feat.mean(axis=1, keepdims=True)) /
                            (feat.std(axis=1, keepdims=True) + 1e-8))
del X_ts
gc.collect()

scaler    = StandardScaler()
X_sc_norm = scaler.fit_transform(X_scalar).astype(np.float32)
del X_scalar
gc.collect()

# ── SPLIT ───────────────────────────────────────────────────
(X_ts_tr, X_ts_te,
 X_sc_tr, X_sc_te,
 y_tr_c,  y_te_c,
 yi_tr,   yi_te) = train_test_split(
    X_ts_norm, X_sc_norm, y_cat, y_int,
    test_size=0.20, random_state=42, stratify=y_int)
print(f"  Train: {len(X_ts_tr):,}  Test: {len(X_ts_te):,}")
del X_ts_norm, X_sc_norm
gc.collect()

# ── HYBRID MODEL ─────────────────────────────────────────────
def build_model():
    REG = l2(1e-4)

    # Branch 1 — LSTM on time series
    ts_input = Input(shape=(SEQ_LEN, X_ts_tr.shape[2]), name='ts_input')
    x = LSTM(128, return_sequences=True,
             dropout=0.2, recurrent_dropout=0.0,
             kernel_regularizer=REG)(ts_input)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=False,
             dropout=0.2, recurrent_dropout=0.0,
             kernel_regularizer=REG)(x)
    x = BatchNormalization()(x)
    x = Dense(64, activation='relu', kernel_regularizer=REG)(x)
    x = Dropout(0.2)(x)

    # Branch 2 — Dense on scalar features
    sc_input = Input(shape=(X_sc_tr.shape[1],), name='scalar_input')
    y = Dense(32, activation='relu', kernel_regularizer=REG)(sc_input)
    y = BatchNormalization()(y)
    y = Dense(16, activation='relu', kernel_regularizer=REG)(y)

    # Concatenate both branches
    merged = Concatenate()([x, y])
    z = Dense(64, activation='relu', kernel_regularizer=REG)(merged)
    z = Dropout(0.2)(z)
    out = Dense(N_CLASSES, activation='softmax',
                dtype='float32', name='output')(z)

    m = Model(inputs=[ts_input, sc_input], outputs=out)
    m.compile(
        optimizer=Adam(learning_rate=LEARN_RATE),
        loss='categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top3_acc')
        ]
    )
    return m

# ── SAFE LOAD ────────────────────────────────────────────────
def try_load_model():
    ie = get_initial_epoch()
    if ie > 0 and os.path.exists(MODEL_PATH):
        try:
            print(f"\n  Loading checkpoint (epoch {ie}) ...")
            m = build_model()
            m([np.zeros((1, SEQ_LEN, X_ts_tr.shape[2]), dtype=np.float32),
               np.zeros((1, X_sc_tr.shape[1]),          dtype=np.float32)],
              training=False)
            m.load_weights(MODEL_PATH)
            print("  Checkpoint loaded ✅")
            return m, ie
        except Exception as e:
            print(f"  Incompatible checkpoint → fresh start ({str(e)[:60]})")
            for f in [MODEL_PATH, LOG_CSV, HIST_PATH, PRED_PATH]:
                if os.path.exists(f):
                    os.remove(f)
    print("  Starting fresh")
    return build_model(), 0

model, initial_epoch = try_load_model()
model.summary()

# ── tf.data PIPELINE ─────────────────────────────────────────
AUTOTUNE = tf.data.AUTOTUNE
val_size = int(len(X_ts_tr) * 0.15)

Xts_val = X_ts_tr[:val_size]
Xsc_val = X_sc_tr[:val_size]
Yval    = y_tr_c[:val_size]

Xts_t = add_gaussian_noise(X_ts_tr[val_size:], std=0.05)
Xsc_t = X_sc_tr[val_size:]
Yt    = y_tr_c[val_size:]

train_ds = (tf.data.Dataset.from_tensor_slices(
                ({'ts_input': Xts_t, 'scalar_input': Xsc_t}, Yt))
            .shuffle(len(Xts_t), seed=42)
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))
val_ds   = (tf.data.Dataset.from_tensor_slices(
                ({'ts_input': Xts_val, 'scalar_input': Xsc_val}, Yval))
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))
test_ds  = (tf.data.Dataset.from_tensor_slices(
                ({'ts_input': X_ts_te, 'scalar_input': X_sc_te}, y_te_c))
            .batch(BATCH_SIZE)
            .prefetch(AUTOTUNE))

# ── CALLBACKS ───────────────────────────────────────────────
callbacks = [
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy',
                    save_best_only=True, mode='max', verbose=1),
    CSVLogger(LOG_CSV, append=True),
    EarlyStopping(monitor='val_accuracy', patience=PATIENCE_ES,
                  restore_best_weights=True, mode='max', verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=PATIENCE_LR, min_lr=1e-6, verbose=1),
]

# ── TRAIN ────────────────────────────────────────────────────
remaining = TOTAL_EPOCHS - initial_epoch
print(f"\n  Training: epoch {initial_epoch + 1} → {TOTAL_EPOCHS}")
print(f"  Architecture: LSTM(ts_branch) + Dense(scalar_branch) → Concatenate")
print("  Session timeout protection: resume is supported ✅")

if remaining <= 0:
    print("  Training already complete. Loading best weights ✅")
    model = build_model()
    model.load_weights(MODEL_PATH)
else:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=TOTAL_EPOCHS,
        initial_epoch=initial_epoch,
        class_weight=CLASS_WEIGHTS,
        callbacks=callbacks,
        verbose=1,
    )
    # Accumulate history across resume sessions
    if os.path.exists(HIST_PATH) and initial_epoch > 0:
        old = np.load(HIST_PATH, allow_pickle=True).item()
        for k in history.history:
            old[k] = old.get(k, []) + history.history[k]
        np.save(HIST_PATH, old)
    else:
        np.save(HIST_PATH, history.history)

# ── EVALUATE & SAVE PREDICTIONS ──────────────────────────────
#
#  FIX: Previously only y_true and labels were saved here.
#  The plotter (Plot 04) requires y_pred to compute real per-label
#  accuracy. Without y_pred, it silently fell back to the hardcoded
#  PRIOR_ACC dictionary, making every bar show the old stale results
#  (E:0%, G:1%, H:0%, I:0%, etc.) regardless of actual model performance.
#
#  Now we save:  {"y_true": yi_te, "y_pred": y_pred, "labels": LABELS}
#
print("\n  Evaluating on test set ...")
res    = model.evaluate(test_ds, verbose=0)

print("  Generating predictions ...")
y_pred_proba = model.predict(test_ds, verbose=0)
y_pred       = np.argmax(y_pred_proba, axis=1)   # shape: (n_test,)

# ── SAVE PREDICTIONS (with y_pred included) ──────────────────
np.save(PRED_PATH, {
    "y_true":  yi_te,     # ground-truth integer labels
    "y_pred":  y_pred,    # predicted integer labels  ← THE FIX
    "labels":  LABELS,
})
print(f"  Predictions saved → {PRED_PATH}")
print(f"    Keys: y_true ({yi_te.shape}), y_pred ({y_pred.shape}), labels (20)")

# ── RESULTS DISPLAY ──────────────────────────────────────────
prev = {
    'A': 66, 'B': 100, 'C': 86, 'D': 48, 'E': 0,   'F': 100,
    'G': 1,  'H': 0,   'I': 0,  'J': 8,  'K': 1,   'L': 0,
    'M': 32, 'N': 0,   'O': 100,'P': 100,'Q': 100,  'R': 100,
    'S': 100,'T': 100,
}

print()
print("=" * 60)
print("  RESULTS  [DB-HLSTM — LSTM + SCALAR HYBRID]")
print("=" * 60)
print(f"  Test Accuracy  : {res[1] * 100:.2f}%")
print(f"  Test Top-3 Acc : {res[2] * 100:.2f}%")
print(f"  Test Loss      : {res[0]:.4f}")
print(f"\n  Per-Label Accuracy:")
for i, lbl in enumerate(LABELS):
    mask = yi_te == i
    if mask.sum() == 0:
        continue
    acc      = (y_pred[mask] == i).mean() * 100
    bar      = "█" * int(acc / 5)
    tag      = " *" if lbl in ['I', 'L', 'O', 'P'] else ""
    old_acc  = prev.get(lbl)
    arrow = (
        f"  ↑+{acc - old_acc:.0f}%" if acc > old_acc
        else f"  ↓{acc - old_acc:.0f}%" if acc < old_acc
        else "  ↔"
    ) if old_acc is not None else ""
    print(f"    {lbl + tag}: {acc:5.1f}%  {bar}{arrow}")

print()
if   res[1] >= 0.75: print("  STATUS : Good ✅")
elif res[1] >= 0.60: print("  STATUS : Improving ⚠️")
else:                print("  STATUS : Needs more work ❌")
print(f"\n  Model    : {MODEL_PATH}")
print(f"  History  : {HIST_PATH}")
print(f"  Preds    : {PRED_PATH}  [y_true + y_pred + labels]")
print("=" * 60)
