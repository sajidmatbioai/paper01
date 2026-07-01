"""
============================================================
 Step 1.2 — HRF Extraction  [KAGGLE VERSION]

 Convolves spike train s(t) with 3 HRF kernels:
   h_c  (t) = Canonical HRF          → y_c  = s(t) * h_c
   h_td (t) = Temporal Derivative    → y_td = s(t) * h_td
   h_dd (t) = Dispersion Derivative  → y_dd = s(t) * h_dd

 Based on: Friston et al. (1998) — SPM HRF model
   Canonical: h(t) = Gp(t) - C·Gu(t)
   where Gp, Gu are Gamma distributions
   a1=6, a2=16, b1=1, b2=1, C=1/6

 INPUT  : /kaggle/working/Neural_Research_Data_Final.csv
 OUTPUT : /kaggle/working/Neural_HRF_Data.csv
   New columns: hrf_c_0..599, hrf_td_0..599, hrf_dd_0..599
   Total columns: 2,410 + 1,800 = 4,210
============================================================
"""

import numpy as np
import pandas as pd
from scipy.special import gamma as Gamma
from scipy.signal import convolve
from tqdm.notebook import tqdm
import os, gc

# ── PATHS ───────────────────────────────────────────────────
CSV_IN  = "/kaggle/working/Neural_Research_Data_Final.csv"
CSV_OUT = "/kaggle/working/Neural_HRF_Data.csv"

SEQ_LEN = 600
h_step  = 0.5                             # ms per step
t_ms    = np.arange(SEQ_LEN) * h_step     # 0.0 to 299.5 ms
t_s     = t_ms / 1000.0                   # convert to seconds

# ── HRF PARAMETERS (Friston 1998) ───────────────────────────
a1, a2 = 6.0, 16.0
b1, b2 = 1.0,  1.0
C       = 1.0 / 6.0

# ════════════════════════════════════════════════════════════
#  HRF KERNEL FUNCTIONS
# ════════════════════════════════════════════════════════════
def h_canonical(t):
    """
    Canonical HRF: difference of two Gamma functions.
    Models the positive BOLD peak (~6s) and undershoot (~16s).
    h(t) = Gp(t) - C·Gu(t)
    """
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    Gu = (t**(a2-1) * b2**a2 * np.exp(-b2*t)) / Gamma(a2)
    return Gp - C * Gu

def h_temporal(t):
    """
    Temporal Derivative: dh/dt
    Captures timing shifts in the BOLD response.
    TD = Gp·[(a1-1)/t - b1] - C·Gu·[(a2-1)/t - b2]
    """
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    Gu = (t**(a2-1) * b2**a2 * np.exp(-b2*t)) / Gamma(a2)
    return Gp*((a1-1)/t - b1) - C*Gu*((a2-1)/t - b2)

def h_dispersion(t):
    """
    Dispersion Derivative: dh/db1
    Captures width variations in the BOLD response.
    DD = Gp·[a1/b1 - t]
    """
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    return Gp * (a1/b1 - t)

# Compute and normalize kernels
kernel_c  = h_canonical(t_s)
kernel_td = h_temporal(t_s)
kernel_dd = h_dispersion(t_s)

for k in [kernel_c, kernel_td, kernel_dd]:
    k /= (np.max(np.abs(k)) + 1e-10)

print("=" * 60)
print("  Step 1.2 — HRF Extraction  [KAGGLE]")
print(f"  Input  : {CSV_IN}")
print(f"  Output : {CSV_OUT}")
print("=" * 60)
print(f"\n  HRF kernel peaks (ms):")
print(f"    Canonical  : {t_ms[np.argmax(kernel_c)]:.1f} ms")
print(f"    Temporal D : {t_ms[np.argmax(kernel_td)]:.1f} ms")
print(f"    Dispersion : {t_ms[np.argmax(kernel_dd)]:.1f} ms")

# ── LOAD CSV ─────────────────────────────────────────────────
print(f"\n  Loading CSV ...")
df = pd.read_csv(CSV_IN)
print(f"  Rows : {len(df):,}  |  Cols : {len(df.columns):,}")

s_cols = [f"s_{i}" for i in range(SEQ_LEN)]
assert all(c in df.columns for c in s_cols[:3]), "s(t) columns missing!"
print(f"  s(t) columns : OK ✅")

# ── CONVOLUTION ──────────────────────────────────────────────
print(f"\n  Convolving {len(df):,} spike trains with 3 HRF kernels ...")

yc_all  = np.zeros((len(df), SEQ_LEN), dtype=np.float32)
ytd_all = np.zeros((len(df), SEQ_LEN), dtype=np.float32)
ydd_all = np.zeros((len(df), SEQ_LEN), dtype=np.float32)

for idx in tqdm(range(len(df)), desc="HRF Convolution"):
    s = df.iloc[idx][s_cols].values.astype(float)
    yc_all[idx]  = convolve(s, kernel_c,  mode='full')[:SEQ_LEN]
    ytd_all[idx] = convolve(s, kernel_td, mode='full')[:SEQ_LEN]
    ydd_all[idx] = convolve(s, kernel_dd, mode='full')[:SEQ_LEN]

# ── ADD COLUMNS ──────────────────────────────────────────────
print(f"\n  Adding 1,800 new columns ...")
for i in tqdm(range(SEQ_LEN), desc="Adding columns"):
    df[f"hrf_c_{i}"]  = yc_all[:, i]
    df[f"hrf_td_{i}"] = ytd_all[:, i]
    df[f"hrf_dd_{i}"] = ydd_all[:, i]

del yc_all, ytd_all, ydd_all; gc.collect()

# ── SAVE ─────────────────────────────────────────────────────
print(f"\n  Saving → {CSV_OUT}")
df.to_csv(CSV_OUT, index=False)

print()
print("=" * 60)
print("  HRF EXTRACTION COMPLETE")
print("=" * 60)
print(f"  Rows    : {len(df):,}   (expected 10,000)")
print(f"  Columns : {len(df.columns):,}   (expected 4,210)")
print(f"  Output  : {CSV_OUT}")
print()
for lbl in list("ABCDEFGHIJKLMNOPQRST"):
    cnt = len(df[df['Label']==lbl])
    print(f"  {lbl}: {cnt} samples  {'✅' if cnt==500 else '❌'}", end="  ")
    if list("ABCDEFGHIJKLMNOPQRST").index(lbl) % 5 == 4: print()
print("\n" + "=" * 60)