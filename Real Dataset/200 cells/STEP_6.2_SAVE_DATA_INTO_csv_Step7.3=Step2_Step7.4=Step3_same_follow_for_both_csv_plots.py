"""
============================================================
 Step 1 — Real NWB to Izhikevich Format CSV
 DB-HLSTM Framework

 Converts 300 real NWB cells to same format as
 Izhikevich synthetic CSV so same HRF and Balloon
 codes can be used directly.

 OUTPUT FORMAT (same as Neural_Research_Data_Final.csv):
   Label, Mode, Class, Sample_No, a, b, c, d, I_val,
   v_0..v_599, u_0..u_599, i_0..i_599, s_0..s_599,
   s_t_count

 INPUT  : /kaggle/working/nwb_cells/cell_*.nwb
 OUTPUT : /kaggle/working/Neural_Research_Data_Final.csv

 20-class labeling via KMeans on CV values
 CV Threshold = 0.35 (Xi-Barrier)
============================================================
"""

import os
import h5py
import csv
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from tqdm.notebook import tqdm

# ============================================================
# CONFIGURATION
# ============================================================
NWB_DIR      = '/kaggle/working/nwb_cells/'
OUTPUT_CSV   = '/kaggle/working/Neural_Research_Data_Final.csv'
SEQ_LEN      = 600
CV_THRESHOLD = 0.35
N_CLASSES    = 20

# ============================================================
# STEP A — COLLECT ALL SWEEPS + CV VALUES
# ============================================================
print("=" * 60)
print("  Step 1 — NWB to Izhikevich Format")
print(f"  SEQ_LEN    : {SEQ_LEN}")
print(f"  CV Threshold: {CV_THRESHOLD}")
print(f"  Classes    : {N_CLASSES}")
print("=" * 60)

nwb_files = sorted([f for f in os.listdir(NWB_DIR) if f.endswith('.nwb')])
print(f"\n  NWB files found: {len(nwb_files)}")

# First pass — collect CV values for KMeans
print("\n  Pass 1 — Collecting CV values for KMeans clustering ...")

all_sweeps = []

for nwb_file in tqdm(nwb_files, desc="Pass 1"):
    cell_id = nwb_file.replace('cell_', '').replace('.nwb', '')
    fpath   = os.path.join(NWB_DIR, nwb_file)

    try:
        with h5py.File(fpath, 'r') as f:
            if 'spike_times' not in f['analysis']:
                continue

            spk_grp = f['analysis']['spike_times']
            sweeps  = list(f['acquisition']['timeseries'].keys())

            for sw in sweeps:
                try:
                    raw       = f['acquisition']['timeseries'][sw]['data'][:]
                    data_mv   = raw * 1000.0
                    idx       = np.linspace(0, len(data_mv)-1, SEQ_LEN).astype(int)
                    resampled = data_mv[idx].astype(np.float32)

                    spks = spk_grp[sw][:] if sw in spk_grp else np.array([])
                    n    = len(spks)

                    if n > 2:
                        isi = np.diff(spks)
                        cv  = float(np.std(isi) / (np.mean(isi) + 1e-10))
                    else:
                        cv  = 0.0

                    spike_train = (resampled > 0.0).astype(np.int8)
                    vmin        = float(resampled.min())
                    mode        = 'Irregular' if cv >= CV_THRESHOLD else 'Regular'
                    cls_bin     = 1 if cv >= CV_THRESHOLD else 0

                    all_sweeps.append({
                        'cell_id'    : cell_id,
                        'sweep_id'   : sw,
                        'cv'         : cv,
                        'vmin'       : vmin,
                        'spike_count': n,
                        'mode'       : mode,
                        'class_bin'  : cls_bin,
                        'resampled'  : resampled,
                        'spike_train': spike_train,
                    })

                except Exception:
                    continue

    except Exception:
        continue

print(f"\n  Total sweeps collected: {len(all_sweeps):,}")

# ============================================================
# STEP B — 20-CLASS LABELING VIA KMEANS ON CV
# ============================================================
print("\n  Pass 2 — KMeans 20-class labeling on CV values ...")

cvs    = np.array([s['cv'] for s in all_sweeps]).reshape(-1, 1)
kmeans = KMeans(n_clusters=N_CLASSES, random_state=42, n_init=10)
labels = kmeans.fit_predict(cvs)

for i, sw in enumerate(all_sweeps):
    sw['class20'] = int(labels[i])

# Show class distribution
print("\n  20-class distribution:")
from collections import Counter
dist = Counter([s['class20'] for s in all_sweeps])
for cls in sorted(dist.keys()):
    cv_mean = np.mean([s['cv'] for s in all_sweeps if s['class20'] == cls])
    print(f"    Class {cls:2d}: {dist[cls]:5,} sweeps  MeanCV={cv_mean:.3f}")

# Binary distribution
reg = sum(1 for s in all_sweeps if s['class_bin'] == 0)
irr = sum(1 for s in all_sweeps if s['class_bin'] == 1)
print(f"\n  Binary — Regular: {reg:,}  Irregular: {irr:,}")

# ============================================================
# STEP C — WRITE IZHIKEVICH FORMAT CSV
# ============================================================
print(f"\n  Pass 3 — Writing Izhikevich format CSV ...")
print(f"  Output: {OUTPUT_CSV}")

header = (
    ['Label', 'Mode', 'Class', 'Sample_No', 'a', 'b', 'c', 'd', 'I_val'] +
    [f"v_{i}" for i in range(SEQ_LEN)] +
    [f"u_{i}" for i in range(SEQ_LEN)] +
    [f"i_{i}" for i in range(SEQ_LEN)] +
    [f"s_{i}" for i in range(SEQ_LEN)] +
    ['s_t_count']
)

print(f"  Columns: {len(header):,}  (same as synthetic: 2,410)")

zeros = np.zeros(SEQ_LEN, dtype=np.float32)

with open(OUTPUT_CSV, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(header)

    for sample_no, sw in enumerate(tqdm(all_sweeps, desc="Writing CSV")):
        row = (
            # Metadata — same format as Izhikevich
            [f"Real_{sw['class20']}",  # Label  (e.g. Real_0, Real_1...)
             sw['mode'],               # Mode   (Regular / Irregular)
             sw['class20'],            # Class  (0-19)
             sample_no,                # Sample_No
             0.0,                      # a — not applicable for real data
             0.0,                      # b
             sw['vmin'],               # c — using Vmin as biological equivalent
             float(sw['spike_count']), # d — using spike count
             sw['cv']] +               # I_val — using CV value
            list(np.round(sw['resampled'], 4)) +   # v_0..v_599 — real voltage
            list(np.round(zeros, 4)) +             # u_0..u_599 — zeros
            list(np.round(zeros, 4)) +             # i_0..i_599 — zeros
            list(sw['spike_train'].astype(int)) +  # s_0..s_599 — real spikes
            [sw['spike_count']]                    # s_t_count
        )
        writer.writerow(row)

# ============================================================
# VERIFICATION
# ============================================================
print("\n  Verifying output ...")
df         = pd.read_csv(OUTPUT_CSV, nrows=5)
total_rows = sum(1 for _ in open(OUTPUT_CSV)) - 1
total_cols = len(df.columns)

print()
print("=" * 60)
print("  CONVERSION COMPLETE")
print("=" * 60)
print(f"  Total rows    : {total_rows:,}")
print(f"  Total columns : {total_cols:,}  (expected 2,410)")
print(f"  Output        : {OUTPUT_CSV}")
print(f"  Regular       : {reg:,}")
print(f"  Irregular     : {irr:,}")
print()
print("  Sample rows:")
print(df[['Label','Mode','Class','Sample_No','s_t_count']].to_string())
print()
print("  Now run:")
print("  → kaggle_hrf_extraction_professional.py")
print("  → kaggle_balloon_windkessel_professional.py")
print("  → LSTM Binary + Multiclass")
print("=" * 60)