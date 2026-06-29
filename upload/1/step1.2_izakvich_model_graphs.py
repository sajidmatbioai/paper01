"""
============================================================
 Izhikevich Results Plotter — KAGGLE VERSION
 
 Loads Neural_Research_Data_Final.csv and generates
 diagnostic plots for Regular and Irregular spiking patterns.
 
 PLOTS:
   1. Regular vs Irregular — membrane potential (v) comparison
   2. Spike raster for each label
   3. CV distribution — Regular/Irregular boundary
   4. Spike count per label (bar chart)
   5. Phasic vs Tonic — waveform overlay
   6. ISI (Inter-Spike Interval) histogram
 
 INPUT:  /kaggle/working/Neural_Research_Data_Final.csv
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import warnings
warnings.filterwarnings('ignore')

# ── GLOBAL STYLE (publication quality) ───────────────────────
plt.rcParams.update({
    'figure.facecolor':  'white',
    'axes.facecolor':    'white',
    'axes.edgecolor':    '#333333',
    'axes.linewidth':    1.1,
    'axes.labelweight':  'bold',
    'font.size':          10,
    'savefig.facecolor': 'white',
})

# ── CONFIG ──────────────────────────────────────────────────
CSV_PATH  = "/kaggle/working/Neural_Research_Data_Final.csv"
SEQ_LEN   = 600
H         = 0.5          # time step (ms)
T         = np.arange(0, SEQ_LEN) * H   # 0..299.5 ms
CV_THRESH = 0.35

# ── COLORS (deeper, more saturated — print/journal safe) ─────
COL_REG = "#1f5fb8"   # Regular   — strong royal blue
COL_IRR = "#b3202a"   # Irregular — deep crimson red
COL_INP = "#ffb703"   # Input window highlight — vivid amber
COL_GRID = "#999999"  # neutral grid
COL_THRESH = "#e8743b"  # threshold line — burnt orange (distinct from red/blue)

# ── LABEL INFO ──────────────────────────────────────────────
LABEL_NAMES = {
    'A':'Tonic Spiking',      'B':'Phasic Spiking',
    'C':'Tonic Bursting',     'D':'Phasic Bursting',
    'E':'Mixed Mode',         'F':'Spike Freq Adapt',
    'G':'Class 1 Excitable',  'H':'Class 2 Excitable',
    'I':'Spike Latency',      'J':'Subthresh Osc',
    'K':'Resonator',          'L':'Integrator',
    'M':'Rebound Spike',      'N':'Rebound Burst',
    'O':'Threshold Var',      'P':'Bistability',
    'Q':'Depol After-Pot',    'R':'Accommodation',
    'S':'Inhib-Induced Spk',  'T':'Inhib-Induced Burst',
}

# ── HELPER — extract v columns ──────────────────────────────
V_COLS = [f"v_{i}" for i in range(SEQ_LEN)]
S_COLS = [f"s_{i}" for i in range(SEQ_LEN)]

print("=" * 60)
print("  Loading CSV...")
print("=" * 60)

df = pd.read_csv(CSV_PATH)
print(f"  Shape     : {df.shape}")
print(f"  Labels    : {sorted(df['Label'].unique())}")
print(f"  Modes     : {df['Mode'].value_counts().to_dict()}")
print()

# ────────────────────────────────────────────────────────────
#  PLOT 1 — Regular vs Irregular Membrane Potential
#  Display one Regular and one Irregular sample side by side
# ────────────────────────────────────────────────────────────
print("  Plot 1: Regular vs Irregular comparison...")

fig, axes = plt.subplots(2, 2, figsize=(16, 8))
fig.suptitle("Regular vs Irregular Spiking — Membrane Potential (v)",
             fontsize=15, fontweight='bold', y=1.01)

# Retrieve one Regular and one Irregular sample
reg_row = df[df['Mode'] == 'Regular'].iloc[0]
irr_row = df[df['Mode'] == 'Irregular'].iloc[0]

for row_idx, (row, mode, color) in enumerate([
    (reg_row, "Regular",   COL_REG),
    (irr_row, "Irregular", COL_IRR)
]):
    v   = row[V_COLS].values.astype(float)
    s   = row[S_COLS].values.astype(int)
    lbl  = row['Label']
    name = LABEL_NAMES.get(lbl, lbl)

    # Membrane potential
    ax = axes[row_idx][0]
    ax.axvspan(50, 250, color=COL_INP, alpha=0.30, label='Input window')
    ax.plot(T, v, color=color, lw=2.0, label=f"{lbl}: {name}")
    ax.axhline(30, color='#444444', lw=1.1, ls='--', alpha=0.8, label='Spike threshold (30 mV)')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("v (mV)")
    ax.set_title(f"{mode} — Label {lbl}: {name}", fontweight='bold')
    ax.legend(fontsize=8.5, loc='upper right', framealpha=0.95)
    ax.set_xlim(0, 300)
    ax.grid(True, alpha=0.35, color=COL_GRID)

    # Spike raster
    ax2 = axes[row_idx][1]
    spike_times = T[s == 1]
    ax2.axvspan(50, 250, color=COL_INP, alpha=0.30, label='Input window')
    ax2.vlines(spike_times, 0, 1, color=color, lw=2.2, label=f"Spikes (n={len(spike_times)})")
    ax2.set_xlabel("Time (ms)")
    ax2.set_ylabel("Spike")
    ax2.set_title(f"{mode} — Spike Raster", fontweight='bold')
    ax2.set_xlim(0, 300)
    ax2.set_ylim(-0.1, 1.3)
    ax2.legend(fontsize=8.5, framealpha=0.95)
    ax2.grid(True, alpha=0.35, color=COL_GRID)

plt.tight_layout()
plt.savefig("/kaggle/working/plot1_regular_vs_irregular.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: plot1_regular_vs_irregular.png")

# ────────────────────────────────────────────────────────────
#  PLOT 2 — Waveform Grid for All 20 Labels
# ────────────────────────────────────────────────────────────
print("  Plot 2: All 20 labels waveform grid...")

labels_sorted = sorted(df['Label'].unique())
n_labels = len(labels_sorted)
ncols = 4
nrows = (n_labels + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.2))
fig.suptitle("All 20 Labels — Membrane Potential v(t)",
             fontsize=15, fontweight='bold')

axes_flat = axes.flatten()

for idx, lbl in enumerate(labels_sorted):
    ax  = axes_flat[idx]
    sub = df[df['Label'] == lbl]

    # Plot the first available sample
    row   = sub.iloc[0]
    v     = row[V_COLS].values.astype(float)
    mode  = row['Mode']
    color = COL_REG if mode == 'Regular' else COL_IRR

    ax.axvspan(50, 250, color=COL_INP, alpha=0.28)
    ax.plot(T, v, color=color, lw=1.6)
    ax.axhline(30, color='#444444', lw=0.8, ls='--', alpha=0.7)

    name = LABEL_NAMES.get(lbl, lbl)
    ax.set_title(f"{lbl}: {name}\n[{mode}]", fontsize=8.5, fontweight='bold',
                 color=color)
    ax.set_xlim(0, 300)
    ax.set_xlabel("ms", fontsize=7)
    ax.set_ylabel("mV", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.30, color=COL_GRID)

# Hide unused axes
for idx in range(n_labels, len(axes_flat)):
    axes_flat[idx].set_visible(False)

# Legend
legend_elements = [
    Patch(facecolor=COL_REG, label='Regular (CV < 0.35)'),
    Patch(facecolor=COL_IRR, label='Irregular (CV ≥ 0.35)'),
    Patch(facecolor=COL_INP, alpha=0.4, label='Input window (50–250 ms)'),
]
fig.legend(handles=legend_elements, loc='lower right',
           fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig("/kaggle/working/plot2_all_labels_grid.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: plot2_all_labels_grid.png")

# ────────────────────────────────────────────────────────────
#  # ────────────────────────────────────────────────────────────
#  PLOT 3 — CV Distribution + Regular/Irregular Boundary
# ────────────────────────────────────────────────────────────
print("  Plot 3: CV distribution per label...")

from scipy.stats import gaussian_kde

cv_data = []
for lbl in labels_sorted:
    sub = df[df['Label'] == lbl]
    for _, row in sub.iterrows():
        s       = row[S_COLS].values.astype(int)
        spk_idx = np.where(s == 1)[0]
        if len(spk_idx) > 2:
            isi = np.diff(spk_idx)
            cv  = np.std(isi) / (np.mean(isi) + 1e-10)
        else:
            cv = 999.0
        cv_data.append({'Label': lbl, 'CV': min(cv, 3.0),
                        'Mode': row['Mode']})

cv_df = pd.DataFrame(cv_data)

fig = plt.figure(figsize=(22, 7))
gs  = gridspec.GridSpec(1, 2, wspace=0.38)
fig.suptitle("CV (Coefficient of Variation) Analysis", fontsize=20, fontweight='bold', y=0.995)

# ── Top: Lollipop chart (mean CV per label) ──────────────────
ax = fig.add_subplot(gs[0])

mean_cv = cv_df.groupby('Label')['CV'].mean().reindex(labels_sorted)
std_cv  = cv_df.groupby('Label')['CV'].std().reindex(labels_sorted)
x_pos   = np.arange(len(labels_sorted))
dot_colors = [COL_REG if mean_cv[lbl] < CV_THRESH else COL_IRR for lbl in labels_sorted]

ax.vlines(x_pos, 0, mean_cv.values, color=dot_colors, lw=2.2, alpha=0.85)
ax.errorbar(x_pos, mean_cv.values, yerr=std_cv.values,
            fmt='none', ecolor='#333333', elinewidth=1.6,
            capsize=5, capthick=1.6, zorder=2)
ax.scatter(x_pos, mean_cv.values, c=dot_colors, s=180,
           edgecolor='#000000', linewidth=1.4, zorder=3)
ax.axhline(CV_THRESH, color=COL_THRESH, lw=2.2, ls='--',
           label=f'CV threshold = {CV_THRESH}', zorder=0)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels_sorted, fontsize=13, fontweight='bold')
ax.set_xlabel("Label", fontsize=14, labelpad=10)
ax.set_ylabel("Mean CV (±std)", fontsize=14)
ax.set_title("Mean CV per Label", fontsize=16, fontweight='bold', pad=12)
ax.tick_params(axis='y', labelsize=12)
ax.grid(True, alpha=0.35, axis='y', color=COL_GRID)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements_lollipop = [
    Patch(facecolor=COL_REG, alpha=0.85, edgecolor='#222222', label='Regular (mean CV < 0.35)'),
    Patch(facecolor=COL_IRR, alpha=0.85, edgecolor='#222222', label='Irregular (mean CV ≥ 0.35)'),
]
from matplotlib.lines import Line2D
thresh_line = Line2D([0], [0], color=COL_THRESH, lw=2.2, ls='--', label=f'Threshold = {CV_THRESH}')
ax.legend(handles=legend_elements_lollipop + [thresh_line],
          fontsize=11.5, loc='upper left', framealpha=0.95)

# ── Bottom: KDE density curve ─────────────────────────────────
ax2 = fig.add_subplot(gs[1])

reg_cv_vals = cv_df[cv_df['Mode'] == 'Regular']['CV'].values
irr_cv_vals = cv_df[cv_df['Mode'] == 'Irregular']['CV'].values
irr_cv_clipped = irr_cv_vals[irr_cv_vals < 3.0]

cv_range = np.linspace(0, 2.5, 400)
kde_reg  = gaussian_kde(reg_cv_vals)(cv_range)
kde_irr  = gaussian_kde(irr_cv_clipped)(cv_range)

ax2.plot(cv_range, kde_reg, color=COL_REG, lw=2.6, label=f'Regular (n={len(reg_cv_vals):,})')
ax2.fill_between(cv_range, kde_reg, color=COL_REG, alpha=0.25)
ax2.plot(cv_range, kde_irr, color=COL_IRR, lw=2.6, label=f'Irregular (n={len(irr_cv_clipped):,})')
ax2.fill_between(cv_range, kde_irr, color=COL_IRR, alpha=0.25)
ax2.axvline(CV_THRESH, color=COL_THRESH, lw=2.5, ls='--',
            label=f'Threshold = {CV_THRESH}')

ax2.set_xlabel("CV value", fontsize=14)
ax2.set_ylabel("Density", fontsize=14)
ax2.set_title("CV Density — Regular vs Irregular (KDE)", fontsize=16, fontweight='bold', pad=12)
ax2.tick_params(labelsize=12)
ax2.legend(fontsize=12.5, framealpha=0.95)
ax2.grid(True, alpha=0.35, color=COL_GRID)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

plt.savefig("/kaggle/working/plot3_cv_distribution.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: plot3_cv_distribution.png")

# ────────────────────────────────────────────────────────────
#  PLOT 5 — Multi-Sample Overlay (Regular vs Irregular)
#  Overlay 5 samples for one label each
# ────────────────────────────────────────────────────────────
print("  Plot 5: Multi-sample overlay...")

# Highest Regular count: Label A (Tonic Spiking)
# Highest Irregular count: Label C (Tonic Bursting)
focus_pairs = [
    ('A', COL_REG, 'Regular'),
    ('C', COL_IRR, 'Irregular'),
]

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("5-Sample Overlay — Regular vs Irregular",
             fontsize=14, fontweight='bold')

for ax, (lbl, color, mtype) in zip(axes, focus_pairs):
    sub = df[df['Label'] == lbl].head(5)
    ax.axvspan(50, 250, color=COL_INP, alpha=0.25, label='Input window')
    for i, (_, row) in enumerate(sub.iterrows()):
        v = row[V_COLS].values.astype(float)
        ax.plot(T, v, color=color, lw=1.8, alpha=0.55 + i * 0.09,
                label=f'Sample {i}' if i == 0 else None)
    ax.axhline(30, color='#444444', lw=1.0, ls='--', alpha=0.7)
    name = LABEL_NAMES.get(lbl, lbl)
    ax.set_title(f"Label {lbl}: {name}\n[{mtype}] — 5 samples overlay", fontweight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("v (mV)")
    ax.set_xlim(0, 300)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.legend(fontsize=8.5, framealpha=0.95)

plt.tight_layout()
plt.savefig("/kaggle/working/plot5_multisample_overlay.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: plot5_multisample_overlay.png")

# ────────────────────────────────────────────────────────────
#  PLOT 6 — ISI Histogram per Label
# ────────────────────────────────────────────────────────────
print("  Plot 6: ISI histogram per label...")

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3))
fig.suptitle("ISI (Inter-Spike Interval) Histogram — All Labels",
             fontsize=14, fontweight='bold')

axes_flat = axes.flatten()

for idx, lbl in enumerate(labels_sorted):
    ax      = axes_flat[idx]
    sub     = df[df['Label'] == lbl]
    all_isi = []

    for _, row in sub.iterrows():
        s       = row[S_COLS].values.astype(int)
        spk_idx = np.where(s == 1)[0]
        if len(spk_idx) > 1:
            all_isi.extend(np.diff(spk_idx) * H)   # ms units

    mode_label = sub.iloc[0]['Mode']
    color      = COL_REG if mode_label == 'Regular' else COL_IRR

    if all_isi:
        ax.hist(all_isi, bins=30, color=color, alpha=0.9,
                edgecolor='white', lw=0.5)
        ax.set_title(f"{lbl}: {LABEL_NAMES.get(lbl,'')}\n[{mode_label}]",
                     fontsize=8, color=color, fontweight='bold')
    else:
        ax.text(0.5, 0.5, 'No spikes', ha='center', va='center',
                transform=ax.transAxes, fontsize=9, color='gray')
        ax.set_title(f"{lbl}: {LABEL_NAMES.get(lbl,'')}", fontsize=8)

    ax.set_xlabel("ISI (ms)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.30, color=COL_GRID)

for idx in range(n_labels, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig("/kaggle/working/plot6_isi_histogram.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: plot6_isi_histogram.png")

# ────────────────────────────────────────────────────────────
#  PLOT 7 — Class Distribution Pie Charts per Label
# ────────────────────────────────────────────────────────────
print("  Plot 7: Regular/Irregular distribution per label...")

fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 3))
fig.suptitle("Regular vs Irregular Distribution — Each Label (500 samples)",
             fontsize=14, fontweight='bold')

axes_flat = axes.flatten()

for idx, lbl in enumerate(labels_sorted):
    ax     = axes_flat[idx]
    sub    = df[df['Label'] == lbl]
    counts = sub['Mode'].value_counts()

    reg_n = counts.get('Regular', 0)
    irr_n = counts.get('Irregular', 0)
    total = reg_n + irr_n

    if total > 0:
        sizes      = [reg_n, irr_n]
        labels_pie = [f'Regular\n{reg_n}', f'Irregular\n{irr_n}']
        colors_pie = [COL_REG, COL_IRR]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels_pie, colors=colors_pie,
            autopct='%1.0f%%', startangle=90,
            textprops={'fontsize': 7},
            wedgeprops={'edgecolor': 'white', 'lw': 1}
        )
        for at in autotexts:
            at.set_fontsize(7)
            at.set_fontweight('bold')

    name = LABEL_NAMES.get(lbl, lbl)
    ax.set_title(f"{lbl}: {name}", fontsize=8, fontweight='bold')

for idx in range(n_labels, len(axes_flat)):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.savefig("/kaggle/working/plot7_mode_distribution_pie.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: plot7_mode_distribution_pie.png")

# ────────────────────────────────────────────────────────────
#  SUMMARY
# ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  PLOTTING COMPLETE")
print("=" * 60)
print(f"  Total rows plotted : {len(df):,}")
reg_total = len(df[df['Mode'] == 'Regular'])
irr_total = len(df[df['Mode'] == 'Irregular'])
print(f"  Regular samples    : {reg_total:,} ({100*reg_total/len(df):.1f}%)")
print(f"  Irregular samples  : {irr_total:,} ({100*irr_total/len(df):.1f}%)")
print()
print("  Saved files:")
for i in range(1, 8):
    print(f"    /kaggle/working/plot{i}_*.png")
print("=" * 60)