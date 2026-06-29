"""
============================================================
 Step 1.2 — HRF Extraction Results Plotter  [KAGGLE VERSION]

 Loads Neural_HRF_Data.csv and generates the following plots:

   Plot 1 — 3 HRF kernel shapes (Canonical, TD, DD)
   Plot 2 — Spike train → 3 HRF outputs: Regular vs Irregular
   Plot 3 — Canonical HRF grid for all 20 labels
   Plot 4 — hrf_c / hrf_td / hrf_dd mean ± std band (per class)
   Plot 5 — 3-kernel overlay: one Regular + one Irregular sample
   Plot 6 — HRF amplitude distribution boxplot per label
   Plot 7 — Canonical HRF peak time per label (bar chart)

 INPUT : /kaggle/working/Neural_HRF_Data.csv
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.special import gamma as Gamma
from scipy.signal import convolve
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
CSV_PATH = "/kaggle/working/Neural_HRF_Data.csv"
SEQ_LEN  = 600
h_step   = 0.5                            # ms per step
T_MS     = np.arange(SEQ_LEN) * h_step   # 0..299.5 ms
T_S      = T_MS / 1000.0                  # seconds

# ── COLORS (deeper, more saturated — print/journal safe) ─────
COL_REG  = "#1f5fb8"   # Regular      — strong royal blue
COL_IRR  = "#b3202a"   # Irregular    — deep crimson red
COL_C    = "#1d8a4e"   # Canonical    — forest green
COL_TD   = "#7b3fa0"   # Temporal D   — deep purple
COL_DD   = "#c2630f"   # Dispersion D — burnt orange
COL_S    = "#555555"   # Spike train  — dark gray
COL_GRID = "#999999"   # neutral grid

# ── LABEL NAMES ─────────────────────────────────────────────
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

# ── HRF PARAMETERS — consistent with extraction code ────────
a1, a2 = 6.0, 16.0
b1, b2 = 1.0,  1.0
C_hrf  = 1.0 / 6.0

def h_canonical(t):
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    Gu = (t**(a2-1) * b2**a2 * np.exp(-b2*t)) / Gamma(a2)
    return Gp - C_hrf * Gu

def h_temporal(t):
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    Gu = (t**(a2-1) * b2**a2 * np.exp(-b2*t)) / Gamma(a2)
    return Gp*((a1-1)/t - b1) - C_hrf*Gu*((a2-1)/t - b2)

def h_dispersion(t):
    t  = np.maximum(t, 1e-10)
    Gp = (t**(a1-1) * b1**a1 * np.exp(-b1*t)) / Gamma(a1)
    return Gp * (a1/b1 - t)

# Compute and normalize kernels
kernel_c  = h_canonical(T_S).copy()
kernel_td = h_temporal(T_S).copy()
kernel_dd = h_dispersion(T_S).copy()
kernel_c  /= (np.max(np.abs(kernel_c))  + 1e-10)
kernel_td /= (np.max(np.abs(kernel_td)) + 1e-10)
kernel_dd /= (np.max(np.abs(kernel_dd)) + 1e-10)

# ── COLUMN GROUPS ────────────────────────────────────────────
S_COLS = [f"s_{i}"      for i in range(SEQ_LEN)]
HRF_C  = [f"hrf_c_{i}"  for i in range(SEQ_LEN)]
HRF_TD = [f"hrf_td_{i}" for i in range(SEQ_LEN)]
HRF_DD = [f"hrf_dd_{i}" for i in range(SEQ_LEN)]

# ── LOAD ─────────────────────────────────────────────────────
print("=" * 60)
print("  HRF Extraction Plotter  [KAGGLE]")
print("=" * 60)
print(f"\n  Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"  Shape   : {df.shape}")
print(f"  Labels  : {sorted(df['Label'].unique())}")
print(f"  Classes : {df['Class'].value_counts().to_dict()}")

labels_sorted = sorted(df['Label'].unique())
ncols = 4
nrows = (len(labels_sorted) + ncols - 1) // ncols

# ────────────────────────────────────────────────────────────
#  PLOT 1 — 3 HRF Kernel Shapes
#  Canonical, Temporal Derivative, Dispersion Derivative
# ────────────────────────────────────────────────────────────
print("\n  Plot 1: HRF kernel shapes...")

t_plot = T_S   # 0 to 0.3 s

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("HRF Kernels — Friston (1998) SPM Model\n"
             "Convert spike train s(t) into a BOLD-like signal",
             fontsize=13, fontweight='bold')

kernel_data = [
    (kernel_c,  "Canonical HRF  h_c(t)",
     "Gp(t) − C·Gu(t)\nPositive peak + undershoot",    COL_C),
    (kernel_td, "Temporal Derivative  h_td(t)",
     "dh/dt\nCaptures BOLD timing shifts",              COL_TD),
    (kernel_dd, "Dispersion Derivative  h_dd(t)",
     "dh/db1\nCaptures BOLD width variations",          COL_DD),
]

for ax, (kernel, title, desc, color) in zip(axes, kernel_data):
    ax.plot(t_plot * 1000, kernel, color=color, lw=2.4)
    ax.axhline(0, color='#444444', lw=1.0, ls='--', alpha=0.7)
    ax.fill_between(t_plot * 1000, 0, kernel,
                    where=kernel > 0, color=color, alpha=0.30)
    ax.fill_between(t_plot * 1000, 0, kernel,
                    where=kernel < 0, color=COL_IRR, alpha=0.20)
    peak_ms = T_MS[np.argmax(kernel)]
    ax.axvline(peak_ms, color=color, lw=1.4, ls=':', alpha=0.85,
               label=f'Peak @ {peak_ms:.1f} ms')
    ax.set_title(title, fontsize=10.5, fontweight='bold', color=color)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.text(0.97, 0.03, desc, transform=ax.transAxes,
            ha='right', va='bottom', fontsize=8,
            color='#555555', style='italic')
    ax.legend(fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 300)

plt.tight_layout()
plt.savefig("/kaggle/working/hrf_plot1_kernels.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: hrf_plot1_kernels.png")

# ────────────────────────────────────────────────────────────
#  PLOT 2 — Spike Train → 3 HRF Outputs
#  Regular and Irregular: s(t) + hrf_c + hrf_td + hrf_dd
# ────────────────────────────────────────────────────────────
print("  Plot 2: Spike train → HRF outputs (Regular vs Irregular)...")

reg_row = df[df['Class'] == 0].iloc[0]
irr_row = df[df['Class'] == 1].iloc[0]

fig, axes = plt.subplots(4, 2, figsize=(16, 14), sharex=True)
fig.suptitle("HRF Convolution Pipeline — Regular vs Irregular\n"
             "s(t) → Canonical → Temporal D → Dispersion D",
             fontsize=13, fontweight='bold')

row_labels = ["Spike Train  s(t)",
              "Canonical HRF  hrf_c(t)",
              "Temporal Derivative  hrf_td(t)",
              "Dispersion Derivative  hrf_dd(t)"]
col_data   = [(S_COLS, COL_S), (HRF_C, COL_C),
              (HRF_TD, COL_TD), (HRF_DD, COL_DD)]

for col_idx, (row_data, cls_name, main_color) in enumerate([
    (reg_row, "Regular (Class 0)",   COL_REG),
    (irr_row, "Irregular (Class 1)", COL_IRR),
]):
    lbl  = row_data['Label']
    name = LABEL_NAMES.get(lbl, lbl)

    for row_idx, ((cols, sig_color), row_label) in enumerate(
            zip(col_data, row_labels)):
        ax  = axes[row_idx][col_idx]
        sig = row_data[cols].values.astype(float)

        if row_idx == 0:   # spike train — vertical lines
            spk_t = T_MS[sig == 1]
            ax.vlines(spk_t, 0, 1, color=main_color, lw=1.8,
                      label=f"Spikes n={len(spk_t)}")
            ax.set_ylim(-0.1, 1.3)
            ax.set_title(f"{cls_name} — Label {lbl}: {name}",
                         fontsize=9.5, fontweight='bold', color=main_color)
        else:
            ax.plot(T_MS, sig, color=sig_color, lw=1.8)
            ax.fill_between(T_MS, 0, sig, color=sig_color, alpha=0.28)
            ax.axhline(0, color='#444444', lw=0.8, ls='--', alpha=0.6)

        ax.set_ylabel(row_label, fontsize=8.5, fontweight='bold')
        ax.grid(True, alpha=0.35, color=COL_GRID)
        ax.legend(fontsize=8, loc='upper right', framealpha=0.95) if row_idx == 0 else None

axes[3][0].set_xlabel("Time (ms)")
axes[3][1].set_xlabel("Time (ms)")

plt.tight_layout()
plt.savefig("/kaggle/working/hrf_plot2_pipeline.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: hrf_plot2_pipeline.png")

# ────────────────────────────────────────────────────────────
#  PLOT 3 — Canonical HRF Grid for All 20 Labels
# ────────────────────────────────────────────────────────────
print("  Plot 3: All 20 labels canonical HRF grid...")

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.2))
fig.suptitle("Canonical HRF Output hrf_c(t) — All 20 Labels",
             fontsize=14, fontweight='bold')
axes_flat = axes.flatten()

for idx, lbl in enumerate(labels_sorted):
    ax   = axes_flat[idx]
    row  = df[df['Label'] == lbl].iloc[0]
    sig  = row[HRF_C].values.astype(float)
    s    = row[S_COLS].values.astype(float)
    cls  = int(row['Class'])
    color = COL_REG if cls == 0 else COL_IRR
    name  = LABEL_NAMES.get(lbl, lbl)
    mode  = "Regular" if cls == 0 else "Irregular"

    ax.plot(T_MS, sig, color=color, lw=1.6)
    ax.fill_between(T_MS, 0, sig, color=color, alpha=0.28)
    ax.axhline(0, color='#444444', lw=0.7, ls='--', alpha=0.6)

    # Spike raster drawn AFTER the signal so axis limits are already set
    spk_t = T_MS[s == 1]
    y0, y1 = ax.get_ylim()
    raster_y = y0 - 0.06 * (y1 - y0)
    ax.vlines(spk_t, raster_y, y0, color=COL_S, lw=0.7, alpha=0.6, clip_on=False)
    ax.set_ylim(raster_y, y1)

    ax.set_title(f"{lbl}: {name}\n[{mode}]",
                 fontsize=8.5, color=color, fontweight='bold')
    ax.set_xlabel("ms", fontsize=7)
    ax.set_ylabel("hrf_c", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.30, color=COL_GRID)
    ax.set_xlim(0, 300)

for idx in range(len(labels_sorted), len(axes_flat)):
    axes_flat[idx].set_visible(False)

legend_elements = [
    Patch(facecolor=COL_REG, label='Regular (Class 0)'),
    Patch(facecolor=COL_IRR, label='Irregular (Class 1)'),
]
fig.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("/kaggle/working/hrf_plot3_canonical_grid.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: hrf_plot3_canonical_grid.png")

# ────────────────────────────────────────────────────────────
#  PLOT 4 — Mean ± Std Band: Regular vs Irregular
#  Average time-course for all three kernels
# ────────────────────────────────────────────────────────────
print("  Plot 4: Mean ± std band per class (3 kernels)...")

fig, axes = plt.subplots(3, 2, figsize=(16, 12), sharex=True)
fig.suptitle("HRF Output Mean ± Std Band\n"
             "Regular vs Irregular — Average Across All Samples",
             fontsize=13, fontweight='bold')

kernel_groups = [
    (HRF_C,  "Canonical HRF  hrf_c(t)",         COL_C),
    (HRF_TD, "Temporal Derivative  hrf_td(t)",   COL_TD),
    (HRF_DD, "Dispersion Derivative  hrf_dd(t)", COL_DD),
]

for row_idx, (cols, kernel_name, k_color) in enumerate(kernel_groups):
    for col_idx, (cls_id, cls_name, cls_color) in enumerate([
        (0, "Regular (Class 0)",   COL_REG),
        (1, "Irregular (Class 1)", COL_IRR),
    ]):
        ax       = axes[row_idx][col_idx]
        sub      = df[df['Class'] == cls_id][cols].values.astype(float)
        mean_sig = sub.mean(axis=0)
        std_sig  = sub.std(axis=0)

        ax.plot(T_MS, mean_sig, color=k_color, lw=2.4, label='Mean')
        ax.fill_between(T_MS,
                        mean_sig - std_sig,
                        mean_sig + std_sig,
                        color=k_color, alpha=0.30, label='±1 std')
        ax.axhline(0, color='#444444', lw=0.9, ls='--', alpha=0.6)
        ax.set_ylabel(kernel_name, fontsize=8.5, fontweight='bold')
        if row_idx == 0:
            ax.set_title(f"{cls_name}\nn={len(sub):,} samples",
                         fontsize=10.5, color=cls_color, fontweight='bold')
        ax.legend(fontsize=8.5, framealpha=0.95)
        ax.grid(True, alpha=0.35, color=COL_GRID)
        ax.set_xlim(0, 300)

axes[2][0].set_xlabel("Time (ms)")
axes[2][1].set_xlabel("Time (ms)")

plt.tight_layout()
plt.savefig("/kaggle/working/hrf_plot4_mean_std_band.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: hrf_plot4_mean_std_band.png")

# ────────────────────────────────────────────────────────────
#  PLOT 5 — 3-Kernel Overlay: Regular vs Irregular
#  All three HRF outputs displayed together for one sample each
# ────────────────────────────────────────────────────────────
print("  Plot 5: 3-kernel overlay...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("3 HRF Kernels Overlay — Regular vs Irregular Sample",
             fontsize=13, fontweight='bold')

for ax, (row_data, cls_name, cls_color) in zip(axes, [
    (reg_row, "Regular (Class 0)",   COL_REG),
    (irr_row, "Irregular (Class 1)", COL_IRR),
]):
    lbl  = row_data['Label']
    name = LABEL_NAMES.get(lbl, lbl)

    ax.plot(T_MS, row_data[HRF_C].values.astype(float),
            color=COL_C,  lw=2.2,         label='Canonical hrf_c')
    ax.plot(T_MS, row_data[HRF_TD].values.astype(float),
            color=COL_TD, lw=1.8, ls='--', label='Temporal D hrf_td')
    ax.plot(T_MS, row_data[HRF_DD].values.astype(float),
            color=COL_DD, lw=1.8, ls=':',  label='Dispersion D hrf_dd')
    ax.axhline(0, color='#444444', lw=0.9, ls='--', alpha=0.6)

    # Spike raster at the bottom of the plot
    s   = row_data[S_COLS].values.astype(float)
    spk = T_MS[s == 1]
    ax.vlines(spk, -0.12, -0.05, color=cls_color, lw=1.6,
              alpha=0.85, label='Spikes s(t)')

    ax.set_title(f"{cls_name} — Label {lbl}: {name}", fontsize=10.5,
                 color=cls_color, fontweight='bold')
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Normalized amplitude")
    ax.legend(fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 300)

plt.tight_layout()
plt.savefig("/kaggle/working/hrf_plot5_3kernel_overlay.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: hrf_plot5_3kernel_overlay.png")

# ────────────────────────────────────────────────────────────
#  PLOT 6 — HRF Amplitude Distribution per Label (Lollipop)
# ────────────────────────────────────────────────────────────
print("  Plot 6: HRF amplitude distribution per label...")

# Lollipop chart: 1 row, 3 columns (one per kernel).
# Stem = colored line, head = colored circle, error bar = ±std.
# Regular = blue, Irregular = red.
fig, axes = plt.subplots(1, 3, figsize=(26, 6))
fig.suptitle("HRF Max Amplitude Distribution per Label",
             fontsize=16, fontweight='bold', y=1.02)

legend_elements = [
    Patch(facecolor=COL_REG, edgecolor='#000000', label='Regular'),
    Patch(facecolor=COL_IRR, edgecolor='#000000', label='Irregular'),
]
fig.legend(handles=legend_elements, loc='upper right', ncol=1,
           fontsize=12, framealpha=0.95, bbox_to_anchor=(1.0, 1.02))

for ax, (cols, kernel_name, k_color) in zip(axes, [
    (HRF_C,  "Canonical  hrf_c",      COL_C),
    (HRF_TD, "Temporal D  hrf_td",    COL_TD),
    (HRF_DD, "Dispersion D  hrf_dd",  COL_DD),
]):
    means, stds, colors = [], [], []
    for lbl in labels_sorted:
        sub  = df[df['Label'] == lbl][cols].values.astype(float)
        amps = sub.max(axis=1)
        means.append(amps.mean())
        stds.append(amps.std())
        cls_med = int(df[df['Label'] == lbl].iloc[0]['Class'])
        colors.append(COL_REG if cls_med == 0 else COL_IRR)

    means = np.array(means)
    stds  = np.array(stds)
    x     = np.arange(len(labels_sorted))

    # stems
    ax.vlines(x, 0, means, color=colors, lw=2.2, alpha=0.85)
    # error bars
    ax.errorbar(x, means, yerr=stds, fmt='none', ecolor='#333333',
                elinewidth=1.6, capsize=5, capthick=1.6, zorder=2)
    # circle heads
    ax.scatter(x, means, c=colors, s=180, edgecolor='#000000',
               linewidth=1.4, zorder=3)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_sorted, fontsize=10, fontweight='bold')
    ax.set_title(kernel_name, fontsize=13, fontweight='bold',
                 color=k_color, pad=10)
    ax.set_xlabel("Label", fontsize=11)
    ax.set_ylabel("Max amplitude (mean ± std)", fontsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.35, axis='y', color=COL_GRID)
    ax.set_xlim(-0.6, len(labels_sorted) - 0.4)
    ax.set_ylim(0, means.max() + stds.max() * 3 + 0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

plt.tight_layout(rect=[0, 0, 0.94, 0.95])
plt.savefig("/kaggle/working/hrf_plot6_amplitude_boxplot.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: hrf_plot6_amplitude_boxplot.png")
plt.show()
print("  Saved: hrf_plot6_amplitude_boxplot.png")

# ────────────────────────────────────────────────────────────
#  PLOT 7 — Canonical HRF Peak Time per Label (Bar Chart)
# ────────────────────────────────────────────────────────────
print("  Plot 7: HRF peak time per label...")

peak_times = []
for lbl in labels_sorted:
    sub     = df[df['Label'] == lbl]
    hrf_mat = sub[HRF_C].values.astype(float)
    # Compute mean peak time across all samples for this label
    pt_mean = np.mean([T_MS[np.argmax(row)] for row in hrf_mat])
    pt_std  = np.std( [T_MS[np.argmax(row)] for row in hrf_mat])
    cls     = int(sub.iloc[0]['Class'])
    peak_times.append({'Label': lbl, 'PeakTime': pt_mean,
                       'Std': pt_std, 'Class': cls})

pt_df = pd.DataFrame(peak_times)

fig, ax = plt.subplots(figsize=(15, 5.5))
colors = [COL_REG if c == 0 else COL_IRR for c in pt_df['Class']]
bars = ax.bar(pt_df['Label'], pt_df['PeakTime'],
              yerr=pt_df['Std'], color=colors,
              alpha=0.9, capsize=5, edgecolor='#222222', lw=0.9,
              error_kw=dict(lw=1.4, capthick=1.4))

for bar, row in zip(bars, pt_df.itertuples()):
    name = LABEL_NAMES.get(row.Label, '')
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + pt_df['Std'].max() * 0.05,
            name, ha='center', va='bottom',
            fontsize=7.5, rotation=45, fontweight='bold')

ax.set_xlabel("Label", fontsize=12)
ax.set_ylabel("Mean Peak Time of hrf_c (ms)", fontsize=12)
ax.set_title("Canonical HRF Peak Time per Label\n"
             "(time at which the HRF response reaches its maximum)",
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.35, axis='y', color=COL_GRID)
legend_elements = [
    Patch(facecolor=COL_REG, label='Regular (Class 0)'),
    Patch(facecolor=COL_IRR, label='Irregular (Class 1)'),
]
ax.legend(handles=legend_elements, fontsize=10.5, framealpha=0.95)

plt.tight_layout()
plt.savefig("/kaggle/working/hrf_plot7_peak_time.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: hrf_plot7_peak_time.png")

# ────────────────────────────────────────────────────────────
#  SUMMARY
# ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  HRF PLOTTING COMPLETE")
print("=" * 60)
print(f"  Total rows : {len(df):,}")
print(f"  Regular    : {len(df[df['Class']==0]):,}")
print(f"  Irregular  : {len(df[df['Class']==1]):,}")
print()
print("  Saved files:")
for i in range(1, 8):
    print(f"    /kaggle/working/hrf_plot{i}_*.png")
print("=" * 60)