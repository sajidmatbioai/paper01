"""
============================================================
 Step 1.3 — Balloon-Windkessel BOLD Results Plotter  [KAGGLE]

 Loads Neural_BOLD_Data.csv and generates the following plots:

   Plot 1 — BOLD: Regular vs Irregular full comparison
   Plot 2 — 4 state variable pipeline panel (s→CBF→CBV→dHb→BOLD)
   Plot 3 — BOLD signal grid for all 20 labels
   Plot 4 — BOLD peak amplitude per label (bar + scatter)
   Plot 5 — Multi-sample BOLD overlay (Regular vs Irregular)
   Plot 6 — State variable distributions boxplot (CBF/CBV/dHb/BOLD)
   Plot 7 — BOLD mean ± std band per class

 INPUT : /kaggle/working/Neural_BOLD_Data.csv
============================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
CSV_PATH = "/kaggle/working/Neural_BOLD_Data.csv"
SEQ_LEN  = 600
dt       = 0.05                        # seconds per step
T        = np.arange(SEQ_LEN) * dt    # 0..29.95 s

# ── COLORS (deeper, more saturated — print/journal safe) ─────
COL_REG  = "#1f5fb8"   # Regular      — strong royal blue
COL_IRR  = "#b3202a"   # Irregular    — deep crimson red
COL_CBF  = "#1d8a4e"   # CBF          — forest green
COL_CBV  = "#7b3fa0"   # CBV          — deep purple
COL_DHB  = "#c2630f"   # dHb          — burnt orange
COL_BOLD = "#b3202a"   # BOLD         — deep crimson red
COL_S    = "#555555"   # Neural input — dark gray
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

# ── COLUMN GROUPS ────────────────────────────────────────────
# Neural input (s columns — convolved signal from HRF file)
S_COLS    = [f"s_{i}"    for i in range(SEQ_LEN)]
CBF_COLS  = [f"cbf_{i}"  for i in range(SEQ_LEN)]
CBV_COLS  = [f"cbv_{i}"  for i in range(SEQ_LEN)]
DHB_COLS  = [f"dhb_{i}"  for i in range(SEQ_LEN)]
BOLD_COLS = [f"bold_{i}" for i in range(SEQ_LEN)]

# ── LOAD ─────────────────────────────────────────────────────
print("=" * 60)
print("  Balloon-Windkessel BOLD Plotter  [KAGGLE]")
print("=" * 60)
print(f"\n  Loading: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"  Shape   : {df.shape}")
print(f"  Labels  : {sorted(df['Label'].unique())}")
print(f"  Classes : {df['Class'].value_counts().to_dict()}")

labels_sorted = sorted(df['Label'].unique())
ncols = 4
nrows = (len(labels_sorted) + ncols - 1) // ncols

reg_row = df[df['Class'] == 0].iloc[0]
irr_row = df[df['Class'] == 1].iloc[0]

# ────────────────────────────────────────────────────────────
#  PLOT 1 — BOLD Regular vs Irregular Full Comparison
#  Neural input + state variables + BOLD — side by side
# ────────────────────────────────────────────────────────────
print("\n  Plot 1: BOLD Regular vs Irregular comparison...")

fig, axes = plt.subplots(2, 3, figsize=(18, 9))
fig.suptitle("Balloon-Windkessel — Regular vs Irregular BOLD Signal\n"
             "Neural s(t) → Hemodynamics → BOLD",
             fontsize=14, fontweight='bold')

for row_idx, (row_data, cls_name, cls_color) in enumerate([
    (reg_row, "Regular (Class 0)",   COL_REG),
    (irr_row, "Irregular (Class 1)", COL_IRR),
]):
    lbl  = row_data['Label']
    name = LABEL_NAMES.get(lbl, lbl)
    s_in = row_data[S_COLS].values.astype(float)
    cbf  = row_data[CBF_COLS].values.astype(float)
    cbv  = row_data[CBV_COLS].values.astype(float)
    dhb  = row_data[DHB_COLS].values.astype(float)
    bold = row_data[BOLD_COLS].values.astype(float)

    # Col 0: Neural input s(t)
    ax = axes[row_idx][0]
    ax.plot(T, s_in, color=COL_S, lw=1.6)
    ax.set_title(f"{cls_name}\nLabel {lbl}: {name} — Neural Input s(t)",
                 fontsize=9.5, fontweight='bold', color=cls_color)
    ax.set_ylabel("s(t)")
    ax.set_xlabel("Time (s)")
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 30)

    # Col 1: State variables CBF, CBV, dHb
    ax = axes[row_idx][1]
    ax.plot(T, cbf, color=COL_CBF, lw=2.0, label='CBF f(t)')
    ax.plot(T, cbv, color=COL_CBV, lw=2.0, label='CBV v(t)')
    ax.plot(T, dhb, color=COL_DHB, lw=2.0, label='dHb q(t)')
    ax.axhline(1.0, color='#444444', lw=1.0, ls='--', alpha=0.6,
               label='Baseline (1.0)')
    ax.set_title("Hemodynamic State Variables",
                 fontsize=9.5, fontweight='bold')
    ax.set_ylabel("Normalized")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 30)

    # Col 2: BOLD signal
    ax = axes[row_idx][2]
    ax.plot(T, bold, color=cls_color, lw=2.4,
            label=f"BOLD  peak={bold.max():.5f}")
    ax.fill_between(T, 0, bold, color=cls_color, alpha=0.30)
    ax.axhline(0, color='#444444', lw=1.0, ls='--', alpha=0.6)
    ax.set_title("BOLD Signal  BOLD(t)",
                 fontsize=9.5, fontweight='bold')
    ax.set_ylabel("BOLD (ΔS/S)")
    ax.set_xlabel("Time (s)")
    ax.legend(fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 30)

plt.tight_layout()
plt.savefig("/kaggle/working/bold_plot1_regular_vs_irregular.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: bold_plot1_regular_vs_irregular.png")

# ────────────────────────────────────────────────────────────
#  PLOT 2 — 5-Panel Pipeline: s → CBF → CBV → dHb → BOLD
#  Both Regular and Irregular displayed in the same plot
# ────────────────────────────────────────────────────────────
print("  Plot 2: Full 5-panel state variable pipeline...")

fig, axes = plt.subplots(5, 1, figsize=(14, 15), sharex=True)
fig.suptitle("Balloon-Windkessel Full Pipeline\n"
             "Neural s(t)  →  CBF  →  CBV  →  dHb  →  BOLD",
             fontsize=13, fontweight='bold')

panel_info = [
    (S_COLS,    "Neural Input  s(t)",           "s(t)",       COL_S),
    (CBF_COLS,  "Cerebral Blood Flow  CBF f(t)", "f (norm.)",  COL_CBF),
    (CBV_COLS,  "Cerebral Blood Vol  CBV v(t)",  "v (norm.)",  COL_CBV),
    (DHB_COLS,  "Deoxyhemoglobin  dHb q(t)",    "q (norm.)",  COL_DHB),
    (BOLD_COLS, "BOLD Signal  BOLD(t)",          "BOLD(ΔS/S)", COL_BOLD),
]

for ax, (cols, title, ylabel, sig_color) in zip(axes, panel_info):
    for row_data, cls_name, cls_color in [
        (reg_row, "Regular",   COL_REG),
        (irr_row, "Irregular", COL_IRR),
    ]:
        sig = row_data[cols].values.astype(float)
        lbl = row_data['Label']
        ax.plot(T, sig, color=cls_color, lw=2.0, alpha=0.9,
                label=f"{cls_name} (Label {lbl})")

    ax.set_title(title, fontsize=10.5, fontweight='bold', color=sig_color)
    ax.set_ylabel(ylabel, fontsize=9.5)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 30)

axes[0].legend(fontsize=9.5, loc='upper right', framealpha=0.95)
axes[4].set_xlabel("Time (s)", fontsize=11)

plt.tight_layout()
plt.savefig("/kaggle/working/bold_plot2_pipeline_panel.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: bold_plot2_pipeline_panel.png")

# ────────────────────────────────────────────────────────────
#  PLOT 3 — BOLD Grid for All 20 Labels
# ────────────────────────────────────────────────────────────
print("  Plot 3: All 20 labels BOLD grid...")

fig, axes = plt.subplots(nrows, ncols, figsize=(20, nrows * 3.2))
fig.suptitle("BOLD Signal — All 20 Labels (first sample per label)",
             fontsize=14, fontweight='bold')
axes_flat = axes.flatten()

for idx, lbl in enumerate(labels_sorted):
    ax   = axes_flat[idx]
    row  = df[df['Label'] == lbl].iloc[0]
    bold = row[BOLD_COLS].values.astype(float)
    cls  = int(row['Class'])
    color = COL_REG if cls == 0 else COL_IRR
    name  = LABEL_NAMES.get(lbl, lbl)
    mode  = "Regular" if cls == 0 else "Irregular"

    ax.plot(T, bold, color=color, lw=1.6)
    ax.fill_between(T, 0, bold, color=color, alpha=0.28)
    ax.axhline(0, color='#444444', lw=0.7, ls='--', alpha=0.6)
    ax.set_title(f"{lbl}: {name}\n[{mode}]  peak={bold.max():.4f}",
                 fontsize=8.5, color=color, fontweight='bold')
    ax.set_xlabel("Time (s)", fontsize=7)
    ax.set_ylabel("BOLD", fontsize=7)
    ax.tick_params(labelsize=7)
    ax.grid(True, alpha=0.30, color=COL_GRID)
    ax.set_xlim(0, 30)

for idx in range(len(labels_sorted), len(axes_flat)):
    axes_flat[idx].set_visible(False)

legend_elements = [
    Patch(facecolor=COL_REG, label='Regular (Class 0)'),
    Patch(facecolor=COL_IRR, label='Irregular (Class 1)'),
]
fig.legend(handles=legend_elements, loc='lower right', fontsize=9)

plt.tight_layout()
plt.savefig("/kaggle/working/bold_plot3_all_labels_grid.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: bold_plot3_all_labels_grid.png")

# ────────────────────────────────────────────────────────────
#  PLOT 4 — BOLD Peak Amplitude per Label
# ────────────────────────────────────────────────────────────
print("  Plot 4: BOLD peak amplitude per label...")

peak_stats = []
for lbl in labels_sorted:
    sub      = df[df['Label'] == lbl]
    bold_mat = sub[BOLD_COLS].values.astype(float)
    peaks    = bold_mat.max(axis=1)
    cls      = int(sub.iloc[0]['Class'])
    peak_stats.append({'Label': lbl, 'Mean': peaks.mean(),
                       'Std': peaks.std(), 'Class': cls,
                       'Name': LABEL_NAMES.get(lbl, lbl)})

pt_df = pd.DataFrame(peak_stats)

fig, axes = plt.subplots(1, 2, figsize=(18, 6))
fig.suptitle("BOLD Peak Amplitude per Label", fontsize=13, fontweight='bold')

# Bar chart
ax = axes[0]
colors = [COL_REG if c == 0 else COL_IRR for c in pt_df['Class']]
bars = ax.bar(pt_df['Label'], pt_df['Mean'],
              yerr=pt_df['Std'], color=colors,
              alpha=0.9, capsize=5, edgecolor='#222222', lw=0.9,
              error_kw=dict(lw=1.4, capthick=1.4))
ax.set_xlabel("Label", fontsize=12)
ax.set_ylabel("Mean BOLD Peak (ΔS/S)", fontsize=12)
ax.set_title("Mean Peak BOLD per Label\n(±std across 500 samples)",
             fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.35, axis='y', color=COL_GRID)
for bar, row in zip(bars, pt_df.itertuples()):
    ax.text(bar.get_x() + bar.get_width()/2,
            bar.get_height() + pt_df['Std'].max() * 0.05,
            row.Name, ha='center', va='bottom',
            fontsize=7.5, rotation=45, fontweight='bold')
ax.legend(handles=[
    Patch(facecolor=COL_REG, label='Regular'),
    Patch(facecolor=COL_IRR, label='Irregular'),
], fontsize=10.5, framealpha=0.95)

# Scatter — Regular vs Irregular comparison
ax2 = axes[1]
reg_pts = pt_df[pt_df['Class'] == 0]
irr_pts = pt_df[pt_df['Class'] == 1]
ax2.scatter(reg_pts['Label'], reg_pts['Mean'], color=COL_REG,
            s=140, zorder=5, edgecolor='#222222', lw=1.1, label='Regular')
ax2.scatter(irr_pts['Label'], irr_pts['Mean'], color=COL_IRR,
            s=140, marker='D', zorder=5, edgecolor='#222222', lw=1.1, label='Irregular')
ax2.axhline(reg_pts['Mean'].mean(), color=COL_REG, lw=2.0, ls='--',
            alpha=0.8, label=f"Reg avg={reg_pts['Mean'].mean():.5f}")
ax2.axhline(irr_pts['Mean'].mean(), color=COL_IRR, lw=2.0, ls='--',
            alpha=0.8, label=f"Irr avg={irr_pts['Mean'].mean():.5f}")
ax2.set_xlabel("Label", fontsize=12)
ax2.set_ylabel("Mean BOLD Peak", fontsize=12)
ax2.set_title("Regular vs Irregular BOLD Peak Comparison", fontsize=13, fontweight='bold')
ax2.legend(fontsize=9, framealpha=0.95)
ax2.grid(True, alpha=0.35, color=COL_GRID)

plt.tight_layout()
plt.savefig("/kaggle/working/bold_plot4_peak_amplitude.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: bold_plot4_peak_amplitude.png")

# ────────────────────────────────────────────────────────────
#  PLOT 5 — Multi-Sample BOLD Overlay
#  Overlay 5 samples per class to visualise signal variability
# ────────────────────────────────────────────────────────────
print("  Plot 5: Multi-sample BOLD overlay...")

# Select labels with the highest mean BOLD peak per class
reg_lbl = pt_df[pt_df['Class'] == 0].sort_values('Mean', ascending=False).iloc[0]['Label']
irr_lbl = pt_df[pt_df['Class'] == 1].sort_values('Mean', ascending=False).iloc[0]['Label']

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("BOLD Signal — 5 Sample Overlay per Class",
             fontsize=13, fontweight='bold')

for ax, (lbl, cls_color, cls_name) in zip(axes, [
    (reg_lbl, COL_REG, "Regular"),
    (irr_lbl, COL_IRR, "Irregular"),
]):
    sub      = df[df['Label'] == lbl].head(5)
    bold_mat = sub[BOLD_COLS].values.astype(float)

    for i, bold in enumerate(bold_mat):
        ax.plot(T, bold, color=cls_color, lw=1.6,
                alpha=0.45 + i * 0.11,
                label=f'Sample {i}' if i == 0 else None)

    ax.fill_between(T, bold_mat.min(axis=0), bold_mat.max(axis=0),
                    color=cls_color, alpha=0.18, label='Sample range')
    ax.plot(T, bold_mat.mean(axis=0), color=cls_color,
            lw=2.6, ls='--', label='Mean of 5')
    ax.axhline(0, color='#444444', lw=1.0, ls='--', alpha=0.6)

    name = LABEL_NAMES.get(lbl, lbl)
    ax.set_title(f"Label {lbl}: {name}\n[{cls_name}] — 5 samples overlay",
                 color=cls_color, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BOLD (ΔS/S)")
    ax.legend(fontsize=8.5, framealpha=0.95)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 30)

plt.tight_layout()
plt.savefig("/kaggle/working/bold_plot5_multisample_overlay.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: bold_plot5_multisample_overlay.png")

# ────────────────────────────────────────────────────────────
#  PLOT 6 — State Variable Distributions (Boxplot)
#  CBF, CBV, dHb mean and BOLD peak — Regular vs Irregular
# ────────────────────────────────────────────────────────────
print("  Plot 6: State variable distributions...")

reg_df = df[df['Class'] == 0]
irr_df = df[df['Class'] == 1]

fig, axes = plt.subplots(1, 4, figsize=(18, 6))
fig.suptitle("Hemodynamic State Variable Distributions\n"
             "Regular vs Irregular — Mean per Sample",
             fontsize=13, fontweight='bold')

var_info = [
    (CBF_COLS,  "CBF mean  f",  COL_CBF,  False),
    (CBV_COLS,  "CBV mean  v",  COL_CBV,  False),
    (DHB_COLS,  "dHb mean  q",  COL_DHB,  False),
    (BOLD_COLS, "BOLD peak",    COL_BOLD, True),
]

for ax, (cols, ylabel, var_color, use_peak) in zip(axes, var_info):
    if use_peak:
        reg_vals = reg_df[cols].values.astype(float).max(axis=1)
        irr_vals = irr_df[cols].values.astype(float).max(axis=1)
    else:
        reg_vals = reg_df[cols].values.astype(float).mean(axis=1)
        irr_vals = irr_df[cols].values.astype(float).mean(axis=1)

    bp = ax.boxplot([reg_vals, irr_vals],
                    labels=['Regular', 'Irregular'],
                    patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', lw=2.2),
                    whiskerprops=dict(lw=1.5, color='#333333'),
                    capprops=dict(lw=1.5, color='#333333'),
                    boxprops=dict(lw=1.3, edgecolor='#222222'),
                    flierprops=dict(marker='o', markersize=4, alpha=0.5,
                                     markeredgecolor='none'))
    bp['boxes'][0].set_facecolor(COL_REG)
    bp['boxes'][0].set_alpha(0.85)
    bp['boxes'][1].set_facecolor(COL_IRR)
    bp['boxes'][1].set_alpha(0.85)
    bp['fliers'][0].set_markerfacecolor(COL_REG)
    bp['fliers'][0].set_markeredgecolor(COL_REG)
    bp['fliers'][1].set_markerfacecolor(COL_IRR)
    bp['fliers'][1].set_markeredgecolor(COL_IRR)

    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, color=var_color, fontweight='bold', fontsize=12.5)
    ax.tick_params(axis='x', labelsize=11)
    ax.tick_params(axis='y', labelsize=10)
    ax.grid(True, alpha=0.35, axis='y', color=COL_GRID)

plt.tight_layout()
plt.savefig("/kaggle/working/bold_plot6_state_distributions.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: bold_plot6_state_distributions.png")

# ────────────────────────────────────────────────────────────
#  PLOT 7 — BOLD Mean ± Std Band per Class
#  Average BOLD time-course across the entire dataset
# ────────────────────────────────────────────────────────────
print("  Plot 7: BOLD mean ± std band per class...")

fig, axes = plt.subplots(1, 2, figsize=(16, 5))
fig.suptitle("BOLD Signal — Mean ± Std Band\n"
             "Across All Samples per Class",
             fontsize=13, fontweight='bold')

for ax, (cls_id, cls_name, cls_color) in zip(axes, [
    (0, "Regular (Class 0)",   COL_REG),
    (1, "Irregular (Class 1)", COL_IRR),
]):
    sub       = df[df['Class'] == cls_id][BOLD_COLS].values.astype(float)
    bold_mean = sub.mean(axis=0)
    bold_std  = sub.std(axis=0)

    ax.plot(T, bold_mean, color=cls_color, lw=2.6, label='Mean BOLD')
    ax.fill_between(T,
                    bold_mean - bold_std,
                    bold_mean + bold_std,
                    color=cls_color, alpha=0.32, label='±1 std')
    ax.fill_between(T,
                    bold_mean - 2 * bold_std,
                    bold_mean + 2 * bold_std,
                    color=cls_color, alpha=0.15, label='±2 std')
    ax.axhline(0, color='#444444', lw=1.0, ls='--', alpha=0.6)

    ax.set_title(f"{cls_name}\nn={len(sub):,} samples  "
                 f"peak={bold_mean.max():.5f}",
                 color=cls_color, fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("BOLD (ΔS/S)")
    ax.legend(fontsize=9.5, framealpha=0.95)
    ax.grid(True, alpha=0.35, color=COL_GRID)
    ax.set_xlim(0, 30)

plt.tight_layout()
plt.savefig("/kaggle/working/bold_plot7_mean_std_band.png",
            dpi=300, bbox_inches='tight')
plt.show()
print("  Saved: bold_plot7_mean_std_band.png")

# ────────────────────────────────────────────────────────────
#  SUMMARY
# ────────────────────────────────────────────────────────────
print()
print("=" * 60)
print("  BOLD PLOTTING COMPLETE")
print("=" * 60)
print(f"  Total rows : {len(df):,}")
print(f"  Regular    : {len(df[df['Class']==0]):,}")
print(f"  Irregular  : {len(df[df['Class']==1]):,}")
reg_peak = df[df['Class']==0][BOLD_COLS].values.astype(float).max(axis=1).mean()
irr_peak = df[df['Class']==1][BOLD_COLS].values.astype(float).max(axis=1).mean()
print(f"\n  Regular mean BOLD peak  : {reg_peak:.6f}")
print(f"  Irregular mean BOLD peak: {irr_peak:.6f}")
print(f"  Regular > Irregular     : {'✅' if reg_peak > irr_peak else '⚠️'}")
print()
print("  Saved files:")
for i in range(1, 8):
    print(f"    /kaggle/working/bold_plot{i}_*.png")
print("=" * 60)