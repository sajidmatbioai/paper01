"""
============================================================
 K-FOLD GRAPHS — FULLY FIXED CODE
 Binary + Multi-Class — DB-HLSTM Framework

 Input:  /kaggle/working/kfold_results.csv
 Output: /kaggle/working/kfold_graph1_binary_bar.png
         /kaggle/working/kfold_graph2_multiclass_bar.png
         /kaggle/working/kfold_graph3_combined.png
         /kaggle/working/kfold_graph4_stability.png
============================================================
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os, sys

# ── LOAD CSV ─────────────────────────────────────────────────
CSV_PATH = "/kaggle/working/kfold_results.csv"
if not os.path.exists(CSV_PATH):
    print(f"ERROR: {CSV_PATH} not found!")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
df.columns = [c.strip().lower() for c in df.columns]
print("Columns:", list(df.columns))
print(df)

# ── EXTRACT DATA ─────────────────────────────────────────────
cols = list(df.columns)

if 'task' in cols:
    df['task'] = df['task'].str.strip().str.lower()
    acc_col  = next((c for c in cols if 'acc' in c), None)
    fold_col = next((c for c in cols if 'fold' in c), 'fold')

    bin_df = df[df['task'].isin(['binary','bin'])].reset_index(drop=True)
    mc_df  = df[df['task'].isin(['multiclass','multi','mc','multi-class'])].reset_index(drop=True)

    if bin_df.empty or mc_df.empty:
        tasks = df['task'].unique()
        bin_df = df[df['task'] == tasks[0]].reset_index(drop=True)
        mc_df  = df[df['task'] == tasks[1]].reset_index(drop=True)

    bin_accs  = pd.to_numeric(bin_df[acc_col], errors='coerce').dropna().values
    mc_accs   = pd.to_numeric(mc_df[acc_col],  errors='coerce').dropna().values
    bin_folds = np.arange(1, len(bin_accs)+1)
    mc_folds  = np.arange(1, len(mc_accs)+1)

elif any('binary' in c or 'bin' in c for c in cols) and \
     any('multi' in c or 'mc' in c for c in cols):
    bin_col = next(c for c in cols if 'binary' in c or 'bin' in c)
    mc_col  = next(c for c in cols if 'multi' in c or 'mc' in c)
    df_c    = df[pd.to_numeric(df[bin_col], errors='coerce').notna()].reset_index(drop=True)
    bin_accs  = pd.to_numeric(df_c[bin_col], errors='coerce').dropna().values
    mc_accs   = pd.to_numeric(df_c[mc_col],  errors='coerce').dropna().values
    bin_folds = np.arange(1, len(bin_accs)+1)
    mc_folds  = np.arange(1, len(mc_accs)+1)

else:
    acc_col  = next((c for c in cols if 'acc' in c), cols[-1])
    all_accs = pd.to_numeric(df[acc_col], errors='coerce').dropna().values
    half     = len(all_accs) // 2
    bin_accs  = all_accs[:half]
    mc_accs   = all_accs[half:]
    bin_folds = np.arange(1, len(bin_accs)+1)
    mc_folds  = np.arange(1, len(mc_accs)+1)

# Normalize 0-1 → percentage
if bin_accs.max() <= 1.0: bin_accs = bin_accs * 100
if mc_accs.max()  <= 1.0: mc_accs  = mc_accs  * 100

n_folds  = len(bin_accs)
bin_mean = np.mean(bin_accs); bin_std = np.std(bin_accs)
bin_min  = np.min(bin_accs);  bin_max = np.max(bin_accs)
mc_mean  = np.mean(mc_accs);  mc_std  = np.std(mc_accs)
mc_min   = np.min(mc_accs);   mc_max  = np.max(mc_accs)

print(f"\nBINARY     — Mean: {bin_mean:.2f}% ± {bin_std:.2f}%  Min: {bin_min:.2f}%  Max: {bin_max:.2f}%")
print(f"MULTICLASS — Mean: {mc_mean:.2f}% ± {mc_std:.2f}%  Min: {mc_min:.2f}%  Max: {mc_max:.2f}%")

# ── COLORS ───────────────────────────────────────────────────
BLUE       = '#1565C0'
LIGHT_BLUE = '#90CAF9'
RED        = '#C62828'
LIGHT_RED  = '#EF9A9A'
GREEN      = '#2E7D32'
ORANGE     = '#F57F17'

def smart_ylim(mn, mx):
    """Always show bars properly regardless of accuracy range"""
    margin = (mx - mn) * 0.3 if (mx - mn) > 0 else 2
    ylo = max(0, mn - margin - 1)
    yhi = min(105, mx + margin + 1)
    return ylo, yhi

# ══════════════════════════════════════════════════════════════
# GRAPH 1 — Binary Bar Chart
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFBFC')

x = np.arange(n_folds)
colors = [BLUE] * n_folds
colors[int(np.argmax(bin_accs))] = GREEN
colors[int(np.argmin(bin_accs))] = '#E53935'

bars = ax.bar(x, bin_accs, width=0.5, color=colors,
              edgecolor='white', linewidth=1.5, zorder=3)

for bar, acc in zip(bars, bin_accs):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.1,
            f'{acc:.2f}%', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#222222')

ax.axhline(bin_mean, color=ORANGE, lw=2.5, linestyle='--',
           zorder=4, label=f'Mean = {bin_mean:.2f}%')
ax.axhspan(bin_mean-bin_std, bin_mean+bin_std,
           alpha=0.12, color=ORANGE, zorder=2,
           label=f'±Std = ±{bin_std:.2f}%')

summary = (f'Mean ± Std : {bin_mean:.2f}% ± {bin_std:.2f}%\n'
           f'Min / Max  : {bin_min:.2f}% / {bin_max:.2f}%\n'
           f'Range      : {bin_max-bin_min:.2f}%')
ax.text(0.98, 0.04, summary, transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', fc='#E3F2FD',
                  ec=BLUE, alpha=0.95))

patches = [
    mpatches.Patch(color=BLUE,      label='Fold Accuracy'),
    mpatches.Patch(color=GREEN,     label=f'Best: {bin_max:.2f}%'),
    mpatches.Patch(color='#E53935', label=f'Min:  {bin_min:.2f}%'),
    plt.Line2D([0],[0], color=ORANGE, lw=2.5, linestyle='--',
               label=f'Mean: {bin_mean:.2f}%'),
]
ax.legend(handles=patches, fontsize=10, loc='upper left',
          framealpha=0.95, edgecolor='#CCCCCC')

ax.set_xticks(x)
ax.set_xticklabels([f'Fold {int(f)}' for f in bin_folds],
                   fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('K-Fold Split', fontsize=13, fontweight='bold')
ylo, yhi = smart_ylim(bin_min, bin_max)
ax.set_ylim(ylo, yhi)
ax.set_xlim(-0.5, n_folds-0.5)
ax.grid(axis='y', alpha=0.4, linestyle='--', zorder=0)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.set_title(
    f'Binary Classification — {n_folds}-Fold Cross-Validation\n'
    f'DB-HLSTM | Regular vs Irregular NVC',
    fontsize=14, fontweight='bold', color='#1F4E79', pad=15)

plt.tight_layout()
plt.savefig('/kaggle/working/kfold_graph1_binary_bar.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Graph 1 saved ✅")

# ══════════════════════════════════════════════════════════════
# GRAPH 2 — Multi-Class Bar Chart
# ══════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor('white')
ax.set_facecolor('#FAFBFC')

x = np.arange(n_folds)
mc_colors = [RED] * n_folds
mc_colors[int(np.argmax(mc_accs))] = GREEN
mc_colors[int(np.argmin(mc_accs))] = '#B71C1C'

bars2 = ax.bar(x, mc_accs, width=0.5, color=mc_colors,
               edgecolor='white', linewidth=1.5, zorder=3)

for bar, acc in zip(bars2, mc_accs):
    ax.text(bar.get_x()+bar.get_width()/2,
            bar.get_height()+0.2,
            f'{acc:.2f}%', ha='center', va='bottom',
            fontsize=12, fontweight='bold', color='#222222')

ax.axhline(mc_mean, color=ORANGE, lw=2.5, linestyle='--',
           zorder=4, label=f'Mean = {mc_mean:.2f}%')
ax.axhspan(mc_mean-mc_std, mc_mean+mc_std,
           alpha=0.12, color=ORANGE, zorder=2,
           label=f'±Std = ±{mc_std:.2f}%')

summary2 = (f'Mean ± Std : {mc_mean:.2f}% ± {mc_std:.2f}%\n'
            f'Min / Max  : {mc_min:.2f}% / {mc_max:.2f}%\n'
            f'Range      : {mc_max-mc_min:.2f}%')
ax.text(0.98, 0.04, summary2, transform=ax.transAxes,
        ha='right', va='bottom', fontsize=10,
        bbox=dict(boxstyle='round,pad=0.5', fc='#FFEBEE',
                  ec=RED, alpha=0.95))

patches2 = [
    mpatches.Patch(color=RED,       label='Fold Accuracy'),
    mpatches.Patch(color=GREEN,     label=f'Best: {mc_max:.2f}%'),
    mpatches.Patch(color='#B71C1C', label=f'Min:  {mc_min:.2f}%'),
    plt.Line2D([0],[0], color=ORANGE, lw=2.5, linestyle='--',
               label=f'Mean: {mc_mean:.2f}%'),
]
ax.legend(handles=patches2, fontsize=10, loc='upper left',
          framealpha=0.95, edgecolor='#CCCCCC')

ax.set_xticks(x)
ax.set_xticklabels([f'Fold {int(f)}' for f in mc_folds],
                   fontsize=12, fontweight='bold')
ax.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')
ax.set_xlabel('K-Fold Split', fontsize=13, fontweight='bold')
ylo2, yhi2 = smart_ylim(mc_min, mc_max)
ax.set_ylim(ylo2, yhi2)
ax.set_xlim(-0.5, n_folds-0.5)
ax.grid(axis='y', alpha=0.4, linestyle='--', zorder=0)
for sp in ['top','right']: ax.spines[sp].set_visible(False)
ax.set_title(
    f'Multi-Class Classification — {n_folds}-Fold Cross-Validation\n'
    f'DB-HLSTM | 20 Izhikevich Neuron Types',
    fontsize=14, fontweight='bold', color='#1F4E79', pad=15)

plt.tight_layout()
plt.savefig('/kaggle/working/kfold_graph2_multiclass_bar.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Graph 2 saved ✅")

# ══════════════════════════════════════════════════════════════
# GRAPH 3 — Combined Side by Side
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.patch.set_facecolor('white')

for ax, accs, folds, mean_v, std_v, mn, mx, color, title in [
    (axes[0], bin_accs, bin_folds, bin_mean, bin_std,
     bin_min, bin_max, BLUE,
     f'Binary | Mean: {bin_mean:.2f}% ± {bin_std:.2f}%'),
    (axes[1], mc_accs, mc_folds, mc_mean, mc_std,
     mc_min, mc_max, RED,
     f'Multi-Class | Mean: {mc_mean:.2f}% ± {mc_std:.2f}%'),
]:
    ax.set_facecolor('#FAFBFC')
    x = np.arange(len(accs))
    c_list = [color]*len(accs)
    c_list[int(np.argmax(accs))] = GREEN
    c_list[int(np.argmin(accs))] = '#B71C1C'

    bars_ = ax.bar(x, accs, width=0.5, color=c_list,
                   edgecolor='white', linewidth=1.5, zorder=3)
    for bar, acc in zip(bars_, accs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.1,
                f'{acc:.2f}%', ha='center', va='bottom',
                fontsize=10, fontweight='bold')

    ax.axhline(mean_v, color=ORANGE, lw=2.5, linestyle='--',
               zorder=4, label=f'Mean={mean_v:.2f}%')
    ax.axhspan(mean_v-std_v, mean_v+std_v,
               alpha=0.12, color=ORANGE, zorder=2,
               label=f'±{std_v:.2f}%')

    ax.set_xticks(x)
    ax.set_xticklabels([f'F{int(f)}' for f in folds],
                       fontsize=11, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ylo_, yhi_ = smart_ylim(mn, mx)
    ax.set_ylim(ylo_, yhi_)
    ax.grid(axis='y', alpha=0.4, linestyle='--', zorder=0)
    ax.legend(fontsize=10, loc='lower right', framealpha=0.95)
    ax.set_title(title, fontsize=13, fontweight='bold', color='#1F4E79')
    for sp in ['top','right']: ax.spines[sp].set_visible(False)

fig.suptitle(
    f'DB-HLSTM — {n_folds}-Fold Cross-Validation Results\n'
    f'Binary (Left) vs Multi-Class (Right)',
    fontsize=14, fontweight='bold', color='#1F4E79')
plt.tight_layout(rect=[0, 0, 1, 0.93])
plt.savefig('/kaggle/working/kfold_graph3_combined.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Graph 3 saved ✅")

# ══════════════════════════════════════════════════════════════
# GRAPH 4 — Stability Analysis (Line + Box) — 2x2 grid
# ══════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.patch.set_facecolor('white')

for row_i, (accs, mean_v, std_v, mn, mx,
            color, light_c, label) in enumerate([
    (bin_accs, bin_mean, bin_std, bin_min, bin_max,
     BLUE, LIGHT_BLUE, 'Binary'),
    (mc_accs, mc_mean, mc_std, mc_min, mc_max,
     RED, LIGHT_RED, 'Multi-Class'),
]):
    n = len(accs)
    folds_x = list(range(1, n+1))
    ylo_, yhi_ = smart_ylim(mn, mx)

    # ── Line plot ─────────────────────────────────────────────
    ax_l = axes[row_i][0]
    ax_l.set_facecolor('#FAFBFC')
    ax_l.plot(folds_x, accs, 'o-', color=color, lw=2.5,
              markersize=10, markerfacecolor='white',
              markeredgewidth=2.5, zorder=4)
    ax_l.axhline(mean_v, color=ORANGE, lw=2, linestyle='--',
                 label=f'Mean: {mean_v:.2f}%')
    ax_l.fill_between(folds_x,
                      [mean_v-std_v]*n, [mean_v+std_v]*n,
                      alpha=0.15, color=ORANGE)
    for f, a in zip(folds_x, accs):
        ax_l.annotate(f'{a:.2f}%', xy=(f, a),
                      xytext=(0, 10), textcoords='offset points',
                      ha='center', fontsize=9,
                      fontweight='bold', color=color)
    ax_l.set_xticks(folds_x)
    ax_l.set_xticklabels([f'Fold {f}' for f in folds_x], fontsize=10)
    ax_l.set_ylim(ylo_, yhi_)
    ax_l.grid(alpha=0.4, linestyle='--')
    ax_l.legend(fontsize=10)
    ax_l.set_title(f'{label} — Fold Accuracy Trend',
                   fontsize=12, fontweight='bold', color='#1F4E79')
    for sp in ['top','right']: ax_l.spines[sp].set_visible(False)

    # ── Box plot ──────────────────────────────────────────────
    ax_b = axes[row_i][1]
    ax_b.set_facecolor('#FAFBFC')
    ax_b.boxplot(accs, positions=[1], widths=0.4,
                 patch_artist=True, zorder=3,
                 boxprops=dict(facecolor=light_c,
                               color=color, linewidth=2),
                 medianprops=dict(color=GREEN, linewidth=2.5),
                 whiskerprops=dict(color=color, linewidth=1.5),
                 capprops=dict(color=color, linewidth=2))
    np.random.seed(42)
    jitter = np.random.uniform(-0.08, 0.08, n)
    ax_b.scatter([1+j for j in jitter], accs,
                 color=color, s=80, zorder=5,
                 edgecolors='white', linewidth=1.5)
    ax_b.scatter([1], [mean_v], color=ORANGE, s=150,
                 zorder=6, marker='D', edgecolors='white',
                 linewidth=1.5, label=f'Mean: {mean_v:.2f}%')

    for val, lbl in [(mean_v, f'{mean_v:.2f}%'),
                     (mean_v+std_v, f'+1σ: {mean_v+std_v:.2f}%'),
                     (mean_v-std_v, f'-1σ: {mean_v-std_v:.2f}%')]:
        if ylo_ < val < yhi_:
            ax_b.text(1.28, val, lbl, va='center',
                      fontsize=9, color=ORANGE if lbl==f'{mean_v:.2f}%' else '#555555',
                      fontweight='bold' if lbl==f'{mean_v:.2f}%' else 'normal')

    ax_b.set_xlim(0.5, 2.0)
    ax_b.set_ylim(ylo_, yhi_)
    ax_b.set_xticks([1])
    ax_b.set_xticklabels([f'{label}\nClassification'],
                         fontsize=11, fontweight='bold')
    ax_b.set_ylabel('Accuracy (%)', fontsize=11, fontweight='bold')
    ax_b.legend(fontsize=10, loc='lower right')
    ax_b.grid(axis='y', alpha=0.4, linestyle='--')
    ax_b.set_title(f'{label} — Distribution',
                   fontsize=12, fontweight='bold', color='#1F4E79')
    for sp in ['top','right']: ax_b.spines[sp].set_visible(False)

fig.suptitle(
    f'DB-HLSTM — {n_folds}-Fold Cross-Validation Stability Analysis\n'
    f'Binary: {bin_mean:.2f}% ± {bin_std:.2f}%   |   '
    f'Multi-Class: {mc_mean:.2f}% ± {mc_std:.2f}%',
    fontsize=13, fontweight='bold', color='#1F4E79')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('/kaggle/working/kfold_graph4_stability.png',
            dpi=200, bbox_inches='tight', facecolor='white')
plt.close()
print("Graph 4 saved ✅")

print(f"\n{'='*60}")
print(f"  ALL GRAPHS SAVED!")
print(f"  Binary   : {bin_mean:.2f}% ± {bin_std:.2f}%")
print(f"  Multi    : {mc_mean:.2f}% ± {mc_std:.2f}%")
print(f"{'='*60}")