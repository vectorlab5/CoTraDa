#!/usr/bin/env python3
"""
Generate synthetic experimental charts for CoTDA planning draft.
IMPORTANT: These are for PLANNING ONLY. Replace with real data before submission.

Generates:
  - convergence.pdf       (Fig. 2: training reward curves)
  - sensitivity.pdf       (Fig. 3: 3-panel sensitivity analysis)
  - ablation_bar.pdf      (Fig. 4: ablation bar chart)
  - tradeoff.pdf          (Fig. 5: utility vs computation scatter)
  - trust_evolution.pdf   (Fig. 6: trust score trajectories)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# ============================================================
# NATURE JOURNAL COLOR THEME
# ============================================================
NATURE_COLORS = {
    'blue':       '#0C5DA5',   # Our method
    'orange':     '#FF9500',   # Main baseline
    'green':      '#00B945',   # Secondary baseline
    'red':        '#FF2C00',   # Ablation / negative
    'purple':     '#845B97',   # Additional
    'gray':       '#474747',   # Text
    'light_gray': '#9E9E9E',   # Secondary elements
    'teal':       '#17BECF',   # New baseline 1
    'brown':      '#8C564B',   # New baseline 2
    'pink':       '#E377C2',   # New baseline 3
}
PALETTE = ['#0C5DA5', '#FF9500', '#00B945', '#FF2C00', '#845B97', '#9E9E9E',
           '#17BECF', '#8C564B', '#E377C2']

def set_style():
    plt.rcParams.update({
        'font.size': 10,           # Decreased from 12 as axis text was "too big"
        'font.family': 'serif',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'axes.edgecolor': NATURE_COLORS['gray'],
        'axes.labelcolor': NATURE_COLORS['gray'],
        'axes.spines.top': False,
        'axes.spines.right': False,
        'legend.frameon': False,
        'lines.linewidth': 2.0,    # Increased from 1.5
    })

set_style()
OUT_DIR = os.path.join(os.path.dirname(__file__), '..')


# ============================================================
# FIG 2: CONVERGENCE CURVES
# ============================================================
def plot_convergence():
    np.random.seed(42)
    episodes = np.arange(1, 2001)

    def make_curve(final_reward, speed, noise=0.8):
        base = -2.0 + (final_reward + 2.0) * (1 - np.exp(-episodes / speed))
        return base + np.random.normal(0, noise * np.exp(-episodes / 500), len(episodes))

    ours   = make_curve(8.5, 180, 0.9)
    ddpg   = make_curve(5.8, 280, 1.1)
    fedveh = make_curve(5.2, 320, 1.0)

    # Smooth for plotting
    def smooth(y, w=30):
        return np.convolve(y, np.ones(w)/w, mode='valid')

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ep_s = episodes[:len(smooth(ours))]
    ax.plot(ep_s, smooth(ours),   color=PALETTE[0], label='CoTDA')
    ax.plot(ep_s, smooth(ddpg),   color=PALETTE[1], label='DDPG-Offload', linestyle='--')
    ax.plot(ep_s, smooth(fedveh), color=PALETTE[4], label='FedVeh', linestyle='-.')
    # Lyapunov horizontal
    ax.axhline(y=6.4, color=PALETTE[2], linestyle=':', linewidth=1.2, label='Lyapunov-VEC')

    ax.set_xlabel('Episode', fontsize=11)
    ax.set_ylabel('Episodic Reward', fontsize=11)
    ax.set_xlim(0, 2000)
    ax.legend(fontsize=9, loc='lower right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'convergence.pdf'))
    plt.close()
    print("Generated: convergence.pdf")


# ============================================================
# FIG 3: SENSITIVITY (3-panel) - 8 baselines
# ============================================================
def plot_sensitivity():
    np.random.seed(123)

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.0))
    methods = ['CoTDA', 'AoI-LGFS', 'Lyapunov-VEC', 'DDPG-Offload',
               'FedVeh', 'Greedy-AoI', 'Joint-Heuristic', 'AO-Joint', 'LP-Relaxed']
    colors  = [PALETTE[0], PALETTE[1], PALETTE[2], PALETTE[3],
               PALETTE[4], PALETTE[5], PALETTE[6], PALETTE[8], PALETTE[7]]
    markers = ['o', 's', '^', 'D', 'v', 'x', 'P', '*', 'h']

    # (a) W-AoI vs Vehicle count
    V_vals = [10, 20, 30, 40, 50, 60, 70]
    base_aoi = {
        'CoTDA': [3.41, 3.92, 4.51, 5.12, 5.68, 6.21, 6.82],
        'AoI-LGFS':              [4.18, 4.89, 5.72, 6.43, 7.18, 7.91, 8.59],
        'Lyapunov-VEC':          [4.52, 5.31, 6.15, 6.89, 7.74, 8.42, 9.13],
        'DDPG-Offload':          [5.21, 5.92, 6.78, 7.51, 8.23, 8.97, 9.64],
        'FedVeh':                [4.89, 5.62, 6.45, 7.21, 7.98, 8.71, 9.38],
        'Greedy-AoI':            [6.83, 7.81, 8.92, 9.87, 10.74, 11.62, 12.41],
        'Joint-Heuristic':       [3.78, 4.35, 5.01, 5.72, 6.34, 6.91, 7.52],
        'AO-Joint':              [3.52, 4.05, 4.68, 5.32, 5.91, 6.48, 7.08],
        'LP-Relaxed':            [3.58, 4.12, 4.74, 5.38, 5.97, 6.52, 7.11],
    }
    ax = axes[0]
    for i, m in enumerate(methods):
        jitter = np.random.normal(0, 0.08, len(V_vals))
        ax.plot(V_vals, np.array(base_aoi[m]) + jitter, color=colors[i],
                marker=markers[i], markersize=5, linewidth=1.5)
    ax.set_xlabel('Number of Vehicles $V$', fontsize=10)
    ax.set_ylabel('Weighted AoI (slots)', fontsize=10)
    ax.set_title('(a)', fontsize=11, loc='left')

    # (b) Utility vs Bandwidth
    bw_vals = [5, 10, 15, 20, 25, 30, 35, 40]
    base_util = {
        'CoTDA': [0.48, 0.56, 0.60, 0.63, 0.65, 0.66, 0.66, 0.67],
        'AoI-LGFS':              [0.32, 0.40, 0.46, 0.50, 0.53, 0.55, 0.57, 0.59],
        'Lyapunov-VEC':          [0.29, 0.37, 0.43, 0.47, 0.51, 0.54, 0.56, 0.59],
        'DDPG-Offload':          [0.26, 0.34, 0.39, 0.43, 0.47, 0.50, 0.53, 0.55],
        'FedVeh':                [0.28, 0.36, 0.41, 0.45, 0.49, 0.52, 0.54, 0.57],
        'Greedy-AoI':            [0.15, 0.22, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42],
        'Joint-Heuristic':       [0.39, 0.47, 0.52, 0.55, 0.57, 0.59, 0.60, 0.61],
        'AO-Joint':              [0.45, 0.53, 0.58, 0.61, 0.63, 0.64, 0.65, 0.66],
        'LP-Relaxed':            [0.42, 0.50, 0.55, 0.58, 0.60, 0.62, 0.63, 0.64],
    }
    ax = axes[1]
    for i, m in enumerate(methods):
        jitter = np.random.normal(0, 0.005, len(bw_vals))
        ax.plot(bw_vals, np.array(base_util[m]) + jitter, color=colors[i],
                marker=markers[i], markersize=5, linewidth=1.5)
    ax.set_xlabel('Per-RSU Bandwidth (MHz)', fontsize=10)
    ax.set_ylabel('Data Utility', fontsize=10)
    ax.set_title('(b)', fontsize=11, loc='left')

    # (c) Delivery Ratio vs Link Failure Prob
    fail_vals = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    base_del = {
        'CoTDA': [93.1, 91.2, 88.7, 86.1, 83.2, 81.4, 79.6, 77.1, 74.8],
        'AoI-LGFS':              [88.1, 85.3, 82.1, 78.4, 74.7, 71.2, 67.8, 64.1, 60.3],
        'Lyapunov-VEC':          [85.4, 82.1, 78.5, 74.2, 70.3, 66.8, 63.2, 59.4, 55.7],
        'DDPG-Offload':          [85.4, 82.0, 77.8, 73.4, 69.1, 65.2, 61.7, 59.8, 58.2],
        'FedVeh':                [86.2, 83.1, 79.8, 75.9, 71.8, 68.1, 64.5, 61.2, 57.8],
        'Greedy-AoI':            [78.2, 74.1, 69.5, 64.8, 60.1, 55.7, 51.2, 46.8, 42.5],
        'Joint-Heuristic':       [91.2, 88.8, 85.9, 82.8, 79.5, 77.1, 74.8, 72.1, 69.5],
        'AO-Joint':              [92.5, 90.1, 87.5, 84.8, 81.8, 79.3, 76.9, 74.2, 71.8],
        'LP-Relaxed':            [92.0, 89.6, 87.1, 84.2, 81.1, 78.8, 76.4, 73.8, 71.2],
    }
    ax = axes[2]
    for i, m in enumerate(methods):
        jitter = np.random.normal(0, 0.4, len(fail_vals))
        ax.plot(fail_vals, np.array(base_del[m]) + jitter, color=colors[i],
                marker=markers[i], markersize=5, linewidth=1.5)
    ax.set_xlabel('Link Failure Probability', fontsize=10)
    ax.set_ylabel('Delivery Ratio (%)', fontsize=10)
    ax.set_title('(c)', fontsize=11, loc='left')

    # Shared legend at top
    handles = []
    labels_list = []
    for i, m in enumerate(methods):
        h, = axes[0].plot([], [], color=colors[i], marker=markers[i], markersize=3.5,
                          linewidth=1.2, label=m)
        handles.append(h)
        labels_list.append(m)
    fig.legend(handles, labels_list, loc='upper center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.28))

    plt.tight_layout()
    plt.subplots_adjust(top=0.74)
    plt.savefig(os.path.join(OUT_DIR, 'sensitivity.pdf'))
    plt.close()
    print("Generated: sensitivity.pdf")


# ============================================================
# FIG 4: ABLATION BAR CHART
# ============================================================
def plot_ablation_bar():
    variants = ['Full Model', 'w/o Acquisition', 'w/o Compression',
                'w/o Rel. Routing', 'w/o Feedback']
    utility  = [0.627, 0.518, 0.561, 0.543, 0.578]
    drops    = [0.0, -0.109, -0.066, -0.084, -0.049]
    colors   = [NATURE_COLORS['blue']] + [NATURE_COLORS['red']] * 4

    fig, ax = plt.subplots(figsize=(5.0, 3.2))
    bars = ax.barh(variants[::-1], utility[::-1],
                   color=colors[::-1], edgecolor='white', linewidth=0.5)

    for bar, drop in zip(bars, drops[::-1]):
        if drop != 0:
            ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                    f'{drop:+.3f}', va='center', fontsize=7, color=NATURE_COLORS['red'])

    ax.set_xlabel('Data Utility')
    ax.set_xlim(0.45, 0.70)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'ablation_bar.pdf'))
    plt.close()
    print("Generated: ablation_bar.pdf")


# ============================================================
# FIG 5: TRADE-OFF SCATTER - includes all 8 baselines
# ============================================================
def plot_tradeoff():
    methods_data = {
        'Greedy-AoI':     (1.2,  0.298),
        'AoI-LGFS':       (3.4,  0.512),
        'Lyapunov-VEC':   (8.6,  0.468),
        'FedVeh':         (47.2, 0.487),
        'DDPG-Offload':   (91.3, 0.421),
        'Joint-Heuristic':(14.1, 0.534),
        'AO-Joint':       (61.3, 0.598),
        'LP-Relaxed':     (52.8, 0.568),
        'Oracle-3':       (287.6, 0.689),
    }
    ours = (38.4, 0.627)

    fig, ax = plt.subplots(figsize=(5.5, 3.8))

    # Color-code: external baselines gray, our new baselines teal/brown/pink
    new_baseline_colors = {
        'Joint-Heuristic': NATURE_COLORS['teal'],
        'AO-Joint': NATURE_COLORS['pink'],
        'LP-Relaxed': NATURE_COLORS['brown'],
        'Oracle-3': '#BCBd22',
    }

    for name, (time_ms, util) in methods_data.items():
        c = new_baseline_colors.get(name, NATURE_COLORS['light_gray'])
        ax.scatter(time_ms, util, c=c, s=80, zorder=2)
        offset_x, offset_y = 5, 0.01
        if name == 'DDPG-Offload':
            offset_x = -75
        elif name == 'Oracle-3':
            offset_x = -65
            offset_y = -0.02
        elif name == 'LP-Relaxed':
            offset_x = 5
            offset_y = -0.03
        elif name == 'AO-Joint':
            offset_x = 5
            offset_y = 0.015
        ax.annotate(name, (time_ms, util), fontsize=8.5,
                    xytext=(offset_x, offset_y), textcoords='offset points')

    ax.scatter(*ours, c=NATURE_COLORS['blue'], s=250, marker='*', zorder=3)
    ax.annotate('Ours', ours, fontsize=11, fontweight='bold',
                xytext=(6, 10), textcoords='offset points', color=NATURE_COLORS['blue'])

    # Slot budget line
    ax.axvline(x=100, color=NATURE_COLORS['red'], linestyle=':', linewidth=1.5, alpha=0.6)
    ax.text(105, 0.30, '100 ms\nslot budget', fontsize=8.5, color=NATURE_COLORS['red'], alpha=0.8)

    ax.set_xlabel('Per-Slot Time (ms)')
    ax.set_ylabel('Data Utility')
    ax.set_xlim(-5, 320)
    ax.set_ylim(0.25, 0.75)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'tradeoff.pdf'))
    plt.close()
    print("Generated: tradeoff.pdf")


# ============================================================
# FIG 6: TRUST EVOLUTION (renamed: Delivery-Reliability Score)
# ============================================================
def plot_trust_evolution():
    np.random.seed(77)
    slots = np.arange(0, 501)

    # Honest agents: start at 0.5, rise to ~0.85, stay stable
    honest_mean = 0.5 + 0.35 * (1 - np.exp(-slots / 60))
    honest_noise = np.random.normal(0, 0.015, len(slots))
    honest = np.clip(honest_mean + honest_noise, 0, 1)

    # Corrupted agents: start at 0.5, drop to ~0.15
    corrupt_mean = 0.5 - 0.35 * (1 - np.exp(-slots / 40))
    corrupt_noise = np.random.normal(0, 0.02, len(slots))
    corrupt = np.clip(corrupt_mean + corrupt_noise, 0, 1)

    # Bands (multiple agents)
    honest_upper = np.clip(honest + 0.04, 0, 1)
    honest_lower = np.clip(honest - 0.04, 0, 1)
    corrupt_upper = np.clip(corrupt + 0.05, 0, 1)
    corrupt_lower = np.clip(corrupt - 0.05, 0, 1)

    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    ax.fill_between(slots, honest_lower, honest_upper,
                    alpha=0.15, color=NATURE_COLORS['blue'])
    ax.plot(slots, honest, color=NATURE_COLORS['blue'], label='Honest agents')

    ax.fill_between(slots, corrupt_lower, corrupt_upper,
                    alpha=0.15, color=NATURE_COLORS['red'])
    ax.plot(slots, corrupt, color=NATURE_COLORS['red'], label='Corrupted agents',
            linestyle='--')

    # Exclusion threshold
    ax.axhline(y=0.3, color=NATURE_COLORS['light_gray'], linestyle=':', linewidth=1)
    ax.text(420, 0.32, 'exclusion\nthreshold', fontsize=6,
            color=NATURE_COLORS['light_gray'])

    ax.set_xlabel('Time Slot')
    ax.set_ylabel('Reliability Score $\\tau_v^t$')
    ax.set_xlim(0, 500)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, loc='center right')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'trust_evolution.pdf'))
    plt.close()
    print("Generated: trust_evolution.pdf")


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("Generating SYNTHETIC charts for CoTDA planning draft...")
    print("WARNING: Replace with real data before submission!")
    print("-" * 55)

    plot_convergence()
    plot_sensitivity()
    plot_ablation_bar()
    plot_tradeoff()
    plot_trust_evolution()

    print("-" * 55)
    print(f"All charts saved to {OUT_DIR}/")
