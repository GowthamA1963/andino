#!/usr/bin/env python3
"""
Plot Results Script
===================
Generates publication-quality figures from the CSV files produced by data_logger.
Requires: matplotlib, numpy (Removes pandas dependency for better portability)

Figures generated:
1. Trajectory Tracking (XY plot with reference and robot path)
2. Tracking Errors (Position and Heading over time)
3. Control Signals (Model vs RL contribution)
4. RL Correction Magnitude
"""

import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def plot_run(csv_path):
    # Set aesthetics
    try:
        plt.style.use('seaborn-v0_8-paper')
    except:
        plt.style.use('ggplot') # Fallback

    params = {
        'axes.labelsize': 10,
        'font.size': 10,
        'legend.fontsize': 8,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.figsize': [12, 10],
        'font.family': 'serif'
    }
    plt.rcParams.update(params)

    # Load data using csv module
    data = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})

    # Convert to numpy arrays for easier plotting
    t = np.array([d['time'] for d in data])
    x = np.array([d['x'] for d in data])
    y = np.array([d['y'] for d in data])
    ref_x = np.array([d['ref_x'] for d in data])
    ref_y = np.array([d['ref_y'] for d in data])
    e_pos = np.array([d['e_pos'] for d in data])
    v_model = np.array([d['v_model'] for d in data])
    v_rl = np.array([d['v_rl'] for d in data])
    rl_norm = np.array([d['rl_norm'] for d in data])
    rms_pos = np.array([d['rms_pos'] for d in data])

    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(f"Hybrid Control Performance Analysis\nSource: {os.path.basename(csv_path)}", fontsize=12)

    # 1. XY Trajectory Plot
    axs[0, 0].plot(ref_x, ref_y, 'g--', label='Reference Path', alpha=0.7)
    axs[0, 0].plot(x, y, 'b-', label='Actual Robot Path', linewidth=1.2)
    
    # NEW: Show robot start point
    axs[0, 0].plot(x[0], y[0], 'ro', markersize=8, label='Start Position', zorder=5)
    
    # NEW: Legend for disturbances (clearly labeled)
    # Zone A (Blue) at (2,0), Zone B (Red) at (0,2), Zone C (Green) at (0,-2)
    axs[0, 0].add_patch(plt.Circle((2.0, 0.0), 0.7, color='blue', alpha=0.2, label='Low friction (μ=0.1)'))
    axs[0, 0].add_patch(plt.Circle((0.0, 2.0), 0.7, color='red', alpha=0.2, label='High friction (μ=3.0)'))
    axs[0, 0].add_patch(plt.Circle((0.0, -2.0), 0.7, color='green', alpha=0.2, label='Nominal friction (μ=0.7)'))
    
    axs[0, 0].set_title('Path Tracking Performance')
    axs[0, 0].set_xlabel('X [m]')
    axs[0, 0].set_ylabel('Y [m]')
    axs[0, 0].legend(loc='upper right', frameon=True, fontsize=7)
    axs[0, 0].axis('equal')
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)

    # 2. Tracking Errors
    axs[0, 1].plot(t, e_pos, 'r-', label='Pos Error [m]', alpha=0.8)
    axs[0, 1].plot(t, rms_pos, 'k-', label='RMS Error', linewidth=1.5)
    
    # NEW: Add final RMS value annotation
    final_rms = rms_pos[-1]
    axs[0, 1].text(0.5, 0.9, f"Final RMS error = {final_rms:.2f} m", 
                  transform=axs[0, 1].transAxes, fontsize=10, fontweight='bold',
                  bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round'))

    axs[0, 1].set_title('Tracking Error (m)')
    axs[0, 1].set_xlabel('Time [s]')
    axs[0, 1].set_ylabel('Error [m]')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # 3. Control Action Comparison
    axs[1, 0].plot(t, v_model, 'b-', label='v_model', alpha=0.4)
    axs[1, 0].plot(t, v_rl, 'r-', label='v_RL (Correction)', linewidth=1.2)
    axs[1, 0].set_title('Linear Velocity: Model vs RL')
    axs[1, 0].set_xlabel('Time [s]')
    axs[1, 0].set_ylabel('v [m/s]')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)

    # 4. RL Correction Magnitude
    axs[1, 1].plot(t, rl_norm, 'm-', label='RL Magnitude')
    axs[1, 1].set_title('RL Disturbance Compensation Effort')
    axs[1, 1].set_xlabel('Time [s]')
    axs[1, 1].set_ylabel('|Δu|')
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save results
    pdf_path = csv_path.replace('.csv', '_perf_report.pdf')
    png_path = csv_path.replace('.csv', '_perf_report.png')
    
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=120) 
    
    print(f"Successfully generated plots:")
    print(f"  PDF (Paper-ready): {pdf_path}")
    print(f"  PNG (Quick view):  {png_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 plot_results.py <path_to_csv>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if os.path.exists(csv_file):
        plot_run(csv_file)
    else:
        print(f"Error: File {csv_file} not found.")
