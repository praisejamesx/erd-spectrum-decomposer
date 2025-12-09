#!/usr/bin/env python3
"""
Example 3: Benchmark ERD against standard manual curve fitting.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from erd import ElegantRecursiveDiscovery, load_ruff_data, add_noise


def composite_gaussian(x, A_exp, k_exp, A1, mu1, sigma1, A2, mu2, sigma2):
    """Manual 3-component model (Exponential Baseline + 2 Gaussians)."""
    exp_decay = A_exp * np.exp(-k_exp * x)
    gauss1 = A1 * np.exp(-((x - mu1) / sigma1)**2)
    gauss2 = A2 * np.exp(-((x - mu2) / sigma2)**2)
    return exp_decay + gauss1 + gauss2


def calculate_r2(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0


def run_benchmark():
    print("=" * 60)
    print("BENCHMARK: ERD vs. Standard Curve Fitting (Noisy Data)")
    print("=" * 60)

    # --- Load and prepare data ---
    data_path = './data/quartz_data.txt'  # Placeholder path
    try:
        x, y_clean = load_ruff_data(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}.")
        print("Please download the quartz data and update the path.")
        return

    y_noisy = add_noise(y_clean, noise_level=0.05, seed=42)

    # --- A. Run ERD Engine ---
    print("A. Running ERD Engine (Autonomous Discovery)...")
    erd = ElegantRecursiveDiscovery(philosophy='comprehensive', max_components=5)
    erd_components, _, erd_r2 = erd.discover(x, y_noisy, verbose=False)
    erd_total_components = len(erd_components)

    # --- B. Run Standard Method ---
    print("\nB. Running Standard Method (Manual Guess & Fit)...")
    initial_guesses = [800, 0.003, 5500, 464, 7, 300, 205, 15]

    try:
        popt, _ = curve_fit(composite_gaussian, x, y_noisy,
                            p0=initial_guesses, maxfev=5000)
        y_pred_std = composite_gaussian(x, *popt)
        std_r2 = calculate_r2(y_noisy, y_pred_std)
        std_peaks_found = 2
        success = True
    except RuntimeError:
        print("Standard Fit FAILED to converge.")
        std_r2 = 0.0
        std_peaks_found = 0
        success = False

    # --- C. Comparison ---
    print("\n" + "=" * 60)
    print("FINAL BENCHMARK RESULTS")
    print("=" * 60)

    results_table = (
        f"{'Metric':<30} {'ERD Engine (Auto)':<25} {'Standard Fit (Manual)':<25}\n"
        f"{'-'*75}\n"
        f"{'Final R² on Noisy Data':<30} {erd_r2:.6f}{'':<17} {std_r2:.6f}\n"
        f"{'Total Components Found':<30} {erd_total_components}{'':<24} {std_peaks_found + 1}\n"
        f"{'Input Parameters Required':<30} {'0 (Fully Autonomous)':<25} {len(initial_guesses)} (Manual Guesses)\n"
        f"{'Success Condition':<30} {'Autonomous Pattern Match':<25} "
        f"{'Successful Convergence' if success else 'FAILED'}\n"
    )
    print(results_table)

    # --- D. Plot ---
    if success:
        plt.figure(figsize=(10, 6))
        plt.plot(x, y_noisy, 'k.', label='Noisy Data (5% Noise)', alpha=0.3)
        plt.plot(x, y_pred_std, 'r-', linewidth=2,
                 label=f'Standard Fit (R²={std_r2:.4f})')

        y_pred_erd = np.zeros_like(x)
        for comp in erd_components:
            y_pred_erd += comp.evaluate(x)
        plt.plot(x, y_pred_erd, 'b--', linewidth=2,
                 label=f'ERD Engine (R²={erd_r2:.4f})')

        plt.title('ERD vs. Standard Curve Fitting on Noisy Quartz Spectrum')
        plt.xlabel('Raman Shift ($\mathrm{cm}^{-1}$)')
        plt.ylabel('Intensity')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()


if __name__ == "__main__":
    run_benchmark()