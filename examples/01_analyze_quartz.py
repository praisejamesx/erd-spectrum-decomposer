#!/usr/bin/env python3
"""
Example 1: Analyze noisy quartz Raman spectrum using the ERD engine.
Reproduces the results from the paper.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from erd import ElegantRecursiveDiscovery, plot_results, load_ruff_data, add_noise

def main():
    print("=" * 70)
    print("ELEGANT RECURSIVE DISCOVERY: Quartz Raman Spectrum Analysis")
    print("=" * 70)

    # --- Load Data ---
    # USER: Update this path to your actual quartz data file
    data_path = './data/quartz_data.txt'  # Placeholder path
    try:
        x, y_clean = load_ruff_data(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}.")
        print("Please download the quartz data from RRUFF (R040003) and update the path.")
        return

    # --- Add Noise (5%) ---
    y_noisy = add_noise(y_clean, noise_level=0.05, seed=42)
    y = y_noisy

    print(f"Loaded spectrum: {len(x)} points from {x[0]:.1f} to {x[-1]:.1f} cm⁻¹")
    print("NOTE: Data is running with 5.0% Gaussian Noise added (Seed 42).\n")

    # --- Run ERD (Comprehensive Mode) ---
    print("Running ERD Engine in COMPREHENSIVE mode...")
    erd = ElegantRecursiveDiscovery(philosophy='comprehensive', max_components=6)
    components, final_expr, final_r2 = erd.discover(x, y, verbose=True)

    # --- Display Results ---
    print("\n" + "=" * 70)
    print("DECOMPOSITION COMPLETE: FORMAL SCIENTIFIC MODEL")
    print("=" * 70)
    print(f"Engine Philosophy: COMPREHENSIVE")
    print(f"Total Components Found: {len(components)}")
    print(f"Final R² (Model Elegantly Fits Data/Ignores Noise): {final_r2:.6f}")

    print("\nComponents Found:")
    for i, comp in enumerate(components):
        if comp.component_type in ['polynomial', 'exponential_decay']:
            print(f"  {i+1}. BASELINE ({comp.component_type.split('_')[0].capitalize()}): "
                  f"{comp.clean_expression}")
        else:
            params = comp.parameters
            width = params.get('width', params.get('width_fwhm', 'N/A'))
            print(f"  {i+1}. PEAK ({comp.component_type.split('_')[0]}): "
                  f"Center={params['center']:.1f} cm⁻¹, "
                  f"Height={params['amplitude']:.1f}, Width={width:.1f} cm⁻¹")

    print("\n--- FINAL LINEAR SUPERPOSITION MODEL ---")
    print(final_expr)

    # --- Visualize ---
    plot_results(x, y, components, y_clean)

if __name__ == "__main__":
    main()