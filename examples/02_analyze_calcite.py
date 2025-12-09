#!/usr/bin/env python3
"""
Example 2: Analyze calcite Raman spectrum (Lorentzian generality test).
Demonstrates autonomous lineshape selection.
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from erd import ElegantRecursiveDiscovery, plot_results, load_ruff_data

def main():
    print("=" * 70)
    print("ELEGANT RECURSIVE DISCOVERY: Calcite (Lorentzian) Generality Test")
    print("=" * 70)

    # --- Load Data ---
    # USER: Update this path to your actual calcite data file
    data_path = './data/calcite_data.txt'  # Placeholder path
    try:
        x, y_clean = load_ruff_data(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found at {data_path}.")
        print("Please download the calcite data from RRUFF (R050048) and update the path.")
        return

    y = y_clean  # Using clean data for clear lineshape competition

    print(f"Loaded spectrum: {len(x)} points from {x[0]:.1f} to {x[-1]:.1f} cm⁻¹")
    print("NOTE: Data is running with 0.0% Noise added (Testing Generality).\n")

    # --- Run ERD (Purist Mode) ---
    print("Running ERD Engine in PURIST mode...")
    erd = ElegantRecursiveDiscovery(philosophy='purist', max_components=6)
    components, final_expr, final_r2 = erd.discover(x, y, verbose=True)

    # --- Display Results ---
    print("\n" + "=" * 70)
    print("DECOMPOSITION COMPLETE: FORMAL SCIENTIFIC MODEL")
    print("=" * 70)
    print(f"Engine Philosophy: PURIST")
    print(f"Total Components Found: {len(components)}")
    print(f"Final R² (Model Elegantly Fits Data): {final_r2:.6f}")

    print("\nComponents Found (with Elegance Scores):")
    for i, comp in enumerate(components):
        if comp.component_type in ['polynomial', 'exponential_decay']:
            print(f"  {i+1}. BASELINE ({comp.component_type.split('_')[0].capitalize()}): "
                  f"{comp.clean_expression}")
        else:
            params = comp.parameters
            print(f"  {i+1}. PEAK ({comp.component_type.split('_')[0]}): "
                  f"Center={params['center']:.1f} cm⁻¹, "
                  f"Height={params['amplitude']:.1f}, "
                  f"Score={comp.elegance_score:.2f}, R²={comp.variance_explained:.4f}")

    print("\n--- FINAL LINEAR SUPERPOSITION MODEL ---")
    print(final_expr)

    # --- Visualize ---
    plot_results(x, y, components, y_clean)

if __name__ == "__main__":
    main()