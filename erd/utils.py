import numpy as np
import matplotlib.pyplot as plt

def plot_results(x, y_true, components, y_clean=None):
    """
    Generates a standard 2x2 diagnostic plot for ERD decomposition results.

    Parameters
    ----------
    x : array_like
        Independent variable.
    y_true : array_like
        Original (potentially noisy) signal.
    components : list of ElegantComponentNode
        Components discovered by ERD.
    y_clean : array_like, optional
        Clean/reference signal for comparison.
    """
    y_pred = np.zeros_like(x)
    for comp in components:
        y_pred += comp.evaluate(x)

    plt.figure(figsize=(12, 8))

    # Subplot 1: Spectrum vs. ERD Reconstruction
    plt.subplot(2, 2, 1)
    plt.plot(x, y_true, 'k-', linewidth=2, label='Noisy Spectrum', alpha=0.7)
    plt.plot(x, y_pred, 'r--', linewidth=2, label='ERD Reconstruction')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title('Spectrum vs. ERD Fit (Final Synthesis)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 2: Clean Signal vs. Noisy Data & Fit
    plt.subplot(2, 2, 2)
    if y_clean is not None:
        plt.plot(x, y_clean, 'k:', linewidth=1, label='Original Clean Signal')
    plt.plot(x, y_true, 'k-', linewidth=0.5, label='Noisy Signal', alpha=0.4)
    plt.plot(x, y_pred, 'r--', linewidth=2, label='ERD Fit')
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title('Clean Signal vs. Noisy Data & Fit')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Subplot 3: Final Residual
    residual = y_true - y_pred
    plt.subplot(2, 2, 3)
    plt.plot(x, residual, 'b-', linewidth=1.5)
    plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    plt.xlabel('Raman Shift (cm⁻¹)')
    plt.ylabel('Intensity')
    plt.title('Final Residual (Should look like noise)')
    plt.grid(True, alpha=0.3)

    # Subplot 4: Component Types
    plt.subplot(2, 2, 4)
    types = [comp.component_type for comp in components]
    type_counts = {t: types.count(t) for t in set(types)}
    bars = plt.bar(type_counts.keys(), type_counts.values(), color='green', alpha=0.7)
    plt.ylabel('Number Found')
    plt.title('Component Types')
    for bar in bars:
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            str(int(bar.get_height())),
            ha='center', va='bottom'
        )

    plt.tight_layout()
    plt.show()


def load_ruff_data(filepath, delimiter=','):
    """
    Simple helper to load RRUFF-style CSV data.

    Parameters
    ----------
    filepath : str
        Path to the data file.
    delimiter : str, optional
        Column delimiter.

    Returns
    -------
    x : ndarray
        First column (Raman shift).
    y : ndarray
        Second column (Intensity).
    """
    data = np.loadtxt(filepath, delimiter=delimiter)
    return data[:, 0], data[:, 1]


def add_noise(y_clean, noise_level=0.05, seed=42):
    """
    Adds Gaussian noise to a clean signal.

    Parameters
    ----------
    y_clean : array_like
        Clean signal.
    noise_level : float, optional
        Noise amplitude as fraction of signal range.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    y_noisy : ndarray
        Noisy signal.
    """
    np.random.seed(seed)
    intensity_range = np.max(y_clean) - np.min(y_clean)
    noise_amplitude = noise_level * intensity_range
    noise = np.random.normal(0, noise_amplitude, size=y_clean.shape)
    return y_clean + noise