import numpy as np
from erd.core import ElegantPolynomialSearcher, ElegantExponentialSearcher, GaussianPeakSearcher, ElegantLorentzianSearcher

# =============================================================================
# PATTERN RECOGNITION ENGINE
# =============================================================================
class SpectralPatternEngine:
    """
    The 'artificial intuition' module that classifies residuals and proposes
    component types. Implements dynamic thresholds for 'purist' vs 'comprehensive' modes.
    """
    def __init__(self, philosophy='purist'):
        if philosophy == 'comprehensive':
            self.MIN_R2_BASELINE = 0.01
            self.MIN_R2_PEAK = 0.0001
            self.MIN_AMP_TO_NOISE = 4.0
            self.GLOBAL_STOP_VAR_THRESHOLD = 0.005
        else:  # 'purist' (default)
            self.MIN_R2_BASELINE = 0.05
            self.MIN_R2_PEAK = 0.001
            self.MIN_AMP_TO_NOISE = 5.0
            self.GLOBAL_STOP_VAR_THRESHOLD = 0.005

    def _extract_features(self, y_residual, total_variance):
        """Calculates simple features to categorize the residual."""
        residual_variance = np.var(y_residual) / total_variance
        max_amplitude = np.max(y_residual)
        noise_floor = np.std(y_residual)
        amplitude_to_noise = max_amplitude / (noise_floor + 1e-6)
        smoothness_index = np.max(y_residual) / (np.max(np.abs(np.diff(y_residual))) + 1e-6)

        return residual_variance, amplitude_to_noise, smoothness_index

    def analyze(self, x, y_residual, depth, current_components, searchers, total_variance):
        """
        Main pattern analysis logic. Decides what type of component to fit next.
        """
        residual_var, amp_to_noise, smooth_idx = self._extract_features(y_residual, total_variance)

        # Rule 1: STOP if residual variance is extremely low AND amplitude-to-noise is low
        if residual_var < self.GLOBAL_STOP_VAR_THRESHOLD and amp_to_noise < 4.0:
            return {'component_type': 'noise', 'command': 'STOP_SEARCH'}

        # Rule 2: Peak (High-Frequency, Localized Structure) - Prioritized after Depth 1
        if amp_to_noise > self.MIN_AMP_TO_NOISE and depth > 0:
            gauss_comp = searchers['gaussian_peak'].search(x, y_residual, {})
            lor_comp = searchers['lorentzian_peak'].search(x, y_residual, {})

            candidates = [
                c for c in [gauss_comp, lor_comp]
                if c is not None and c.variance_explained > self.MIN_R2_PEAK
            ]

            if candidates:
                best_comp = max(candidates, key=lambda c: c.variance_explained * c.elegance_score)
                return {
                    'component_type': best_comp.component_type,
                    'command': f'FIT_{best_comp.component_type.upper()}'
                }

        # Rule 3: Baseline (Low-Frequency, Broad Structure)
        if smooth_idx > 50.0 or depth == 0:
            poly_comp = searchers['polynomial'].search(x, y_residual, {})
            exp_comp = searchers['exponential_decay'].search(x, y_residual, {})

            candidates = [
                c for c in [poly_comp, exp_comp]
                if c is not None and c.variance_explained > self.MIN_R2_BASELINE
            ]

            if candidates:
                best_comp = max(candidates, key=lambda c: c.variance_explained * c.elegance_score)
                return {
                    'component_type': best_comp.component_type,
                    'command': f'FIT_{best_comp.component_type.upper()}'
                }

        # Default Fallback: STOP
        return {'component_type': 'noise', 'command': 'STOP_SEARCH'}


# =============================================================================
# ELEGANT RECURSIVE DISCOVERY ENGINE
# =============================================================================
class ElegantRecursiveDiscovery:
    """
    Main ERD engine class. Orchestrates the Recursive Model Decomposition (RMD) loop.
    """
    def __init__(self, philosophy='purist', max_components=5):
        self.max_components = max_components
        self.philosophy = philosophy.lower()

        self.searchers = {
            'polynomial': ElegantPolynomialSearcher(),
            'exponential_decay': ElegantExponentialSearcher(),
            'gaussian_peak': GaussianPeakSearcher(),
            'lorentzian_peak': ElegantLorentzianSearcher()
        }

        self.pattern_engine = SpectralPatternEngine(philosophy=self.philosophy)
        self.components = []

    def discover(self, x, y_true, verbose=True):
        """
        Executes the full RMD loop to decompose the signal y_true over domain x.

        Parameters
        ----------
        x : array_like
            Independent variable (e.g., Raman shift in cm⁻¹).
        y_true : array_like
            Signal to decompose.
        verbose : bool, optional
            Whether to print progress.

        Returns
        -------
        components : list of ElegantComponentNode
            The discovered components.
        final_expr : str
            The formal linear superposition model expression.
        final_r2 : float
            R² of the full reconstructed model against y_true.
        """
        current_target = y_true.copy()
        total_variance = np.var(y_true)

        for depth in range(self.max_components):
            hypothesis = self.pattern_engine.analyze(
                x, current_target, depth, self.components,
                self.searchers, total_variance
            )

            if hypothesis['command'] == 'STOP_SEARCH':
                if verbose:
                    print("Stopping: Residual classified as noise or negligible variance.")
                break

            if verbose:
                print(f"Depth {depth+1}: {hypothesis['command']} (Auto-Generated Hypothesis)")

            component_type = hypothesis['component_type']
            component = self.searchers[component_type].search(x, current_target, hypothesis)

            if component is None or component.variance_explained < 0.0001:
                if verbose:
                    print(f"Stopping: {component_type} search failed or negligible variance explained.")
                break

            component_pred = component.evaluate(x)
            current_target = current_target - component_pred
            component.depth = depth
            self.components.append(component)

            if verbose:
                print(f"  Found: {component}\n")

        return self.components, self._compose_expression(), self._calculate_final_r2(x, y_true)

    def _compose_expression(self):
        """Generates the formal linear superposition model expression."""
        if not self.components:
            return "0"

        expression_parts = [f"({comp.clean_expression})" for comp in self.components]
        return "Y_pred(x) = " + " + ".join(expression_parts)

    def _calculate_final_r2(self, x, y_true):
        y_pred = np.zeros_like(x)
        for comp in self.components:
            y_pred += comp.evaluate(x)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0