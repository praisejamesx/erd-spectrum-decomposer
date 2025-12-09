import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter1d
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# ELEGANT COMPONENT NODE
# =============================================================================
class ElegantComponentNode:
    """
    Represents a single, interpretable component found by the ERD engine.
    """
    def __init__(self, component_type, expression, parameters=None,
                 variance_explained=0.0, elegance_score=1.0):
        self.component_type = component_type
        self.expression = expression
        self.parameters = parameters or {}
        self.variance_explained = variance_explained
        self.elegance_score = elegance_score
        self.clean_expression = self._clean_expression()

    def evaluate(self, x):
        """Evaluate the component's expression over the array x."""
        try:
            eval_expr = self.expression.replace("²", "**2").replace(" ", "")
            return eval(eval_expr, {'x': x, 'np': np, 'sin': np.sin, 'exp': np.exp})
        except:
            # Fallback to direct parameter evaluation
            if self.component_type == 'polynomial':
                a = self.parameters.get('a', 0)
                b = self.parameters.get('b', 0)
                c = self.parameters.get('c', 0)
                return a * x**2 + b * x + c
            elif self.component_type == 'gaussian_peak':
                A = self.parameters.get('amplitude', 0)
                μ = self.parameters.get('center', 0)
                σ = self.parameters.get('width', 1)
                sigma_sq = σ**2 if σ > 0 else 1
                return A * np.exp(-((x - μ)**2) / (2 * sigma_sq))
            elif self.component_type == 'exponential_decay':
                A = self.parameters.get('amplitude', 0)
                k = self.parameters.get('decay_rate', 0.1)
                return A * np.exp(-k * x)
            elif self.component_type == 'lorentzian_peak':
                A = self.parameters.get('amplitude', 0)
                μ = self.parameters.get('center', 0)
                Γ = self.parameters.get('width_fwhm', 1)
                return A * (Γ**2 / ((x - μ)**2 + (Γ/2)**2))
            return np.zeros_like(x)

    def _clean_expression(self):
        expr = self.expression
        if expr.startswith("+ "):
            expr = expr[2:]
        return expr.replace("  ", " ").replace(" )", ")")

    def __str__(self):
        return (f"{self.component_type}: {self.clean_expression} "
                f"(R²={self.variance_explained:.3f}, elegance={self.elegance_score:.2f})")


# =============================================================================
# SEARCHER CLASSES (AUTONOMOUS HYPOTHESIS GENERATION)
# =============================================================================
class ElegantPolynomialSearcher:
    """Searcher for 2nd-order polynomial baseline components."""
    def search(self, x, y_target, hypothesis):
        try:
            coeffs = np.polyfit(x, y_target, 2)
            a, b, c = coeffs
            y_pred = a * x**2 + b * x + c
            ss_res = np.sum((y_target - y_pred)**2)
            ss_tot = np.sum((y_target - np.mean(y_target))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            # Build readable expression
            terms = []
            if abs(a) > 1e-10:
                terms.append(f"{a:.4f}x²")
            if abs(b) > 1e-10:
                terms.append(f"{'+' if b > 0 else ''}{b:.4f}x")
            if abs(c) > 1e-10:
                terms.append(f"{'+' if c > 0 else ''}{c:.4f}")
            expr = ' '.join(terms).replace('+ ', '+ ').lstrip('+')

            return ElegantComponentNode(
                'polynomial', expr,
                {'a': a, 'b': b, 'c': c},
                r2, self._elegance_score(a, b, c)
            )
        except Exception:
            return None

    def _elegance_score(self, a, b, c):
        """Penalizes complexity in polynomial coefficients."""
        score = 1.0
        if abs(b) > 0.1:
            score *= 0.8
        if abs(c) > 0.1:
            score *= 0.8
        return score


class ElegantExponentialSearcher:
    """Searcher for exponential decay baseline components."""
    @staticmethod
    def _exp_decay(x, A, k):
        return A * np.exp(-k * x)

    def search(self, x, y_target, hypothesis):
        try:
            A_guess = y_target[0] if y_target[0] > 0 else 1.0
            k_guess = 0.001
            bounds = ([0.0, 0.0], [np.inf, np.inf])

            popt, _ = optimize.curve_fit(
                self._exp_decay, x, y_target,
                p0=[A_guess, k_guess], bounds=bounds
            )
            A, k = popt

            y_pred = self._exp_decay(x, A, k)
            ss_res = np.sum((y_target - y_pred)**2)
            ss_tot = np.sum((y_target - np.mean(y_target))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            expr = f"{A:.2f}*exp(-{k:.4f}x)"
            return ElegantComponentNode(
                'exponential_decay', expr,
                {'amplitude': A, 'decay_rate': k},
                r2, 0.95
            )
        except Exception:
            return None


class GaussianPeakSearcher:
    """Searcher for Gaussian peak components."""
    @staticmethod
    def _gaussian(x, A, μ, σ):
        return A * np.exp(-((x - μ) / σ)**2)

    def search(self, x, y_target, hypothesis):
        try:
            # Guided initialization using smoothing
            y_smoothed = gaussian_filter1d(y_target, sigma=5)
            max_idx = np.argmax(y_smoothed)

            A_guess = max(y_target[max_idx], 1e-6)
            μ_guess = x[max_idx]
            σ_guess = (np.max(x) - np.min(x)) * 0.01

            # Tighter bounds for realistic peaks
            max_width = (np.max(x) - np.min(x)) * 0.1
            bounds = (
                [0.0, np.min(x), 1e-3],
                [np.inf, np.max(x), max_width]
            )

            popt, _ = optimize.curve_fit(
                self._gaussian, x, y_target,
                p0=[A_guess, μ_guess, σ_guess],
                bounds=bounds
            )
            A, μ, σ = popt

            y_pred = self._gaussian(x, A, μ, σ)
            ss_res = np.sum((y_target - y_pred)**2)
            ss_tot = np.sum((y_target - np.mean(y_target))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            expr = f"{A:.2f}*exp(-(x-{μ:.1f})²/(2*{σ:.2f}²))"
            return ElegantComponentNode(
                'gaussian_peak', expr,
                {'amplitude': A, 'center': μ, 'width': σ},
                r2, 0.90
            )
        except Exception:
            return None


class ElegantLorentzianSearcher:
    """Searcher for Lorentzian peak components."""
    @staticmethod
    def _lorentzian(x, A, μ, Γ):
        return A * (Γ**2 / ((x - μ)**2 + (Γ/2)**2))

    def search(self, x, y_target, hypothesis):
        try:
            # Guided initialization
            y_smoothed = gaussian_filter1d(y_target, sigma=5)
            max_idx = np.argmax(y_smoothed)

            A_guess = max(y_target[max_idx], 1e-6)
            μ_guess = x[max_idx]
            Γ_guess = (np.max(x) - np.min(x)) * 0.001  # Narrow initial guess

            max_gamma = (np.max(x) - np.min(x)) * 0.1
            bounds = (
                [0.0, np.min(x), 1e-3],
                [np.inf, np.max(x), max_gamma]
            )

            popt, _ = optimize.curve_fit(
                self._lorentzian, x, y_target,
                p0=[A_guess, μ_guess, Γ_guess],
                bounds=bounds
            )
            A, μ, Γ = popt

            y_pred = self._lorentzian(x, A, μ, Γ)
            ss_res = np.sum((y_target - y_pred)**2)
            ss_tot = np.sum((y_target - np.mean(y_target))**2)
            r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            expr = f"{A:.2f}*({Γ:.2f}² / ((x-{μ:.1f})² + ({Γ/2:.2f})²))"
            return ElegantComponentNode(
                'lorentzian_peak', expr,
                {'amplitude': A, 'center': μ, 'width_fwhm': Γ},
                r2, 0.95
            )
        except Exception:
            return None