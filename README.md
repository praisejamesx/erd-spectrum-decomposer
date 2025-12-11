# Elegant Recursive Discovery (ERD) Engine

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![arXiv](https://img.shields.io/badge/arXiv-2405.XXXXX-b31b1b.svg)](https://arxiv.org/abs/2405.XXXXX)

An autonomous AI framework for **explanatory decomposition** of scientific data. Extracts interpretable models from complex spectra **without manual parameter input**.

> **Paper**: ["Elegant Recursive Discovery (ERD): An Autonomous AI Framework for Explanatory Decomposition of Scientific Data"](https://arxiv.org/abs/2405.XXXXX)

## Quick Start

### Installation
```bash
# Clone and install
git clone https://github.com/praisejamesx/erd-spectrum-decomposer.git
cd erd-spectrum-decomposer
pip install -e .
```

### Run the Validation Examples
```bash
# 1. Quartz with 5% noise (robustness test)
python examples/01_analyze_quartz.py

# 2. Calcite (autonomous Lorentzian selection)
python examples/02_analyze_calcite.py

# 3. Full benchmark vs. manual fitting
python examples/03_run_benchmark.py
```

## What Problem Does ERD Solve?

Traditional spectral decomposition requires expert intuition for:
- **Initial guesses** (amplitude, center, width for each peak)
- **Lineshape selection** (Gaussian vs. Lorentzian)
- **Baseline modeling**

ERD eliminates **all manual parameter input** through:
1. **Recursive Model Decomposition** – Breaks complex fitting into simple steps
2. **Autonomous Hypothesis Generation** – Parallel searchers propose candidate components
3. **Elegance Bias Scoring** – Theory-guided selection of physically plausible forms

## Key Results from the Paper

| Task | ERD Performance | Manual Input Required |
|------|----------------|----------------------|
| Quartz (5% noise) | Identified peaks at 464.2 cm⁻¹ and 204.7 cm⁻¹ (R²=0.685) | **0 parameters** |
| Calcite | Autonomously selected Lorentzian lineshapes (R²=0.910) | **0 parameters** |
| Benchmark | 100% reduction in manual input vs. standard fitting | **8 → 0 parameters** |

## How It Works: The Dual-Process Architecture

```python
# System 1: Fast, parallel hypothesis generation
searchers = [GaussianSearcher(), LorentzianSearcher(), 
             ExponentialDecaySearcher(), PolynomialSearcher()]

# System 2: Deliberative selection with elegance bias
# Selection Score S = R² × E (E = elegance score 0-1)
# Higher E for physically motivated forms (Lorentzian: 0.95, Gaussian: 0.90)

# Recursive loop: Fit → Select → Subtract → Repeat
```

The engine autonomously builds the final model:  
**M_final(x) = Σ C_k(x)** where each C_k is the most elegant component at that recursion depth.

## 🔧 Using ERD in Your Research

```python
from erd import ElegantRecursiveDiscovery, plot_results
import numpy as np

# 1. Prepare your data
x = np.linspace(100, 1000, 500)
y = your_spectral_data  # Replace with your measurements

# 2. Initialize and run (choose mode)
erd = ElegantRecursiveDiscovery(philosophy='comprehensive')  # or 'purist'
components, final_model, r2 = erd.discover(x, y, verbose=True)

# 3. Inspect results
print(f"Discovered {len(components)} components")
print(f"Final R²: {r2:.3f}")
for comp in components:
    print(f"  {comp}")

# 4. Visualize
plot_results(x, y, components, clean_signal=None)
```

### Philosophy Modes
- **`purist`**: Stops early (residual < 0.5%), avoids overfitting
- **`comprehensive`**: Continues deeper (residual < 2.0%), finds weak signals

## Project Structure

```
spectrum-decomposer/
├── erd/                    # Core engine
│   ├── engine.py          # Main ERD class
│   ├── searchers/         # System 1: Gaussian, Lorentzian, etc.
│   └── utils.py           # Data loading, noise addition, plotting
├── examples/              # Paper validation scripts
│   ├── 01_analyze_quartz.py
│   ├── 02_analyze_calcite.py
│   └── 03_run_benchmark.py
├── data/                  # RRUFF spectra (see data/README.md)
├── tests/                 # Unit tests
├── requirements.txt       # Dependencies
├── setup.py              # Package installation
└── README.md             # This file
```

## Data Requirements

The validation examples use Raman spectra from the [RRUFF Database](https://rruff.info/). To run them:

1. **Download the data files**:
   - [Quartz (R100134)](https://rruff.info/R100134) → Save as `data/quartz_data.txt`
   - [Calcite (R050048)](https://rruff.info/R050048) → Save as `data/calcite_data.txt`

2. **Place in `data/` directory** (create if needed)

Detailed instructions: [data/README_data.md](data/README_data.md)

## Citation

If you use ERD in your research, please cite:

```bibtex
@article{james2024erd,
  title={Elegant Recursive Discovery (ERD): An Autonomous AI Framework for Explanatory Decomposition of Scientific Data},
  author={James, Praise},
  preprint={Zenodo. doi:10.5281/zenodo.17879292.},
  year={2025}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- The [RRUFF Project](https://rruff.info/) for open-access Raman spectra
- Independent research project conducted by Praise James
```

