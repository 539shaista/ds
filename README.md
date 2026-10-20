# Probability Distribution Simulator & Interactive Interface

A complete Data Science toolkit to simulate and plot **Normal (Gaussian)**, **Binomial**, and **Poisson** distributions using built-in functions, complete with theoretical vs. empirical comparisons, statistical convergence metrics, and multiple user interfaces.

---

## 🌟 Available Interfaces

This project provides **three distinct interface options** to fit any workflow:

### 1. 🚀 Instant Web Interface (`index.html`) — *Zero Setup Required*
- **No Python or packages needed** — open instantly in any browser (Safari, Chrome, Firefox).
- **Features**:
  - Live interactive parameter tuning (sliders and inputs).
  - High-DPI Canvas rendering of empirical histograms and relative frequencies.
  - Smooth overlays of theoretical Probability Density Functions (PDF) and Probability Mass Functions (PMF).
  - Real-time convergence dashboard: Theoretical vs. Sample Mean, Variance, Std Dev, Skewness, Kurtosis, and % Error.
  - Side-by-side comparison mode for all three distributions.
  - Interactive hover tooltips for exact bin/frequency details.
  - One-click **Export to CSV** and **Export to PNG**.

**How to run**:
```bash
open index.html
```
*(Or simply double-click `index.html` in Finder).*

---

### 2. 📊 Streamlit Interactive Web Dashboard (`app.py`)
- Python-based dashboard using Streamlit, NumPy, SciPy, and Matplotlib.
- **Features**:
  - Sidebar distribution selector.
  - Parameter controls ($\mu, \sigma, n, p, \lambda$) and sample size $N$ slider.
  - Metric delta cards showing convergence error percentage.
  - Data preview tables and built-in CSV download buttons.

**How to run**:
```bash
pip install -r requirements.txt
streamlit run app.py
```

---

### 3. 🖥️ Native Desktop GUI (`gui_tkinter.py`)
- A standalone desktop window built with Python's standard `tkinter` library and embedded `matplotlib`.
- **Features**:
  - Fast desktop interface with sliders and dropdown selector.
  - Live redraws on parameter adjustment.
  - Embedded Matplotlib canvas with publication-grade graphs.

**How to run**:
```bash
python3 gui_tkinter.py
```

---

### 4. 📈 Standalone Simulation Script (`simulation.py`)
- A clean, well-commented script that runs simulations for each distribution using built-in functions and generates publication-ready plots saved directly to disk.

**How to run**:
```bash
python3 simulation.py
```

---

## 📐 Mathematical Formulation & Built-in Functions

### 1. Normal (Gaussian) Distribution (Continuous)
- **Parameters**: Mean $\mu \in \mathbb{R}$, Standard Deviation $\sigma > 0$
- **Probability Density Function (PDF)**:
  $$f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)$$
- **Theoretical Mean**: $\mu$
- **Theoretical Variance**: $\sigma^2$
- **Built-in Functions**:
  - NumPy: `numpy.random.normal(loc=mu, scale=sigma, size=N)`
  - SciPy PDF: `scipy.stats.norm.pdf(x, loc=mu, scale=sigma)`
  - Box-Muller transform (Web): $Z = \sqrt{-2\ln(U_1)} \cos(2\pi U_2)$, $X = \mu + \sigma Z$

---

### 2. Binomial Distribution (Discrete)
- **Parameters**: Number of trials $n \in \mathbb{N}$, Probability of success $p \in [0, 1]$
- **Probability Mass Function (PMF)**:
  $$P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}, \quad k \in \{0, 1, \dots, n\}$$
- **Theoretical Mean**: $n \cdot p$
- **Theoretical Variance**: $n \cdot p \cdot (1 - p)$
- **Built-in Functions**:
  - NumPy: `numpy.random.binomial(n=n, p=p, size=N)`
  - SciPy PMF: `scipy.stats.binom.pmf(k, n=n, p=p)`
  - Bernoulli trials sum (Web): $X = \sum_{i=1}^n \text{Bernoulli}(p)$

---

### 3. Poisson Distribution (Discrete)
- **Parameters**: Rate parameter $\lambda > 0$ (expected number of occurrences)
- **Probability Mass Function (PMF)**:
  $$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k \in \{0, 1, 2, \dots\}$$
- **Theoretical Mean**: $\lambda$
- **Theoretical Variance**: $\lambda$
- **Built-in Functions**:
  - NumPy: `numpy.random.poisson(lam=lambda, size=N)`
  - SciPy PMF: `scipy.stats.poisson.pmf(k, mu=lambda)`
  - Knuth Poisson Generator (Web): multiplication method for generating Poisson variates

---

## 🛠️ Python Environment Setup (if using Python)

If you wish to run the Python scripts (`app.py`, `gui_tkinter.py`, `simulation.py`):

1. **Install Xcode Command Line Tools** (if not already installed on macOS):
   ```bash
   xcode-select --install
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

---

## 📁 Project Structure

```
ds project/
├── index.html            # Standalone interactive Web interface (zero install)
├── app.py                # Interactive Streamlit dashboard
├── gui_tkinter.py        # Desktop Tkinter GUI application
├── simulation.py         # Pure Python simulation & plotting script
├── requirements.txt      # Python dependencies
└── README.md             # Complete documentation
```
