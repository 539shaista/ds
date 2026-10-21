"""
Distribution Simulation, Prediction, and Plotting Suite
Simulate and predict Normal, Binomial, and Poisson distributions using built-in functions
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

# Default seed for reproducibility
DEFAULT_SEED = 42

# Ensure output directory exists for saved plots
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def simulate_normal(mu=0.0, sigma=1.0, size=5000, num_bins=40, save_path=None, seed=DEFAULT_SEED):
    """
    Simulate and plot Normal (Gaussian) Distribution using NumPy and SciPy built-ins.
    
    Parameters:
        mu (float): Distribution mean
        sigma (float): Distribution standard deviation
        size (int): Number of simulated samples
        num_bins (int): Number of histogram bins
        save_path (str, optional): Path to save figure
        seed (int, optional): Random seed
        
    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray, dict)
    """
    rng = np.random.default_rng(seed)
    samples = rng.normal(loc=mu, scale=sigma, size=size)
    sample_mean = float(np.mean(samples))
    sample_var = float(np.var(samples, ddof=1))
    sample_std = float(np.std(samples, ddof=1))
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    count, bins, _ = ax.hist(
        samples, bins=num_bins, density=True, alpha=0.6,
        color='#3B82F6', edgecolor='#1D4ED8', label=f'Simulated Histogram (N={size:,})'
    )
    x_grid = np.linspace(min(bins[0], mu - 4 * sigma), max(bins[-1], mu + 4 * sigma), 500)
    theo_pdf = stats.norm.pdf(x_grid, loc=mu, scale=sigma)
    ax.plot(x_grid, theo_pdf, color='#DC2626', linewidth=2.5,
            label=f'Theoretical Probability Density Function $\\mathcal{{N}}({mu}, {sigma}^2)$')
    
    # Statistical Summary Box
    stats_text = (
        f"Theoretical Mean: {mu:.4f}\n"
        f"Sample Mean:      {sample_mean:.4f}\n"
        f"Theoretical Var:  {sigma**2:.4f}\n"
        f"Sample Variance:  {sample_var:.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#CBD5E1', alpha=0.9),
            fontfamily='monospace')
    
    ax.set_title(f'Normal Distribution Simulation\n($\\mu={mu}$, $\\sigma={sigma}$, $N={size:,}$)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Value (x)', fontsize=12)
    ax.set_ylabel('Probability Density', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Normal plot to: {save_path}")
    
    stats_dict = {
        'theoretical_mean': mu,
        'sample_mean': sample_mean,
        'theoretical_var': sigma**2,
        'sample_var': sample_var,
        'sample_std': sample_std
    }
    return fig, samples, stats_dict


def simulate_binomial(n=20, p=0.5, size=5000, save_path=None, seed=DEFAULT_SEED):
    """
    Simulate and plot Binomial Distribution using NumPy and SciPy built-ins.
    
    Parameters:
        n (int): Number of trials
        p (float): Success probability per trial
        size (int): Number of simulated samples
        save_path (str, optional): Path to save figure
        seed (int, optional): Random seed
        
    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray, dict)
    """
    rng = np.random.default_rng(seed)
    samples = rng.binomial(n=n, p=p, size=size)
    sample_mean = float(np.mean(samples))
    sample_var = float(np.var(samples, ddof=1))
    
    theo_mean = n * p
    theo_var = n * p * (1 - p)
    theo_std = np.sqrt(theo_var)
    
    # Range of k to display (focused on non-zero probability region)
    if n > 30:
        k_min = max(0, int(theo_mean - 4 * theo_std))
        k_max = min(n, int(theo_mean + 4 * theo_std))
    else:
        k_min = 0
        k_max = n
        
    k_vals = np.arange(k_min, k_max + 1)
    counts = np.bincount(samples, minlength=n + 1)
    empirical_prob = counts[k_min:k_max + 1] / size
    theo_pmf = stats.binom.pmf(k_vals, n=n, p=p)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    bar_width = 0.4
    ax.bar(k_vals - bar_width/2, empirical_prob, width=bar_width, alpha=0.7,
           color='#10B981', edgecolor='#047857', label=f'Simulated Relative Freq (N={size:,})')
    ax.bar(k_vals + bar_width/2, theo_pmf, width=bar_width, alpha=0.7,
           color='#F59E0B', edgecolor='#B45309', label=f'Theoretical PMF $B({n}, {p})$')
    
    # Statistical Summary Box
    stats_text = (
        f"Theoretical Mean: {theo_mean:.4f}\n"
        f"Sample Mean:      {sample_mean:.4f}\n"
        f"Theoretical Var:  {theo_var:.4f}\n"
        f"Sample Variance:  {sample_var:.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#CBD5E1', alpha=0.9),
            fontfamily='monospace')
    
    ax.set_title(f'Binomial Distribution Simulation\n($n={n}$ trials, $p={p}$, $N={size:,}$)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Successes (k)', fontsize=12)
    ax.set_ylabel('Probability $P(X = k)$', fontsize=12)
    ax.set_xticks(k_vals[::max(1, len(k_vals) // 15)])
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Binomial plot to: {save_path}")
        
    stats_dict = {
        'theoretical_mean': theo_mean,
        'sample_mean': sample_mean,
        'theoretical_var': theo_var,
        'sample_var': sample_var
    }
    return fig, samples, stats_dict


def simulate_poisson(lam=4.0, size=5000, save_path=None, seed=DEFAULT_SEED):
    """
    Simulate and plot Poisson Distribution using NumPy and SciPy built-ins.
    
    Parameters:
        lam (float): Rate parameter lambda
        size (int): Number of simulated samples
        save_path (str, optional): Path to save figure
        seed (int, optional): Random seed
        
    Returns:
        tuple: (matplotlib.figure.Figure, np.ndarray, dict)
    """
    rng = np.random.default_rng(seed)
    samples = rng.poisson(lam=lam, size=size)
    sample_mean = float(np.mean(samples))
    sample_var = float(np.var(samples, ddof=1))
    
    theo_mean = lam
    theo_var = lam
    theo_std = np.sqrt(lam)
    
    k_min = max(0, int(lam - 4 * theo_std)) if lam > 20 else 0
    k_max = max(int(np.max(samples)), int(lam + 4 * theo_std))
    
    k_vals = np.arange(k_min, k_max + 1)
    counts = np.bincount(samples, minlength=k_max + 1)
    empirical_prob = counts[k_min:k_max + 1] / size
    theo_pmf = stats.poisson.pmf(k_vals, mu=lam)
    
    fig, ax = plt.subplots(figsize=(8, 5), dpi=120)
    bar_width = 0.4
    ax.bar(k_vals - bar_width/2, empirical_prob, width=bar_width, alpha=0.7,
           color='#8B5CF6', edgecolor='#6D28D9', label=f'Simulated Relative Freq (N={size:,})')
    ax.bar(k_vals + bar_width/2, theo_pmf, width=bar_width, alpha=0.7,
           color='#EC4899', edgecolor='#BE185D', label=f'Theoretical PMF Poisson($\\lambda={lam}$)')
    
    # Statistical Summary Box
    stats_text = (
        f"Theoretical Mean: {theo_mean:.4f}\n"
        f"Sample Mean:      {sample_mean:.4f}\n"
        f"Theoretical Var:  {theo_var:.4f}\n"
        f"Sample Variance:  {sample_var:.4f}"
    )
    ax.text(0.97, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#CBD5E1', alpha=0.9),
            fontfamily='monospace')
    
    ax.set_title(f'Poisson Distribution Simulation\n($\\lambda={lam}$, $N={size:,}$)',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('Events (k)', fontsize=12)
    ax.set_ylabel('Probability $P(X = k)$', fontsize=12)
    ax.set_xticks(k_vals[::max(1, len(k_vals) // 15)])
    ax.grid(True, linestyle='--', alpha=0.5, axis='y')
    ax.legend(loc='upper left')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"Saved Poisson plot to: {save_path}")
        
    stats_dict = {
        'theoretical_mean': theo_mean,
        'sample_mean': sample_mean,
        'theoretical_var': theo_var,
        'sample_var': sample_var
    }
    return fig, samples, stats_dict


def fit_and_predict_dataset(csv_path, column_name=None, model_type="auto", save_path=None, seed=DEFAULT_SEED):
    """
    Load a dataset, estimate/predict parameters for Normal, Binomial, or Poisson,
    and generate a plot comparing the actual dataset with the predicted theoretical distribution
    and synthetic re-simulated data.
    
    Parameters:
        csv_path (str): Path to CSV file
        column_name (str): Column to analyze (defaults to first numeric column)
        model_type (str): 'auto', 'normal', 'binomial', or 'poisson'
        save_path (str): Output image path
        seed (int, optional): Random seed for re-simulation
    """
    df = pd.read_csv(csv_path)
    if column_name is None:
        column_name = df.select_dtypes(include=[np.number]).columns[0]
        
    data = df[column_name].dropna().values
    n = len(data)
    mean_val = float(np.mean(data))
    var_val = float(np.var(data, ddof=1)) if n > 1 else 0.0
    std_val = float(np.std(data, ddof=1)) if n > 1 else 0.0
    
    # Auto-detection heuristic
    if model_type == "auto":
        is_int = np.all(np.floor(data) == data)
        is_pos = np.all(data >= 0)
        ratio = var_val / (mean_val if mean_val != 0 else 1.0)
        if is_int and is_pos and np.max(data) <= 100 and ratio < 1.0:
            model = "binomial"
        elif is_int and is_pos and abs(ratio - 1.0) < 0.6:
            model = "poisson"
        else:
            model = "normal"
    else:
        model = model_type.lower()
        
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    rng = np.random.default_rng(seed)
    
    if model == "normal":
        mu_hat, sigma_hat = mean_val, max(1e-4, std_val)
        count, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.55,
                                 color='#3B82F6', edgecolor='#1D4ED8', label=f'Actual Dataset: {column_name} (N={n})')
        x_grid = np.linspace(min(bins[0], mu_hat - 3.5*sigma_hat), max(bins[-1], mu_hat + 3.5*sigma_hat), 400)
        theo_pdf = stats.norm.pdf(x_grid, loc=mu_hat, scale=sigma_hat)
        ax.plot(x_grid, theo_pdf, color='#DC2626', lw=2.5, label=f'Predicted Normal Probability Density Function ($\\hat{{\\mu}}={mu_hat:.2f}, \\hat{{\\sigma}}={sigma_hat:.2f}$)')
        
        # Monte-Carlo re-simulation from predicted parameters
        resim = rng.normal(loc=mu_hat, scale=sigma_hat, size=n)
        ax.hist(resim, bins=bins, density=True, histtype='step', color='#F59E0B', lw=2, label='Re-simulated Data')
        ax.set_ylabel("Probability Density")
        
    elif model == "binomial":
        n_trials = int(max(np.max(data), 1))
        p_hat = min(0.999, max(0.001, mean_val / n_trials))
        k_vals = np.arange(0, n_trials + 1)
        emp_prob = np.bincount(data.astype(int), minlength=n_trials + 1)[:n_trials + 1] / n
        theo_pmf = stats.binom.pmf(k_vals, n=n_trials, p=p_hat)
        
        bar_w = 0.4
        ax.bar(k_vals - bar_w/2, emp_prob, width=bar_w, alpha=0.7, color='#3B82F6', label=f'Actual Data ({column_name})')
        ax.bar(k_vals + bar_w/2, theo_pmf, width=bar_w, alpha=0.7, color='#DC2626', label=f'Predicted Binomial ($n={n_trials}, \\hat{{p}}={p_hat:.2f}$)')
        ax.set_ylabel("Probability")
        ax.set_xlabel("Count of Successes (k)")
        
    elif model == "poisson":
        lam_hat = max(1e-4, mean_val)
        max_k = int(max(np.max(data), lam_hat + 3.5 * np.sqrt(lam_hat)))
        k_vals = np.arange(0, max_k + 1)
        emp_prob = np.bincount(data.astype(int), minlength=max_k + 1)[:max_k + 1] / n
        theo_pmf = stats.poisson.pmf(k_vals, mu=lam_hat)
        
        bar_w = 0.4
        ax.bar(k_vals - bar_w/2, emp_prob, width=bar_w, alpha=0.7, color='#3B82F6', label=f'Actual Data ({column_name})')
        ax.bar(k_vals + bar_w/2, theo_pmf, width=bar_w, alpha=0.7, color='#DC2626', label=f'Predicted Poisson ($\\hat{{\\lambda}}={lam_hat:.2f}$)')
        ax.set_ylabel("Probability")
        ax.set_xlabel("Occurrences (k)")

    ax.set_title(f"Dataset Distribution Prediction & Fit\n(Predicted Model: {model.capitalize()})", fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved dataset prediction plot to: {save_path}")
    return fig


if __name__ == '__main__':
    print("=" * 65)
    print("Probability Distribution Simulation Suite (Built-in Functions)")
    print("=" * 65)
    
    # Output file paths
    norm_file = os.path.join(OUTPUT_DIR, 'normal_distribution.png')
    binom_file = os.path.join(OUTPUT_DIR, 'binomial_distribution.png')
    poiss_file = os.path.join(OUTPUT_DIR, 'poisson_distribution.png')
    pred_plot = os.path.join(OUTPUT_DIR, 'dataset_prediction_plot.png')
    sample_csv = os.path.join(OUTPUT_DIR, 'sample_data.csv')
    
    # 1. Simulate & Plot Normal Distribution
    print("\n1. Simulating Normal Distribution (mu=0.0, sigma=1.0, N=10,000)...")
    fig_norm, _, stats_norm = simulate_normal(mu=0.0, sigma=1.0, size=10000, save_path=norm_file)
    plt.close(fig_norm)
    print(f"   -> Mean: Sample = {stats_norm['sample_mean']:.4f} vs Theo = {stats_norm['theoretical_mean']:.4f}")
    print(f"   -> Var:  Sample = {stats_norm['sample_var']:.4f} vs Theo = {stats_norm['theoretical_var']:.4f}")
    
    # 2. Simulate & Plot Binomial Distribution
    print("\n2. Simulating Binomial Distribution (n=20, p=0.5, N=10,000)...")
    fig_binom, _, stats_binom = simulate_binomial(n=20, p=0.5, size=10000, save_path=binom_file)
    plt.close(fig_binom)
    print(f"   -> Mean: Sample = {stats_binom['sample_mean']:.4f} vs Theo = {stats_binom['theoretical_mean']:.4f}")
    print(f"   -> Var:  Sample = {stats_binom['sample_var']:.4f} vs Theo = {stats_binom['theoretical_var']:.4f}")
    
    # 3. Simulate & Plot Poisson Distribution
    print("\n3. Simulating Poisson Distribution (lambda=4.0, N=10,000)...")
    fig_poiss, _, stats_poiss = simulate_poisson(lam=4.0, size=10000, save_path=poiss_file)
    plt.close(fig_poiss)
    print(f"   -> Mean: Sample = {stats_poiss['sample_mean']:.4f} vs Theo = {stats_poiss['theoretical_mean']:.4f}")
    print(f"   -> Var:  Sample = {stats_poiss['sample_var']:.4f} vs Theo = {stats_poiss['theoretical_var']:.4f}")
    
    # 4. Fit & Predict from Sample Dataset (if available)
    if os.path.exists(sample_csv):
        print(f"\n4. Fitting and predicting from dataset: {sample_csv}")
        fig_pred = fit_and_predict_dataset(sample_csv, column_name="exam_score", save_path=pred_plot)
        plt.close(fig_pred)
        
    print("\n" + "=" * 65)
    print("All simulations completed successfully! Plots saved to disk.")
    print("=" * 65)
