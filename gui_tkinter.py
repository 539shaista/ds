"""
Desktop GUI Interface for Probability Distribution Simulation (Tkinter + Matplotlib)
Run with: python3 gui_tkinter.py
"""

import tkinter as tk
from tkinter import ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy import stats

class DistributionApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Probability Distribution Simulator - Desktop GUI")
        self.geometry("1100x750")
        self.minsize(900, 600)
        
        # Style configuration
        self.style = ttk.Style(self)
        self.style.theme_use('clam')
        
        self._create_layout()
        self.update_plot()
        
    def _create_layout(self):
        # Left Control Panel
        ctrl_frame = ttk.LabelFrame(self, text=" Simulation Controls ", padding=15)
        ctrl_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10, pady=10)
        
        # Distribution Selection
        ttk.Label(ctrl_frame, text="Select Distribution:", font=('Arial', 10, 'bold')).pack(anchor=tk.W, pady=(0, 5))
        self.dist_var = tk.StringVar(value="Normal")
        dist_combo = ttk.Combobox(
            ctrl_frame, textvariable=self.dist_var,
            values=["Normal", "Binomial", "Poisson"],
            state="readonly"
        )
        dist_combo.pack(fill=tk.X, pady=(0, 15))
        dist_combo.bind("<<ComboboxSelected>>", self._on_dist_change)
        
        # Sample Size Control
        ttk.Label(ctrl_frame, text="Sample Size (N):", font=('Arial', 10, 'bold')).pack(anchor=tk.W)
        self.n_samples_var = tk.IntVar(value=5000)
        self.n_slider = ttk.Scale(ctrl_frame, from_=500, to=50000, variable=self.n_samples_var, command=lambda e: self._on_slider_change())
        self.n_slider.pack(fill=tk.X, pady=5)
        self.n_label = ttk.Label(ctrl_frame, text=f"N = {self.n_samples_var.get()}")
        self.n_label.pack(anchor=tk.W, pady=(0, 15))
        
        # Dynamic Parameter Frame
        self.param_container = ttk.LabelFrame(ctrl_frame, text=" Parameters ", padding=10)
        self.param_container.pack(fill=tk.X, pady=(0, 15))
        
        # Normal Parameters
        self.mu_var = tk.DoubleVar(value=0.0)
        self.sigma_var = tk.DoubleVar(value=1.5)
        
        # Binomial Parameters
        self.binom_n_var = tk.IntVar(value=20)
        self.binom_p_var = tk.DoubleVar(value=0.5)
        
        # Poisson Parameters
        self.lam_var = tk.DoubleVar(value=5.0)
        
        # Stats Display Frame
        self.stats_frame = ttk.LabelFrame(ctrl_frame, text=" Empirical vs Theoretical Stats ", padding=10)
        self.stats_frame.pack(fill=tk.X, pady=(0, 15))
        self.stats_text = tk.StringVar(value="Loading...")
        ttk.Label(self.stats_frame, textvariable=self.stats_text, justify=tk.LEFT, font=('Monospace', 9)).pack(anchor=tk.W)
        
        # Re-simulate Button
        btn = ttk.Button(ctrl_frame, text="🔄 Re-simulate (New Seed)", command=self.update_plot)
        btn.pack(fill=tk.X, pady=5)
        
        # Right Plot Area
        plot_frame = ttk.Frame(self)
        plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.fig, self.ax = plt.subplots(figsize=(7, 5), dpi=100)
        self.fig.patch.set_facecolor('#F8FAFC')
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self._build_param_ui()

    def _build_param_ui(self):
        for widget in self.param_container.winfo_children():
            widget.destroy()
            
        dist = self.dist_var.get()
        if dist == "Normal":
            ttk.Label(self.param_container, text="Mean (μ):").pack(anchor=tk.W)
            s_mu = ttk.Scale(self.param_container, from_=-20.0, to=20.0, variable=self.mu_var, command=lambda e: self._on_slider_change())
            s_mu.pack(fill=tk.X, pady=2)
            self.mu_label = ttk.Label(self.param_container, text=f"μ = {self.mu_var.get():.1f}")
            self.mu_label.pack(anchor=tk.W, pady=(0, 10))
            
            ttk.Label(self.param_container, text="Std Dev (σ):").pack(anchor=tk.W)
            s_sigma = ttk.Scale(self.param_container, from_=0.2, to=10.0, variable=self.sigma_var, command=lambda e: self._on_slider_change())
            s_sigma.pack(fill=tk.X, pady=2)
            self.sigma_label = ttk.Label(self.param_container, text=f"σ = {self.sigma_var.get():.1f}")
            self.sigma_label.pack(anchor=tk.W)
            
        elif dist == "Binomial":
            ttk.Label(self.param_container, text="Trials (n):").pack(anchor=tk.W)
            s_n = ttk.Scale(self.param_container, from_=1, to=80, variable=self.binom_n_var, command=lambda e: self._on_slider_change())
            s_n.pack(fill=tk.X, pady=2)
            self.n_binom_label = ttk.Label(self.param_container, text=f"n = {int(self.binom_n_var.get())}")
            self.n_binom_label.pack(anchor=tk.W, pady=(0, 10))
            
            ttk.Label(self.param_container, text="Prob of Success (p):").pack(anchor=tk.W)
            s_p = ttk.Scale(self.param_container, from_=0.01, to=0.99, variable=self.binom_p_var, command=lambda e: self._on_slider_change())
            s_p.pack(fill=tk.X, pady=2)
            self.p_label = ttk.Label(self.param_container, text=f"p = {self.binom_p_var.get():.2f}")
            self.p_label.pack(anchor=tk.W)
            
        elif dist == "Poisson":
            ttk.Label(self.param_container, text="Rate Parameter (λ):").pack(anchor=tk.W)
            s_lam = ttk.Scale(self.param_container, from_=0.5, to=25.0, variable=self.lam_var, command=lambda e: self._on_slider_change())
            s_lam.pack(fill=tk.X, pady=2)
            self.lam_label = ttk.Label(self.param_container, text=f"λ = {self.lam_var.get():.1f}")
            self.lam_label.pack(anchor=tk.W)

    def _on_dist_change(self, event=None):
        self._build_param_ui()
        self.update_plot()
        
    def _on_slider_change(self):
        N = int(self.n_samples_var.get())
        self.n_label.config(text=f"N = {N:,}")
        
        dist = self.dist_var.get()
        if dist == "Normal":
            self.mu_label.config(text=f"μ = {self.mu_var.get():.1f}")
            self.sigma_label.config(text=f"σ = {self.sigma_var.get():.1f}")
        elif dist == "Binomial":
            self.n_binom_label.config(text=f"n = {int(self.binom_n_var.get())}")
            self.p_label.config(text=f"p = {self.binom_p_var.get():.2f}")
        elif dist == "Poisson":
            self.lam_label.config(text=f"λ = {self.lam_var.get():.1f}")
            
        self.update_plot()

    def update_plot(self):
        self.ax.clear()
        N = int(self.n_samples_var.get())
        dist = self.dist_var.get()
        
        if dist == "Normal":
            mu = float(self.mu_var.get())
            sigma = float(self.sigma_var.get())
            samples = np.random.normal(loc=mu, scale=sigma, size=N)
            
            _, bins, _ = self.ax.hist(samples, bins=35, density=True, alpha=0.6,
                                     color='#3B82F6', edgecolor='#1D4ED8', label=f'Simulated (N={N:,})')
            x_vals = np.linspace(bins[0], bins[-1], 300)
            theo_pdf = stats.norm.pdf(x_vals, loc=mu, scale=sigma)
            self.ax.plot(x_vals, theo_pdf, color='#DC2626', lw=2.5, label='Theoretical PDF')
            self.ax.set_title(f"Normal Distribution: μ={mu:.1f}, σ={sigma:.1f}")
            self.ax.set_xlabel("x")
            self.ax.set_ylabel("Density")
            
            theo_mean, theo_var = mu, sigma**2
            emp_mean, emp_var = np.mean(samples), np.var(samples, ddof=1)
            
        elif dist == "Binomial":
            n = int(self.binom_n_var.get())
            p = float(self.binom_p_var.get())
            samples = np.random.binomial(n=n, p=p, size=N)
            
            k_vals = np.arange(0, n + 1)
            counts = np.bincount(samples, minlength=n + 1)[:n + 1]
            emp_prob = counts / N
            theo_pmf = stats.binom.pmf(k_vals, n=n, p=p)
            
            width = 0.4
            self.ax.bar(k_vals - width/2, emp_prob, width=width, alpha=0.7, color='#10B981', label=f'Simulated (N={N:,})')
            self.ax.bar(k_vals + width/2, theo_pmf, width=width, alpha=0.7, color='#F59E0B', label='Theoretical PMF')
            self.ax.set_title(f"Binomial Distribution: n={n}, p={p:.2f}")
            self.ax.set_xlabel("Successes (k)")
            self.ax.set_ylabel("Probability")
            self.ax.set_xticks(k_vals[::max(1, len(k_vals)//15)])
            
            theo_mean, theo_var = n * p, n * p * (1 - p)
            emp_mean, emp_var = np.mean(samples), np.var(samples, ddof=1)
            
        elif dist == "Poisson":
            lam = float(self.lam_var.get())
            samples = np.random.poisson(lam=lam, size=N)
            
            max_k = max(int(np.max(samples)), int(lam + 3.5 * np.sqrt(lam)))
            k_vals = np.arange(0, max_k + 1)
            counts = np.bincount(samples, minlength=max_k + 1)[:max_k + 1]
            emp_prob = counts / N
            theo_pmf = stats.poisson.pmf(k_vals, mu=lam)
            
            width = 0.4
            self.ax.bar(k_vals - width/2, emp_prob, width=width, alpha=0.7, color='#8B5CF6', label=f'Simulated (N={N:,})')
            self.ax.bar(k_vals + width/2, theo_pmf, width=width, alpha=0.7, color='#EC4899', label='Theoretical PMF')
            self.ax.set_title(f"Poisson Distribution: λ={lam:.1f}")
            self.ax.set_xlabel("Events (k)")
            self.ax.set_ylabel("Probability")
            self.ax.set_xticks(k_vals[::max(1, len(k_vals)//15)])
            
            theo_mean, theo_var = lam, lam
            emp_mean, emp_var = np.mean(samples), np.var(samples, ddof=1)
            
        self.ax.grid(True, linestyle='--', alpha=0.4)
        self.ax.legend()
        self.fig.tight_layout()
        self.canvas.draw()
        
        self.stats_text.set(
            f"Theo Mean: {theo_mean:.4f}\n"
            f"Samp Mean: {emp_mean:.4f}\n\n"
            f"Theo Var:  {theo_var:.4f}\n"
            f"Samp Var:  {emp_var:.4f}"
        )

if __name__ == '__main__':
    app = DistributionApp()
    app.mainloop()
