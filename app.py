"""
Streamlit Web Interface for Simulating, Plotting, and Predicting Distributions from Datasets
Run with: streamlit run app.py
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import pandas as pd
import os

# Page Configuration
st.set_page_config(
    page_title="Probability Distribution Simulator & Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">📊 Probability Distribution Simulator & Predictor</div>', unsafe_allow_html=True)
st.write("Simulate and plot **Normal**, **Binomial**, and **Poisson** distributions using built-in functions, or **upload a dataset to predict its distribution and generate fitted plots**.")

# Sidebar Configuration
st.sidebar.header("⚙️ Simulation Mode")

mode_choice = st.sidebar.selectbox(
    "Select Mode:",
    [
        "Simulate: Normal (Gaussian)",
        "Simulate: Binomial",
        "Simulate: Poisson",
        "Simulate: Compare All Three",
        "📂 Predict from Dataset (Upload CSV)"
    ]
)

# Helper function for metrics
def display_metrics(theo_mean, emp_mean, theo_var, emp_var):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Theoretical Mean", f"{theo_mean:.4f}")
    with col2:
        error_mean = abs(emp_mean - theo_mean) / (abs(theo_mean) if theo_mean != 0 else 1.0) * 100
        st.metric("Sample Mean", f"{emp_mean:.4f}", delta=f"{error_mean:.2f}% error", delta_color="inverse")
    with col3:
        st.metric("Theoretical Variance", f"{theo_var:.4f}")
    with col4:
        error_var = abs(emp_var - theo_var) / (abs(theo_var) if theo_var != 0 else 1.0) * 100
        st.metric("Sample Variance", f"{emp_var:.4f}", delta=f"{error_var:.2f}% error", delta_color="inverse")

# ==================== DATASET PREDICTION MODE ====================
if mode_choice == "📂 Predict from Dataset (Upload CSV)":
    st.subheader("📂 Dataset Distribution Prediction & Fitting")
    st.write("Upload a CSV file or load our built-in dataset to predict the underlying distribution, estimate optimal parameters, and generate comparison graphs.")

    col_up1, col_up2 = st.columns([2, 1])
    with col_up1:
        uploaded_file = st.file_uploader("Upload CSV File:", type=["csv", "txt"])
    with col_up2:
        load_sample = st.button("📊 Load Built-in 'sample_data.csv'")

    df = None
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif load_sample or "df_loaded" in st.session_state:
        st.session_state["df_loaded"] = True
        sample_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_data.csv")
        if os.path.exists(sample_path):
            df = pd.read_csv(sample_path)
        else:
            st.warning("sample_data.csv not found on disk.")

    if df is not None:
        st.success(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
        
        # Column selection
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols:
            st.error("No numeric columns found in the dataset.")
        else:
            selected_col = st.selectbox("Select numeric column to fit/predict:", numeric_cols)
            data = df[selected_col].dropna().values
            n_samples = len(data)

            # Summary stats
            sample_mean = np.mean(data)
            sample_var = np.var(data, ddof=1) if n_samples > 1 else 0
            sample_std = np.std(data, ddof=1) if n_samples > 1 else 0

            # Auto-prediction logic
            is_integer = np.all(np.floor(data) == data)
            is_non_neg = np.all(data >= 0)
            var_mean_ratio = sample_var / (sample_mean if sample_mean != 0 else 1.0)

            if is_integer and is_non_neg and np.max(data) <= 100 and var_mean_ratio < 1.0:
                recommended = "Binomial"
                rec_reason = f"Data consists of bounded integer counts with variance < mean (Var/Mean = {var_mean_ratio:.2f})."
            elif is_integer and is_non_neg and abs(var_mean_ratio - 1.0) < 0.6:
                recommended = "Poisson"
                rec_reason = f"Data consists of non-negative event counts with variance close to mean (Var/Mean = {var_mean_ratio:.2f} ≈ 1.0)."
            else:
                recommended = "Normal (Gaussian)"
                rec_reason = f"Data spans continuous or symmetric bell-shaped values around central mean."

            st.info(f"🤖 **Auto-Prediction Recommendation**: Best fit is **{recommended}**.\n*{rec_reason}*")

            # Model Selection
            model_fit = st.radio("Choose Distribution Model to Fit:", ["Normal (Gaussian)", "Binomial", "Poisson"], horizontal=True,
                                 index=["Normal (Gaussian)", "Binomial", "Poisson"].index(recommended))

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Dataset Sample Size", f"{n_samples:,}")
            col_m2.metric("Sample Mean", f"{sample_mean:.3f}")
            col_m3.metric("Sample Variance", f"{sample_var:.3f}")
            col_m4.metric("Sample Std Dev", f"{sample_std:.3f}")

            # Plotting & Fitting
            fig, ax = plt.subplots(figsize=(10, 5.5), dpi=120)

            if model_fit == "Normal (Gaussian)":
                mu_fit = sample_mean
                sigma_fit = max(1e-4, sample_std)
                st.write(f"**Fitted Parameters**: $\\hat{{\\mu}} = {mu_fit:.3f}$, $\\hat{{\\sigma}} = {sigma_fit:.3f}$")

                count, bins, _ = ax.hist(data, bins=30, density=True, alpha=0.55, color='#3B82F6', edgecolor='#1D4ED8', label=f'Actual Dataset (N={n_samples})')
                x_grid = np.linspace(min(bins[0], mu_fit - 3.5*sigma_fit), max(bins[-1], mu_fit + 3.5*sigma_fit), 400)
                theo_pdf = stats.norm.pdf(x_grid, loc=mu_fit, scale=sigma_fit)
                ax.plot(x_grid, theo_pdf, color='#DC2626', linewidth=2.5, label=f'Predicted Normal PDF $\\mathcal{{N}}({mu_fit:.2f}, {sigma_fit:.2f}^2)$')

                # Log-likelihood & AIC
                ll = np.sum(stats.norm.logpdf(data, loc=mu_fit, scale=sigma_fit))
                aic = 2 * 2 - 2 * ll
                st.caption(f"Goodness-of-Fit: Log-Likelihood = {ll:.1f} | AIC = {aic:.1f}")

            elif model_fit == "Binomial":
                n_trials = int(max(np.max(data), 1))
                n_trials_input = st.number_input("Estimated Number of Trials (n):", min_value=int(np.max(data)), value=n_trials)
                p_fit = min(0.999, max(0.001, sample_mean / n_trials_input))
                st.write(f"**Fitted Parameters**: $\\hat{{n}} = {n_trials_input}$, $\\hat{{p}} = {p_fit:.3f}$")

                k_vals = np.arange(0, n_trials_input + 1)
                counts = np.bincount(data.astype(int), minlength=n_trials_input + 1)[:n_trials_input + 1]
                emp_prob = counts / n_samples
                theo_pmf = stats.binom.pmf(k_vals, n=n_trials_input, p=p_fit)

                bar_w = 0.4
                ax.bar(k_vals - bar_w/2, emp_prob, width=bar_w, alpha=0.7, color='#3B82F6', edgecolor='#1D4ED8', label='Actual Dataset Freq')
                ax.bar(k_vals + bar_w/2, theo_pmf, width=bar_w, alpha=0.7, color='#DC2626', edgecolor='#991B1B', label=f'Predicted Binomial PMF $B({n_trials_input}, {p_fit:.2f})$')

            elif model_fit == "Poisson":
                lam_fit = max(1e-4, sample_mean)
                st.write(f"**Fitted Parameter**: $\\hat{{\\lambda}} = {lam_fit:.3f}$")

                max_k = int(max(np.max(data), lam_fit + 3.5 * np.sqrt(lam_fit)))
                k_vals = np.arange(0, max_k + 1)
                counts = np.bincount(data.astype(int), minlength=max_k + 1)[:max_k + 1]
                emp_prob = counts / n_samples
                theo_pmf = stats.poisson.pmf(k_vals, mu=lam_fit)

                bar_w = 0.4
                ax.bar(k_vals - bar_w/2, emp_prob, width=bar_w, alpha=0.7, color='#3B82F6', edgecolor='#1D4ED8', label='Actual Dataset Freq')
                ax.bar(k_vals + bar_w/2, theo_pmf, width=bar_w, alpha=0.7, color='#DC2626', edgecolor='#991B1B', label=f'Predicted Poisson PMF Poisson($\\lambda={lam_fit:.2f}$)')

            ax.set_title(f"Dataset Distribution vs. Predicted {model_fit} Model", fontsize=14, fontweight='bold')
            ax.set_xlabel(selected_col, fontsize=12)
            ax.set_ylabel("Probability / Density", fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.5)
            ax.legend()
            st.pyplot(fig)

            # Interactive Prediction Tool
            st.markdown("### 🎯 Predict New Observations")
            query_x = st.number_input(f"Predict for value ({selected_col}):", value=float(np.round(sample_mean, 1)))
            if model_fit == "Normal (Gaussian)":
                pred_density = stats.norm.pdf(query_x, loc=mu_fit, scale=sigma_fit)
                pred_cdf = stats.norm.cdf(query_x, loc=mu_fit, scale=sigma_fit)
                st.write(f"- Probability Density: **{pred_density:.4f}**")
                st.write(f"- Cumulative Probability $P(X \\le {query_x})$: **{pred_cdf * 100:.2f}%**")
                st.write(f"- Survival Probability $P(X > {query_x})$: **{(1 - pred_cdf) * 100:.2f}%**")
            elif model_fit == "Binomial":
                pred_point = stats.binom.pmf(int(query_x), n=n_trials_input, p=p_fit)
                pred_cdf = stats.binom.cdf(int(query_x), n=n_trials_input, p=p_fit)
                st.write(f"- Point Probability $P(X = {int(query_x)})$: **{pred_point * 100:.2f}%**")
                st.write(f"- Cumulative Probability $P(X \\le {int(query_x)})$: **{pred_cdf * 100:.2f}%**")
            elif model_fit == "Poisson":
                pred_point = stats.poisson.pmf(int(query_x), mu=lam_fit)
                pred_cdf = stats.poisson.cdf(int(query_x), mu=lam_fit)
                st.write(f"- Point Probability $P(X = {int(query_x)})$: **{pred_point * 100:.2f}%**")
                st.write(f"- Cumulative Probability $P(X \\le {int(query_x)})$: **{pred_cdf * 100:.2f}%**")

            with st.expander("🔍 View Raw Dataset Table"):
                st.dataframe(df, height=250)

# ==================== STANDARD SIMULATION MODES ====================
else:
    sample_size = st.sidebar.slider("Sample Size (N):", min_value=100, max_value=100000, value=5000, step=500)
    seed = st.sidebar.number_input("Random Seed (Optional):", min_value=0, max_value=999999, value=42)
    if seed:
        np.random.seed(seed)

    if mode_choice == "Simulate: Normal (Gaussian)":
        st.subheader("🔵 Normal (Gaussian) Distribution: Continuous")
        st.latex(r"f(x) = \frac{1}{\sigma \sqrt{2\pi}} \exp\left(-\frac{(x - \mu)^2}{2\sigma^2}\right)")
        
        col_param1, col_param2, col_param3 = st.columns(3)
        with col_param1:
            mu = st.slider("Mean (μ):", min_value=-50.0, max_value=50.0, value=0.0, step=0.5)
        with col_param2:
            sigma = st.slider("Standard Deviation (σ):", min_value=0.1, max_value=20.0, value=2.0, step=0.1)
        with col_param3:
            bins = st.slider("Histogram Bins:", min_value=10, max_value=100, value=40, step=5)
            
        samples = np.random.normal(loc=mu, scale=sigma, size=sample_size)
        display_metrics(mu, np.mean(samples), sigma**2, np.var(samples, ddof=1))
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        count, bin_edges, _ = ax.hist(samples, bins=bins, density=True, alpha=0.55, color='#3B82F6', edgecolor='#1D4ED8', label=f'Empirical (N={sample_size:,})')
        x_grid = np.linspace(min(bin_edges[0], mu - 4*sigma), max(bin_edges[-1], mu + 4*sigma), 500)
        ax.plot(x_grid, stats.norm.pdf(x_grid, loc=mu, scale=sigma), color='#DC2626', linewidth=2.5, label='Theoretical PDF')
        ax.set_title(f'Normal Distribution: $\\mu={mu}$, $\\sigma={sigma}$')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend()
        st.pyplot(fig)

    elif mode_choice == "Simulate: Binomial":
        st.subheader("🟢 Binomial Distribution: Discrete")
        st.latex(r"P(X = k) = \binom{n}{k} p^k (1 - p)^{n - k}")
        
        col1, col2 = st.columns(2)
        with col1:
            n = st.slider("Number of Trials (n):", min_value=1, max_value=100, value=20, step=1)
        with col2:
            p = st.slider("Success Probability (p):", min_value=0.01, max_value=0.99, value=0.5, step=0.01)
            
        samples = np.random.binomial(n=n, p=p, size=sample_size)
        display_metrics(n * p, np.mean(samples), n * p * (1 - p), np.var(samples, ddof=1))
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        k_vals = np.arange(0, n + 1)
        counts = np.bincount(samples, minlength=n + 1)[:n + 1]
        width = 0.4
        ax.bar(k_vals - width/2, counts / sample_size, width=width, alpha=0.7, color='#10B981', label='Sample Relative Freq')
        ax.bar(k_vals + width/2, stats.binom.pmf(k_vals, n=n, p=p), width=width, alpha=0.7, color='#F59E0B', label='Theoretical PMF')
        ax.set_title(f'Binomial Distribution: $n={n}$, $p={p}$')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.legend()
        st.pyplot(fig)

    elif mode_choice == "Simulate: Poisson":
        st.subheader("🟣 Poisson Distribution: Discrete")
        st.latex(r"P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}")
        
        lam = st.slider("Rate Parameter (λ):", min_value=0.1, max_value=30.0, value=5.0, step=0.5)
        samples = np.random.poisson(lam=lam, size=sample_size)
        display_metrics(lam, np.mean(samples), lam, np.var(samples, ddof=1))
        
        fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
        max_k = max(int(np.max(samples)), int(lam + 4 * np.sqrt(lam)))
        k_vals = np.arange(0, max_k + 1)
        counts = np.bincount(samples, minlength=max_k + 1)[:max_k + 1]
        width = 0.4
        ax.bar(k_vals - width/2, counts / sample_size, width=width, alpha=0.7, color='#8B5CF6', label='Sample Relative Freq')
        ax.bar(k_vals + width/2, stats.poisson.pmf(k_vals, mu=lam), width=width, alpha=0.7, color='#EC4899', label='Theoretical PMF')
        ax.set_title(f'Poisson Distribution: $\\lambda={lam}$')
        ax.grid(True, linestyle='--', alpha=0.5, axis='y')
        ax.legend()
        st.pyplot(fig)

    else:
        st.subheader("📑 Comparative Simulation Overview")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=120)
        
        # 1. Normal
        ns = np.random.normal(loc=0, scale=1, size=sample_size)
        _, bins, _ = axes[0].hist(ns, bins=30, density=True, alpha=0.6, color='#3B82F6')
        xg = np.linspace(bins[0], bins[-1], 200)
        axes[0].plot(xg, stats.norm.pdf(xg, 0, 1), color='#DC2626', lw=2)
        axes[0].set_title('Normal (μ=0, σ=1)')
        
        # 2. Binomial
        bs = np.random.binomial(n=20, p=0.5, size=sample_size)
        kb = np.arange(0, 21)
        axes[1].bar(kb - 0.2, np.bincount(bs, minlength=21)[:21]/sample_size, width=0.4, color='#10B981')
        axes[1].bar(kb + 0.2, stats.binom.pmf(kb, 20, 0.5), width=0.4, color='#F59E0B')
        axes[1].set_title('Binomial (n=20, p=0.5)')
        
        # 3. Poisson
        ps = np.random.poisson(lam=4, size=sample_size)
        kp = np.arange(0, 13)
        axes[2].bar(kp - 0.2, np.bincount(ps, minlength=13)[:13]/sample_size, width=0.4, color='#8B5CF6')
        axes[2].bar(kp + 0.2, stats.poisson.pmf(kp, 4), width=0.4, color='#EC4899')
        axes[2].set_title('Poisson (λ=4)')
        
        st.pyplot(fig)
