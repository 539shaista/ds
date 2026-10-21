"""
Analysis Engine: Data Validation, Descriptive Statistics, and Distribution Fitting
Statistical toolkit supporting Normal, Poisson, and Binomial distribution fitting
with parameter estimation, rigorous goodness-of-fit tests, and category pooling.
"""

import numpy as np
import pandas as pd
from scipy import stats
import io


def validate_and_clean_dataset(df: pd.DataFrame, selected_col: str, drop_invalid: bool = True):
    """
    Validate, inspect, and clean the selected column in a DataFrame.
    
    Parameters:
        df: Input DataFrame
        selected_col: Name of column to extract and analyze
        drop_invalid: If True, exclude missing/infinite/unparseable values
        
    Returns:
        dict: Inspection report containing:
            - 'clean_series': pd.Series of numeric values (or None if halted)
            - 'clean_data': np.ndarray of cleaned float values
            - 'total_rows': Total rows in dataframe
            - 'total_cols': Total columns in dataframe
            - 'valid_count': Number of valid numeric observations
            - 'missing_count': NaN / None / blank count
            - 'invalid_count': Non-numeric or infinite values count
            - 'duplicate_rows': Duplicate row count in dataset
            - 'selected_col': Selected column name
            - 'has_inf': Boolean flag for infinity
            - 'status': 'success', 'warning', or 'error'
            - 'message': Descriptive explanation
    """
    total_rows, total_cols = df.shape
    duplicate_rows = int(df.duplicated().sum())
    
    if selected_col not in df.columns:
        return {
            'clean_series': None,
            'clean_data': np.array([]),
            'total_rows': total_rows,
            'total_cols': total_cols,
            'valid_count': 0,
            'missing_count': 0,
            'invalid_count': 0,
            'duplicate_rows': duplicate_rows,
            'selected_col': selected_col,
            'has_inf': False,
            'status': 'error',
            'message': f"Column '{selected_col}' not found in dataset."
        }
        
    raw_col = df[selected_col]
    missing_count = int(raw_col.isna().sum())
    
    # Attempt numeric coercion
    converted = pd.to_numeric(raw_col, errors='coerce')
    
    # Check infinite values
    inf_mask = np.isinf(converted)
    has_inf = bool(inf_mask.any())
    inf_count = int(inf_mask.sum())
    
    # Non-numeric string entries that became NaN after coercion
    coerced_nan = converted.isna().sum()
    invalid_string_count = int(coerced_nan - missing_count)
    total_invalid = invalid_string_count + inf_count
    
    # Valid mask: numeric, finite, non-null
    valid_mask = converted.notna() & (~inf_mask)
    valid_count = int(valid_mask.sum())
    
    if not drop_invalid and total_invalid + missing_count > 0:
        return {
            'clean_series': None,
            'clean_data': np.array([]),
            'total_rows': total_rows,
            'total_cols': total_cols,
            'valid_count': valid_count,
            'missing_count': missing_count,
            'invalid_count': total_invalid,
            'duplicate_rows': duplicate_rows,
            'selected_col': selected_col,
            'has_inf': has_inf,
            'status': 'warning',
            'message': f"Dataset has {missing_count} missing and {total_invalid} invalid entries. Cleaning disabled by user."
        }
        
    clean_series = converted[valid_mask].astype(float)
    clean_data = clean_series.values
    
    status = 'success'
    if valid_count < 15:
        msg = f"Cleaned {valid_count} observations. Caution: Small sample size (N < 15) may yield unreliable distribution fits."
        status = 'warning'
    elif total_invalid + missing_count > 0:
        msg = f"Successfully extracted {valid_count:,} observations (retained). Excluded {missing_count} missing and {total_invalid} invalid entries."
    else:
        msg = f"Successfully loaded {valid_count:,} valid observations with zero missing or invalid values."
        
    return {
        'clean_series': clean_series,
        'clean_data': clean_data,
        'total_rows': total_rows,
        'total_cols': total_cols,
        'valid_count': valid_count,
        'missing_count': missing_count,
        'invalid_count': total_invalid,
        'duplicate_rows': duplicate_rows,
        'selected_col': selected_col,
        'has_inf': has_inf,
        'status': status,
        'message': msg
    }


def compute_descriptive_stats(data: np.ndarray) -> dict:
    """
    Calculate comprehensive descriptive statistics for numeric observations.
    Handles constant values, zero variance, and small samples safely.
    """
    n = len(data)
    if n == 0:
        return {}
        
    mean_val = float(np.mean(data))
    median_val = float(np.median(data))
    min_val = float(np.min(data))
    max_val = float(np.max(data))
    range_val = float(max_val - min_val)
    
    var_val = float(np.var(data, ddof=1)) if n > 1 else 0.0
    std_val = float(np.std(data, ddof=1)) if n > 1 else 0.0
    
    q1 = float(np.percentile(data, 25))
    q3 = float(np.percentile(data, 75))
    iqr = float(q3 - q1)
    
    p5 = float(np.percentile(data, 5))
    p95 = float(np.percentile(data, 95))
    
    # Skewness & Kurtosis
    if n >= 3 and std_val > 1e-12:
        skew_val = float(stats.skew(data, bias=False))
    else:
        skew_val = 0.0
        
    if n >= 4 and std_val > 1e-12:
        kurt_val = float(stats.kurtosis(data, fisher=True, bias=False)) # Normal = 0
    else:
        kurt_val = 0.0
        
    # Mode calculation
    mode_str = "N/A"
    try:
        rounded = np.round(data, decimals=4)
        vals, counts = np.unique(rounded, return_counts=True)
        max_count = np.max(counts)
        if max_count > 1 and max_count < n:
            modes = vals[counts == max_count]
            if len(modes) <= 3:
                mode_str = ", ".join(f"{m:.2f}".rstrip('0').rstrip('.') for m in modes)
            else:
                mode_str = f"Multimodal ({len(modes)} modes)"
        else:
            mode_str = "All distinct / Uniform"
    except Exception:
        mode_str = "N/A"
        
    is_constant = bool(range_val < 1e-12 or (n > 1 and var_val < 1e-12))
    
    return {
        'n': n,
        'mean': mean_val,
        'median': median_val,
        'mode': mode_str,
        'min': min_val,
        'max': max_val,
        'range': range_val,
        'variance': var_val,
        'std_dev': std_val,
        'q1': q1,
        'q3': q3,
        'iqr': iqr,
        'skewness': skew_val,
        'kurtosis': kurt_val,
        'p5': p5,
        'p95': p95,
        'is_constant': is_constant
    }


def check_distribution_compatibility(data: np.ndarray) -> dict:
    """
    Check mathematical compatibility of dataset against Normal, Poisson, and Binomial.
    
    Returns:
        dict with compatibility status and justification for each model.
    """
    n = len(data)
    if n < 3:
        return {
            'normal': {'compatible': False, 'reason': 'Sample size too small (N < 3).'},
            'poisson': {'compatible': False, 'reason': 'Sample size too small (N < 3).'},
            'binomial': {'compatible': False, 'reason': 'Sample size too small (N < 3).'}
        }
        
    std_val = float(np.std(data, ddof=1)) if n > 1 else 0.0
    if std_val < 1e-12:
        reason = "Data is constant with zero variance (degenerate distribution)."
        return {
            'normal': {'compatible': False, 'reason': reason},
            'poisson': {'compatible': False, 'reason': reason},
            'binomial': {'compatible': False, 'reason': reason}
        }
        
    is_integer = bool(np.all(np.isclose(data, np.round(data), atol=1e-5)))
    is_non_negative = bool(np.all(data >= -1e-8))
    
    # 1. Normal Compatibility
    norm_comp = True
    norm_reason = "Compatible: Continuous numerical observations with non-zero variance."
    
    # 2. Poisson Compatibility
    if not is_non_negative:
        poiss_comp = False
        poiss_reason = "Incompatible: Poisson distribution requires non-negative count values (dataset contains negative values)."
    elif not is_integer:
        poiss_comp = False
        poiss_reason = "Incompatible: Poisson distribution is strictly discrete (dataset contains fractional/continuous values)."
    else:
        poiss_comp = True
        mean_val = float(np.mean(data))
        var_val = float(np.var(data, ddof=1))
        ratio = var_val / (mean_val if mean_val != 0 else 1.0)
        poiss_reason = f"Compatible: Non-negative integer count data (Variance/Mean ratio = {ratio:.2f})."
        
    # 3. Binomial Compatibility
    if not is_non_negative:
        binom_comp = False
        binom_reason = "Incompatible: Binomial requires non-negative integer counts of successes."
    elif not is_integer:
        binom_comp = False
        binom_reason = "Incompatible: Binomial distribution is discrete (dataset contains continuous values)."
    else:
        binom_comp = True
        binom_reason = "Compatible: Discrete integer count values representing successes in bounded trials."
        
    return {
        'normal': {'compatible': norm_comp, 'reason': norm_reason},
        'poisson': {'compatible': poiss_comp, 'reason': poiss_reason},
        'binomial': {'compatible': binom_comp, 'reason': binom_reason}
    }


def lilliefors_normal_test(data: np.ndarray, n_boot: int = 1500, seed: int = 42) -> tuple:
    """
    Compute Lilliefors test for normality with estimated mean and variance.
    Uses parametric bootstrap under the null hypothesis to compute exact p-value.
    
    Returns:
        (D_statistic, p_value)
    """
    n = len(data)
    mu = float(np.mean(data))
    std = float(np.std(data, ddof=1))
    if std < 1e-12 or n < 4:
        return 0.0, 1.0
        
    sorted_data = np.sort(data)
    cdf_vals = stats.norm.cdf(sorted_data, loc=mu, scale=std)
    i_n = np.arange(1, n + 1) / n
    i_minus_1_n = np.arange(0, n) / n
    d_stat = float(max(np.max(i_n - cdf_vals), np.max(cdf_vals - i_minus_1_n)))
    
    # Fast vectorized parametric bootstrap under H0: N(0, 1)
    rng = np.random.default_rng(seed)
    sim_samples = rng.normal(0, 1, size=(n_boot, n))
    sim_mus = np.mean(sim_samples, axis=1, keepdims=True)
    sim_stds = np.std(sim_samples, axis=1, ddof=1, keepdims=True)
    sim_sorted = np.sort(sim_samples, axis=1)
    sim_cdfs = stats.norm.cdf(sim_sorted, loc=sim_mus, scale=sim_stds)
    sim_d = np.maximum(np.max(i_n - sim_cdfs, axis=1), np.max(sim_cdfs - i_minus_1_n, axis=1))
    p_val = float(np.mean(sim_d >= d_stat))
    
    return d_stat, p_val


def fit_normal_distribution(data: np.ndarray, seed: int = 42) -> dict:
    """
    Estimate parameters and evaluate goodness-of-fit for Normal Distribution.
    """
    comp = check_distribution_compatibility(data)
    if not comp['normal']['compatible']:
        return {
            'applicable': False,
            'model': 'Normal',
            'params': {},
            'param_str': 'N/A',
            'log_likelihood': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'test_name': 'Lilliefors / Shapiro-Wilk',
            'test_stat': np.nan,
            'p_value': np.nan,
            'interpretation': comp['normal']['reason']
        }
        
    n = len(data)
    mu_hat = float(np.mean(data))
    sigma_hat = float(np.std(data, ddof=1)) if n > 1 else 1e-4
    sigma_hat = max(sigma_hat, 1e-5)
    
    # Log-likelihood & AIC
    log_lik = float(np.sum(stats.norm.logpdf(data, loc=mu_hat, scale=sigma_hat)))
    k_params = 2
    aic = 2 * k_params - 2 * log_lik
    bic = k_params * np.log(n) - 2 * log_lik
    
    # Goodness-of-fit: Lilliefors test (KS adjusted for estimated params)
    d_stat, p_lilliefors = lilliefors_normal_test(data, n_boot=1500, seed=seed)
    
    # Also compute Shapiro-Wilk if sample size allows
    shapiro_stat, p_shapiro = None, None
    if 3 <= n <= 5000:
        try:
            sw_res = stats.shapiro(data)
            shapiro_stat = float(sw_res.statistic)
            p_shapiro = float(sw_res.pvalue)
        except Exception:
            pass
            
    # Interpretation
    p_eval = p_shapiro if p_shapiro is not None else p_lilliefors
    if p_eval >= 0.05:
        verdict = f"Plausible fit (p = {p_eval:.3f} >= 0.05). Fails to reject H0: Normal."
    elif p_eval >= 0.01:
        verdict = f"Marginal fit (p = {p_eval:.3f}). Slight departure from normality."
    else:
        verdict = f"Poor fit (p = {p_eval:.4f} < 0.01). Significant departure from normality."
        
    return {
        'applicable': True,
        'model': 'Normal',
        'params': {'mu': mu_hat, 'sigma': sigma_hat},
        'param_str': f"μ̂ = {mu_hat:.3f}, σ̂ = {sigma_hat:.3f}",
        'log_likelihood': log_lik,
        'aic': aic,
        'bic': bic,
        'test_name': 'Lilliefors / Shapiro-Wilk' if p_shapiro else 'Lilliefors Test (Parametric Bootstrap)',
        'test_stat': d_stat,
        'p_value': p_eval,
        'shapiro_p': p_shapiro,
        'lilliefors_p': p_lilliefors,
        'interpretation': verdict
    }


def fit_poisson_distribution(data: np.ndarray) -> dict:
    """
    Estimate parameter and evaluate goodness-of-fit for Poisson Distribution
    using Chi-square test with category pooling (Cochran's expected frequency >= 5).
    """
    comp = check_distribution_compatibility(data)
    if not comp['poisson']['compatible']:
        return {
            'applicable': False,
            'model': 'Poisson',
            'params': {},
            'param_str': 'N/A',
            'log_likelihood': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'test_name': 'Chi-Square Goodness-of-Fit',
            'test_stat': np.nan,
            'p_value': np.nan,
            'interpretation': comp['poisson']['reason']
        }
        
    int_data = np.round(data).astype(int)
    n = len(int_data)
    lam_hat = float(max(1e-4, np.mean(int_data)))
    
    # Log-likelihood & AIC
    log_lik = float(np.sum(stats.poisson.logpmf(int_data, mu=lam_hat)))
    k_params = 1
    aic = 2 * k_params - 2 * log_lik
    bic = k_params * np.log(n) - 2 * log_lik
    
    # Chi-square Goodness of Fit with tail accumulation and bin pooling
    counts = np.bincount(int_data)
    max_k = max(len(counts) - 1, int(lam_hat + 4 * np.sqrt(lam_hat)))
    obs_counts = np.zeros(max_k + 1, dtype=float)
    obs_counts[:len(counts)] = counts
    
    k_grid = np.arange(max_k)
    exp_probs = stats.poisson.pmf(k_grid, mu=lam_hat)
    tail_prob = max(0.0, 1.0 - float(np.sum(exp_probs)))
    exp_probs = np.append(exp_probs, tail_prob)
    exp_counts = n * exp_probs
    
    # Category pooling (ensure expected count >= 5.0 in each pooled bin)
    pooled_obs = []
    pooled_exp = []
    curr_o, curr_e = 0.0, 0.0
    for o, e in zip(obs_counts, exp_counts):
        curr_o += o
        curr_e += e
        if curr_e >= 5.0:
            pooled_obs.append(curr_o)
            pooled_exp.append(curr_e)
            curr_o, curr_e = 0.0, 0.0
    if curr_e > 0:
        if len(pooled_exp) > 0:
            pooled_obs[-1] += curr_o
            pooled_exp[-1] += curr_e
        else:
            pooled_obs.append(curr_o)
            pooled_exp.append(curr_e)
            
    df = len(pooled_exp) - 1 - 1 # 1 parameter estimated (lambda)
    if df >= 1:
        chi2_stat = float(np.sum((np.array(pooled_obs) - np.array(pooled_exp))**2 / np.array(pooled_exp)))
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df))
        if p_val >= 0.05:
            verdict = f"Plausible fit (χ² = {chi2_stat:.2f}, df = {df}, p = {p_val:.3f} >= 0.05). Fails to reject H0: Poisson."
        elif p_val >= 0.01:
            verdict = f"Marginal fit (χ² = {chi2_stat:.2f}, df = {df}, p = {p_val:.3f})."
        else:
            verdict = f"Poor fit (χ² = {chi2_stat:.2f}, df = {df}, p = {p_val:.4f} < 0.01). Significant deviation from Poisson."
    else:
        chi2_stat = np.nan
        p_val = np.nan
        verdict = "Sample spread too narrow: Insufficient degrees of freedom after category pooling."
        
    return {
        'applicable': True,
        'model': 'Poisson',
        'params': {'lambda': lam_hat},
        'param_str': f"λ̂ = {lam_hat:.3f}",
        'log_likelihood': log_lik,
        'aic': aic,
        'bic': bic,
        'test_name': f'Chi-Square Test (df={df})' if df >= 1 else 'Chi-Square Test (Low df)',
        'test_stat': chi2_stat,
        'p_value': p_val,
        'interpretation': verdict
    }


def fit_binomial_distribution(data: np.ndarray, user_n_trials: int = None) -> dict:
    """
    Estimate parameters and evaluate goodness-of-fit for Binomial Distribution
    using Chi-square test with category pooling (Cochran's expected frequency >= 5).
    """
    comp = check_distribution_compatibility(data)
    if not comp['binomial']['compatible']:
        return {
            'applicable': False,
            'model': 'Binomial',
            'params': {},
            'param_str': 'N/A',
            'log_likelihood': np.nan,
            'aic': np.nan,
            'bic': np.nan,
            'test_name': 'Chi-Square Goodness-of-Fit',
            'test_stat': np.nan,
            'p_value': np.nan,
            'interpretation': comp['binomial']['reason']
        }
        
    int_data = np.round(data).astype(int)
    n_samples = len(int_data)
    max_observed = int(np.max(int_data))
    
    # Determine number of trials n
    if user_n_trials is not None and user_n_trials >= max_observed:
        n_trials = int(user_n_trials)
    else:
        n_trials = int(max(max_observed, 1))
        
    mean_val = float(np.mean(int_data))
    p_hat = float(min(0.9999, max(0.0001, mean_val / n_trials)))
    
    # Log-likelihood & AIC
    log_lik = float(np.sum(stats.binom.logpmf(int_data, n=n_trials, p=p_hat)))
    k_params = 1 # p estimated with fixed/specified n
    aic = 2 * k_params - 2 * log_lik
    bic = k_params * np.log(n_samples) - 2 * log_lik
    
    # Chi-square Goodness of Fit
    k_vals = np.arange(0, n_trials + 1)
    obs_counts = np.bincount(int_data, minlength=n_trials + 1)[:n_trials + 1].astype(float)
    exp_counts = n_samples * stats.binom.pmf(k_vals, n=n_trials, p=p_hat)
    
    # Category pooling (Cochran: >= 5.0)
    pooled_obs = []
    pooled_exp = []
    curr_o, curr_e = 0.0, 0.0
    for o, e in zip(obs_counts, exp_counts):
        curr_o += o
        curr_e += e
        if curr_e >= 5.0:
            pooled_obs.append(curr_o)
            pooled_exp.append(curr_e)
            curr_o, curr_e = 0.0, 0.0
    if curr_e > 0:
        if len(pooled_exp) > 0:
            pooled_obs[-1] += curr_o
            pooled_exp[-1] += curr_e
        else:
            pooled_obs.append(curr_o)
            pooled_exp.append(curr_e)
            
    df = len(pooled_exp) - 1 - 1 # 1 parameter estimated (p)
    if df >= 1:
        chi2_stat = float(np.sum((np.array(pooled_obs) - np.array(pooled_exp))**2 / np.array(pooled_exp)))
        p_val = float(1.0 - stats.chi2.cdf(chi2_stat, df))
        if p_val >= 0.05:
            verdict = f"Plausible fit (χ² = {chi2_stat:.2f}, df = {df}, p = {p_val:.3f} >= 0.05). Fails to reject H0: Binomial."
        elif p_val >= 0.01:
            verdict = f"Marginal fit (χ² = {chi2_stat:.2f}, df = {df}, p = {p_val:.3f})."
        else:
            verdict = f"Poor fit (χ² = {chi2_stat:.2f}, df = {df}, p = {p_val:.4f} < 0.01). Significant deviation from Binomial."
    else:
        chi2_stat = np.nan
        p_val = np.nan
        verdict = "Insufficient degrees of freedom after category pooling."
        
    return {
        'applicable': True,
        'model': 'Binomial',
        'params': {'n': n_trials, 'p': p_hat},
        'param_str': f"n = {n_trials}, p̂ = {p_hat:.3f}",
        'log_likelihood': log_lik,
        'aic': aic,
        'bic': bic,
        'test_name': f'Chi-Square Test (df={df})' if df >= 1 else 'Chi-Square Test (Low df)',
        'test_stat': chi2_stat,
        'p_value': p_val,
        'interpretation': verdict
    }


def run_full_distribution_comparison(data: np.ndarray, user_n_trials: int = None, seed: int = 42) -> tuple:
    """
    Run statistical comparison across Normal, Poisson, and Binomial.
    
    Returns:
        (comparison_df, best_candidate_summary, detailed_results_dict)
    """
    fit_norm = fit_normal_distribution(data, seed=seed)
    fit_poiss = fit_poisson_distribution(data)
    fit_binom = fit_binomial_distribution(data, user_n_trials=user_n_trials)
    
    results = {
        'Normal': fit_norm,
        'Poisson': fit_poiss,
        'Binomial': fit_binom
    }
    
    table_rows = []
    for name, res in results.items():
        table_rows.append({
            'Distribution': name,
            'Applicable?': '✅ Yes' if res['applicable'] else '❌ Incompatible',
            'Estimated Parameters': res['param_str'],
            'Goodness-of-Fit Statistic': f"{res['test_stat']:.3f}" if not np.isnan(res.get('test_stat', np.nan)) else 'N/A',
            'P-value': f"{res['p_value']:.4f}" if not np.isnan(res.get('p_value', np.nan)) else 'N/A',
            'AIC': f"{res['aic']:.1f}" if not np.isnan(res.get('aic', np.nan)) else 'N/A',
            'Interpretation': res['interpretation']
        })
        
    comp_df = pd.DataFrame(table_rows)
    
    # Determine strongest candidate statistically
    compatible_fits = {k: v for k, v in results.items() if v['applicable'] and not np.isnan(v.get('p_value', np.nan))}
    
    if not compatible_fits:
        summary_verdict = {
            'winner': None,
            'headline': 'No Candidate Distribution Fits Appropriately',
            'explanation': 'The observations are either mathematically incompatible or exhibit severe statistical departures (all models rejected with p < 0.05).',
            'status': 'error'
        }
    else:
        # Check models that fail to be rejected at alpha = 0.05
        plausible = {k: v for k, v in compatible_fits.items() if v['p_value'] >= 0.05}
        
        if len(plausible) == 1:
            win_name, win_obj = list(plausible.items())[0]
            summary_verdict = {
                'winner': win_name,
                'headline': f"Strongest Statistically Supported Candidate: **{win_name}**",
                'explanation': f"{win_name} distribution is compatible and demonstrates solid goodness-of-fit (p = {win_obj['p_value']:.4f} >= 0.05, fails to reject H0).",
                'status': 'success'
            }
        elif len(plausible) > 1:
            # Discrete vs continuous caution: Compare within discrete or report multiple
            sorted_plausible = sorted(plausible.items(), key=lambda item: item[1]['p_value'], reverse=True)
            top_name, top_obj = sorted_plausible[0]
            summary_verdict = {
                'winner': top_name,
                'headline': f"Multiple Plausible Candidates: **{', '.join(plausible.keys())}**",
                'explanation': f"Both candidates show non-significant test statistics (p >= 0.05). {top_name} exhibits the highest p-value ({top_obj['p_value']:.4f}). Contextual domain knowledge should guide the final selection.",
                'status': 'info'
            }
        else:
            # All p < 0.05
            best_p_name, best_p_obj = max(compatible_fits.items(), key=lambda item: item[1]['p_value'])
            summary_verdict = {
                'winner': None,
                'headline': "Inconclusive / Departures Detected in All Tested Models",
                'explanation': f"All tested candidate distributions reject H0 at alpha = 0.05. Closest fit was {best_p_name} (p = {best_p_obj['p_value']:.4f}), but underlying data likely follows another distribution or mixture.",
                'status': 'warning'
            }
            
    return comp_df, summary_verdict, results


def generate_analysis_report_csv(stats_dict: dict, comp_df: pd.DataFrame, selected_col: str) -> str:
    """
    Produce a clean formatted CSV report of descriptive statistics and distribution comparisons.
    """
    output = io.StringIO()
    output.write(f"# Distribution Analysis & Goodness-of-Fit Report\n")
    output.write(f"# Column Analyzed: {selected_col}\n")
    output.write(f"# Sample Size (N): {stats_dict.get('n', 'N/A')}\n\n")
    
    output.write("## 1. Descriptive Statistics\n")
    stat_rows = [
        ("Observations (N)", stats_dict.get('n', '')),
        ("Mean", f"{stats_dict.get('mean', 0.0):.4f}"),
        ("Median", f"{stats_dict.get('median', 0.0):.4f}"),
        ("Mode", stats_dict.get('mode', '')),
        ("Minimum", f"{stats_dict.get('min', 0.0):.4f}"),
        ("Maximum", f"{stats_dict.get('max', 0.0):.4f}"),
        ("Range", f"{stats_dict.get('range', 0.0):.4f}"),
        ("Sample Variance (s^2)", f"{stats_dict.get('variance', 0.0):.4f}"),
        ("Sample Std Dev (s)", f"{stats_dict.get('std_dev', 0.0):.4f}"),
        ("First Quartile (Q1)", f"{stats_dict.get('q1', 0.0):.4f}"),
        ("Third Quartile (Q3)", f"{stats_dict.get('q3', 0.0):.4f}"),
        ("Interquartile Range (IQR)", f"{stats_dict.get('iqr', 0.0):.4f}"),
        ("Skewness", f"{stats_dict.get('skewness', 0.0):.4f}"),
        ("Kurtosis (Excess)", f"{stats_dict.get('kurtosis', 0.0):.4f}"),
        ("5th Percentile", f"{stats_dict.get('p5', 0.0):.4f}"),
        ("95th Percentile", f"{stats_dict.get('p95', 0.0):.4f}")
    ]
    pd.DataFrame(stat_rows, columns=["Metric", "Value"]).to_csv(output, index=False)
    
    output.write("\n## 2. Distribution Fit & Goodness-of-Fit Comparison\n")
    comp_df.to_csv(output, index=False)
    
    return output.getvalue()
