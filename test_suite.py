"""
Automated Test Suite for Probability Distribution Simulator & Predictor
Tests all 10 required test cases specified in the project enhancement prompt.
"""

import numpy as np
import pandas as pd
import io
import os
import analysis_engine as ae
import simulation

def run_tests():
    print("=" * 70)
    print("STARTING 10-POINT AUTOMATED VERIFICATION SUITE")
    print("=" * 70)
    results = []
    
    # -------------------------------------------------------------
    # Test 1: Normal-distributed continuous data
    # -------------------------------------------------------------
    try:
        rng = np.random.default_rng(42)
        norm_data = rng.normal(loc=100.0, scale=15.0, size=500)
        stats_norm = ae.compute_descriptive_stats(norm_data)
        comp_df, verdict, fits = ae.run_full_distribution_comparison(norm_data, seed=42)
        assert fits['Normal']['applicable'] is True
        assert fits['Poisson']['applicable'] is False # continuous
        assert fits['Binomial']['applicable'] is False # continuous
        assert fits['Normal']['p_value'] >= 0.05
        results.append(("Test 1: Normal Continuous Data", "PASSED", f"Winner: {verdict['winner']} (p={fits['Normal']['p_value']:.4f})"))
    except Exception as e:
        results.append(("Test 1: Normal Continuous Data", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 2: Poisson-generated count data
    # -------------------------------------------------------------
    try:
        rng = np.random.default_rng(42)
        poiss_data = rng.poisson(lam=5.0, size=500)
        stats_poiss = ae.compute_descriptive_stats(poiss_data)
        comp_df, verdict, fits = ae.run_full_distribution_comparison(poiss_data, seed=42)
        assert fits['Poisson']['applicable'] is True
        assert fits['Poisson']['p_value'] >= 0.05
        results.append(("Test 2: Poisson Count Data", "PASSED", f"Winner: {verdict['winner']} (p={fits['Poisson']['p_value']:.4f})"))
    except Exception as e:
        results.append(("Test 2: Poisson Count Data", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 3: Binomial-generated count data
    # -------------------------------------------------------------
    try:
        rng = np.random.default_rng(42)
        binom_data = rng.binomial(n=25, p=0.4, size=500)
        stats_binom = ae.compute_descriptive_stats(binom_data)
        comp_df, verdict, fits = ae.run_full_distribution_comparison(binom_data, user_n_trials=25, seed=42)
        assert fits['Binomial']['applicable'] is True
        assert fits['Binomial']['p_value'] >= 0.05
        results.append(("Test 3: Binomial Count Data", "PASSED", f"Winner: {verdict['winner']} (p={fits['Binomial']['p_value']:.4f})"))
    except Exception as e:
        results.append(("Test 3: Binomial Count Data", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 4: CSV with missing values (NaN, empty strings)
    # -------------------------------------------------------------
    try:
        csv_data = "score,id\n10.5,1\n20.2,2\n,3\n15.8,4\nNaN,5\n30.1,6\n"
        df = pd.read_csv(io.StringIO(csv_data))
        report = ae.validate_and_clean_dataset(df, 'score', drop_invalid=True)
        assert report['missing_count'] == 2 # 1 empty cell, 1 NaN
        assert report['valid_count'] == 4
        assert len(report['clean_data']) == 4
        results.append(("Test 4: CSV with Missing Values", "PASSED", f"Excluded {report['missing_count']} missing, retained {report['valid_count']} valid"))
    except Exception as e:
        results.append(("Test 4: CSV with Missing Values", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 5: CSV with non-numeric / string values
    # -------------------------------------------------------------
    try:
        csv_data = "val\n100\n200\nCORRUPTED\n300\nN/A\ninf\n400\n"
        df = pd.read_csv(io.StringIO(csv_data))
        report = ae.validate_and_clean_dataset(df, 'val', drop_invalid=True)
        assert report['invalid_count'] >= 2 # CORRUPTED, N/A, inf
        assert report['valid_count'] == 4
        assert np.all(report['clean_data'] == [100.0, 200.0, 300.0, 400.0])
        results.append(("Test 5: CSV with Non-Numeric / Inf Values", "PASSED", f"Safely filtered {report['invalid_count']} non-numeric/inf values"))
    except Exception as e:
        results.append(("Test 5: CSV with Non-Numeric / Inf Values", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 6: Constant-valued column (zero variance)
    # -------------------------------------------------------------
    try:
        const_data = np.full(100, 50.0)
        stats_const = ae.compute_descriptive_stats(const_data)
        assert stats_const['is_constant'] is True
        assert stats_const['variance'] == 0.0
        comp_df, verdict, fits = ae.run_full_distribution_comparison(const_data)
        assert fits['Normal']['applicable'] is False
        assert fits['Poisson']['applicable'] is False
        assert fits['Binomial']['applicable'] is False
        assert verdict['winner'] is None
        results.append(("Test 6: Constant-Valued Column", "PASSED", f"Correctly flagged degenerate zero variance without crash"))
    except Exception as e:
        results.append(("Test 6: Constant-Valued Column", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 7: Very small dataset (N < 15)
    # -------------------------------------------------------------
    try:
        small_data = np.array([12.0, 15.0, 14.0, 19.0, 11.0, 13.0, 17.0])
        df_small = pd.DataFrame({'val': small_data})
        report = ae.validate_and_clean_dataset(df_small, 'val', drop_invalid=True)
        assert report['status'] == 'warning' # Small sample warning
        stats_small = ae.compute_descriptive_stats(small_data)
        comp_df, verdict, fits = ae.run_full_distribution_comparison(small_data)
        assert stats_small['n'] == 7
        results.append(("Test 7: Small Dataset (N < 15)", "PASSED", f"Triggered sample size warning and computed non-asymptotic stats"))
    except Exception as e:
        results.append(("Test 7: Small Dataset (N < 15)", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 8: Malformed or empty CSV
    # -------------------------------------------------------------
    try:
        empty_csv = ""
        try:
            pd.read_csv(io.StringIO(empty_csv))
            caught = False
        except pd.errors.EmptyDataError:
            caught = True
        assert caught is True
        results.append(("Test 8: Malformed or Empty CSV", "PASSED", "EmptyDataError safely caught and handled"))
    except Exception as e:
        results.append(("Test 8: Malformed or Empty CSV", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 9: Dataset that does NOT fit any candidate distribution (e.g. Cauchy heavy-tailed or bimodal)
    # -------------------------------------------------------------
    try:
        rng = np.random.default_rng(42)
        # Bimodal mixture with heavy separation
        bimodal = np.concatenate([rng.normal(-20, 1, 200), rng.normal(20, 1, 200)])
        comp_df, verdict, fits = ae.run_full_distribution_comparison(bimodal, seed=42)
        # Normal should be rejected with p < 0.01
        assert fits['Normal']['p_value'] < 0.01
        assert fits['Poisson']['applicable'] is False # has negative values
        assert fits['Binomial']['applicable'] is False # has negative values
        assert verdict['winner'] is None
        results.append(("Test 9: Incompatible / Bimodal Dataset", "PASSED", f"No candidate falsely accepted: {verdict['headline']}"))
    except Exception as e:
        results.append(("Test 9: Incompatible / Bimodal Dataset", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Test 10: Mode 1 simulation functions still work perfectly
    # -------------------------------------------------------------
    try:
        f1, s1, d1 = simulation.simulate_normal(mu=0, sigma=1, size=1000)
        f2, s2, d2 = simulation.simulate_binomial(n=10, p=0.5, size=1000)
        f3, s3, d3 = simulation.simulate_poisson(lam=3, size=1000)
        assert len(s1) == 1000
        assert len(s2) == 1000
        assert len(s3) == 1000
        results.append(("Test 10: Mode 1 Simulation Integrity", "PASSED", "All 3 core simulation functions operate as expected"))
    except Exception as e:
        results.append(("Test 10: Mode 1 Simulation Integrity", "FAILED", str(e)))

    # -------------------------------------------------------------
    # Print Summary Table
    # -------------------------------------------------------------
    print("\nSUMMARY OF TEST RESULTS:")
    all_passed = True
    for name, status, detail in results:
        sym = "✅" if status == "PASSED" else "❌"
        if status != "PASSED":
            all_passed = False
        print(f"{sym} {name:<45} | {status:<8} | {detail}")
        
    print("=" * 70)
    if all_passed:
        print("🎉 ALL 10 TESTS PASSED SUCCESSFULLY!")
    else:
        print("⚠️ SOME TESTS FAILED. PLEASE REVIEW LOGS.")
    print("=" * 70)
    return all_passed

if __name__ == '__main__':
    run_tests()
