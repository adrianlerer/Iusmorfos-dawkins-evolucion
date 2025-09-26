#!/usr/bin/env python3
"""
Test suite for Iusmorfos qualitative coding validation
=====================================================

Validates that framework meets minimum scientific standards:
- Inter-coder reliability κ ≥ 0.75
- Significance vs random null p < 0.05

Author: Adrian Lerer
Date: September 30, 2025
"""

import pytest
import json
import pathlib
import sys
import os

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from qualitative_coding import main as run_coding_analysis
from null_model import main as run_null_analysis

class TestQualitativeCoding:
    """Test inter-coder reliability validation."""
    
    def test_kappa_calculation_runs(self):
        """Test that kappa calculation completes without error."""
        kappa, results = run_coding_analysis()
        assert isinstance(kappa, float)
        assert isinstance(results, dict)
    
    def test_kappa_threshold_met(self):
        """Test that inter-coder reliability meets minimum threshold."""
        # Run analysis if results don't exist
        results_file = pathlib.Path("results/kappa.json")
        if not results_file.exists():
            run_coding_analysis()
        
        # Load and validate results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        kappa = results["overall_kappa"]
        assert kappa >= 0.75, f"Inter-coder reliability too low: κ = {kappa:.3f} < 0.75"
    
    def test_kappa_results_structure(self):
        """Test that kappa results have expected structure."""
        results_file = pathlib.Path("results/kappa.json")
        if not results_file.exists():
            run_coding_analysis()
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Check required fields
        required_fields = [
            'overall_kappa', 'interpretation', 'threshold_passed', 
            'cases', 'validation_date', 'n_cases'
        ]
        for field in required_fields:
            assert field in results, f"Missing required field: {field}"
        
        # Check kappa is in valid range
        kappa = results["overall_kappa"]
        assert -1 <= kappa <= 1, f"Kappa out of valid range: {kappa}"
        
        # Check we have expected number of cases
        assert results["n_cases"] == 3, f"Expected 3 cases, got {results['n_cases']}"

class TestNullModel:
    """Test null hypothesis validation."""
    
    def test_null_analysis_runs(self):
        """Test that null model analysis completes without error."""
        p_value, stats = run_null_analysis()
        assert isinstance(p_value, float)
        assert isinstance(stats, dict)
    
    def test_significance_threshold_met(self):
        """Test that framework significantly outperforms random null."""
        # Run analysis if results don't exist
        results_file = pathlib.Path("results/null_test.json")
        if not results_file.exists():
            run_null_analysis()
        
        # Load and validate results
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        p_value = results["p_value_one_tailed"]
        assert p_value < 0.05, f"Framework failed significance test: p = {p_value:.4f} ≥ 0.05"
    
    def test_null_results_structure(self):
        """Test that null test results have expected structure."""
        results_file = pathlib.Path("results/null_test.json")
        if not results_file.exists():
            run_null_analysis()
        
        with open(results_file, 'r') as f:
            results = json.load(f)
        
        # Check required fields
        required_fields = [
            'observed_score', 'null_mean', 'null_std', 'effect_size_cohens_d',
            'p_value_one_tailed', 'p_value_two_tailed', 'significant_at_05'
        ]
        for field in required_fields:
            assert field in results, f"Missing required field: {field}"
        
        # Check p-values are in valid range
        p_one = results["p_value_one_tailed"]
        p_two = results["p_value_two_tailed"]
        assert 0 <= p_one <= 1, f"One-tailed p-value out of range: {p_one}"
        assert 0 <= p_two <= 1, f"Two-tailed p-value out of range: {p_two}"

class TestIntegration:
    """Integration tests across framework components."""
    
    def test_both_analyses_pass(self):
        """Test that both reliability and significance criteria are met."""
        # Run both analyses
        kappa, kappa_results = run_coding_analysis()
        p_value, null_results = run_null_analysis()
        
        # Both must pass thresholds
        assert kappa >= 0.75, f"Reliability failed: κ = {kappa:.3f}"
        assert p_value < 0.05, f"Significance failed: p = {p_value:.4f}"
        
        # Integration check: reliability and significance should be consistent
        # High reliability should correlate with significance
        if kappa > 0.8 and p_value < 0.01:
            print("✅ Strong framework validation: high reliability + high significance")
        elif kappa >= 0.75 and p_value < 0.05:
            print("✅ Adequate framework validation: acceptable reliability + significance")
        else:
            pytest.fail("Integration test failed: inconsistent reliability/significance results")
    
    def test_results_files_created(self):
        """Test that all expected output files are created."""
        # Run analyses to ensure files are created
        run_coding_analysis()
        run_null_analysis()
        
        expected_files = [
            "results/kappa.json",
            "results/null_test.json",
            "paper/historical_cases.csv"
        ]
        
        for file_path in expected_files:
            file_obj = pathlib.Path(file_path)
            assert file_obj.exists(), f"Expected output file missing: {file_path}"
            assert file_obj.stat().st_size > 0, f"Output file is empty: {file_path}"

class TestDataQuality:
    """Test quality and consistency of underlying data."""
    
    def test_historical_cases_format(self):
        """Test that historical cases CSV has correct format."""
        csv_file = pathlib.Path("paper/historical_cases.csv")
        
        # Ensure file exists
        if not csv_file.exists():
            run_null_analysis()  # This creates the CSV
        
        import pandas as pd
        df = pd.read_csv(csv_file)
        
        # Check structure
        expected_cols = ['case', 'year', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8', 'D9', 'source']
        assert list(df.columns) == expected_cols, f"Unexpected columns: {list(df.columns)}"
        
        # Check data ranges
        dimension_cols = [f'D{i}' for i in range(1, 10)]
        for col in dimension_cols:
            assert df[col].min() >= 1, f"Dimension {col} has values < 1"
            assert df[col].max() <= 5, f"Dimension {col} has values > 5"
        
        # Check we have expected cases
        expected_cases = {'1981_crisis', '1994_reform', '2001_pesification'}
        actual_cases = set(df['case'])
        assert actual_cases == expected_cases, f"Case mismatch: {actual_cases} vs {expected_cases}"

if __name__ == "__main__":
    # Run tests directly if script is executed
    pytest.main([__file__, "-v", "--tb=short"])