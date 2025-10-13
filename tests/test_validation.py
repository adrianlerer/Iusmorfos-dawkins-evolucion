"""
Unit tests for RetrospectiveValidator

Tests validation methodology, convergence detection,
and calibration with Reality Filter.

Author: Adrian Lerer
Date: 2025-10-13
"""

import pytest
import numpy as np
from iusmorfos.integration.validation import RetrospectiveValidator, ValidationResult


@pytest.fixture
def validator():
    """Create validator instance for tests."""
    return RetrospectiveValidator(tolerance=0.10)


def test_validator_initialization(validator):
    """Test validator loads empirical cases."""
    assert len(validator.empirical_cases) > 0
    assert validator.tolerance == 0.10
    assert validator.operators is not None


def test_validate_single_case(validator):
    """Test single case validation."""
    # Use first available case
    case_id = validator.empirical_cases[0]['case_id']
    
    result = validator.validate_case(case_id, generations=10, verbose=False)
    
    assert isinstance(result, ValidationResult)
    assert result.case_id == case_id
    assert 0.0 <= result.simulated_fitness <= 1.0
    assert 0.0 <= result.observed_fitness <= 1.0
    assert result.validation in ['PASS', 'FAIL']
    assert result.error >= 0.0


def test_validation_result_structure(validator):
    """Test that validation result has all required fields."""
    case_id = validator.empirical_cases[0]['case_id']
    result = validator.validate_case(case_id, generations=10, verbose=False)
    
    # Check all fields present
    assert hasattr(result, 'case_id')
    assert hasattr(result, 'country')
    assert hasattr(result, 'reform_type')
    assert hasattr(result, 'observed_fitness')
    assert hasattr(result, 'simulated_fitness')
    assert hasattr(result, 'error')
    assert hasattr(result, 'validation')
    assert hasattr(result, 'trajectory')
    assert hasattr(result, 'confidence_interval')


def test_validation_confidence_intervals(validator):
    """Test that validation uses confidence intervals properly."""
    case_id = validator.empirical_cases[0]['case_id']
    result = validator.validate_case(case_id, generations=10, verbose=False)
    
    lower_ci, upper_ci = result.confidence_interval
    
    # Check CI bounds
    assert 0.0 <= lower_ci <= 1.0
    assert 0.0 <= upper_ci <= 1.0
    assert lower_ci < upper_ci
    
    # Check CI width is honest (Reality Filter)
    ci_width = upper_ci - lower_ci
    assert ci_width >= 0.30  # Honest uncertainty


def test_validation_within_ci_passes(validator):
    """Test that validation passes if observed falls within CI."""
    case_id = validator.empirical_cases[0]['case_id']
    result = validator.validate_case(case_id, generations=20, verbose=False)
    
    lower_ci, upper_ci = result.confidence_interval
    observed = result.observed_fitness
    
    # If observed is within CI, should pass
    if lower_ci <= observed <= upper_ci:
        assert result.validation == 'PASS'


def test_validate_all_cases(validator):
    """Test batch validation."""
    summary = validator.validate_all_cases(generations=10, verbose=False)
    
    assert summary['total_cases'] > 0
    assert 0.0 <= summary['pass_rate'] <= 1.0
    assert summary['mean_error'] >= 0.0
    assert summary['mean_ci_width'] >= 0.30  # Reality Filter ensures wide CIs
    assert len(summary['cases']) == summary['total_cases']


def test_validation_summary_structure(validator):
    """Test that validation summary has all required fields."""
    summary = validator.validate_all_cases(generations=10, verbose=False)
    
    required_keys = [
        'total_cases', 'passed', 'failed', 'pass_rate',
        'mean_error', 'mean_ci_width', 'tolerance', 'cases'
    ]
    
    for key in required_keys:
        assert key in summary


def test_convergence_detection(validator):
    """Test convergence generation detection."""
    # Create mock trajectory with clear convergence
    trajectory = [
        {'mean_fitness': 0.50},
        {'mean_fitness': 0.55},
        {'mean_fitness': 0.60},
        {'mean_fitness': 0.64},
        {'mean_fitness': 0.65},
        {'mean_fitness': 0.65},
        {'mean_fitness': 0.65},
        {'mean_fitness': 0.65},
        {'mean_fitness': 0.65},
        {'mean_fitness': 0.65},
    ]
    
    convergence_gen = validator._find_convergence(trajectory, window=3)
    assert convergence_gen >= 4  # Should detect stability around gen 5


def test_validation_result_serialization(validator):
    """Test that validation results can be serialized."""
    case_id = validator.empirical_cases[0]['case_id']
    result = validator.validate_case(case_id, generations=10, verbose=False)
    
    result_dict = result.to_dict()
    
    # Check serialization structure
    assert isinstance(result_dict, dict)
    assert 'case_id' in result_dict
    assert 'confidence_interval' in result_dict
    assert 'trajectory_summary' in result_dict


def test_calibration_basic(validator):
    """Test basic calibration functionality."""
    # Run with minimal iterations for speed
    result = validator.calibrate_parameters(
        target_pass_rate=0.70,
        max_iterations=3,
        verbose=False
    )
    
    assert 'optimal_mutation_rate' in result
    assert 'achieved_pass_rate' in result
    assert 0.0 <= result['optimal_mutation_rate'] <= 1.0
    assert 0.0 <= result['achieved_pass_rate'] <= 1.0


def test_save_results(validator, tmp_path):
    """Test saving validation results to file."""
    # Run minimal validation
    validator.validate_all_cases(generations=5, verbose=False)
    
    # Save to temporary directory
    validator.data_dir = tmp_path
    validator.save_results("test_results.json")
    
    # Check file was created
    result_file = tmp_path / "test_results.json"
    assert result_file.exists()


def test_reality_filter_prevents_overconfidence(validator):
    """Test that Reality Filter prevents overconfident predictions."""
    case_id = validator.empirical_cases[0]['case_id']
    result = validator.validate_case(case_id, generations=10, verbose=False)
    
    # Check that prediction is not overconfident
    # (should be regressed toward base rate)
    ci_width = result.confidence_interval[1] - result.confidence_interval[0]
    
    # Reality Filter ensures wide CIs (no overconfidence)
    assert ci_width >= 0.30
    
    # Prediction should not be extreme (regressed toward base rate)
    assert 0.20 <= result.simulated_fitness <= 0.90


def test_multiple_validations_consistent(validator):
    """Test that multiple validations give consistent results."""
    case_id = validator.empirical_cases[0]['case_id']
    
    results = []
    for _ in range(3):
        result = validator.validate_case(case_id, generations=10, verbose=False)
        results.append(result.simulated_fitness)
    
    # Results should be similar (within stochastic variation)
    std_dev = np.std(results)
    assert std_dev < 0.15  # Reasonable variation
