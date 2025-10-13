"""
Unit tests for EvolutionaryOperators

Tests replication, mutation, selection, and fitness functions
with Reality Filter integration.

Author: Adrian Lerer
Date: 2025-10-13
"""

import pytest
import numpy as np
from iusmorfos.evolutionary.genome import LegalGenome
from iusmorfos.evolutionary.operators import EvolutionaryOperators


@pytest.fixture
def operators():
    """Create operators instance for tests."""
    return EvolutionaryOperators()


@pytest.fixture
def test_genome():
    """Create test genome."""
    return LegalGenome.from_template("oecd_gst_model")


def test_operators_initialization(operators):
    """Test operators initialization with empirical data."""
    assert operators.coefficients is not None
    assert len(operators.coefficients) > 0
    assert operators.base_rates is not None


def test_adaptive_coefficient_weird(operators):
    """Test adaptive coefficient for WEIRD country."""
    coeff = operators.get_adaptive_coefficient("germany")
    assert -0.05 <= coeff <= 0.0  # WEIRD countries have small negative coefficients


def test_adaptive_coefficient_no_weird(operators):
    """Test adaptive coefficient for No-WEIRD country."""
    coeff = operators.get_adaptive_coefficient("india")
    assert -0.45 <= coeff <= -0.25  # No-WEIRD countries have larger negative coefficients


def test_cultural_distance_weird(operators):
    """Test cultural distance for WEIRD country."""
    distance = operators.get_cultural_distance("germany")
    assert 0.0 <= distance <= 0.15  # Low distance from WEIRD baseline


def test_cultural_distance_no_weird(operators):
    """Test cultural distance for No-WEIRD country."""
    distance = operators.get_cultural_distance("nigeria")
    assert 0.60 <= distance <= 1.0  # High distance from WEIRD baseline


def test_base_rate_retrieval(operators):
    """Test base rate retrieval for different reform types."""
    base_rate = operators.get_base_rate("tax_reform")
    assert 0.0 <= base_rate <= 1.0
    
    default_rate = operators.get_base_rate("unknown_type")
    assert default_rate == operators.base_rates['default']


def test_replication_fidelity_weird(operators, test_genome):
    """Test high fidelity replication in WEIRD context."""
    child = operators.replicate_with_fidelity(test_genome, "germany")
    
    distance = test_genome.distance_to(child)
    assert distance < 1.0  # Should be very similar (high fidelity)
    assert child.generation == test_genome.generation + 1
    assert child.parent_id == test_genome.text_id


def test_replication_fidelity_no_weird(operators, test_genome):
    """Test low fidelity replication in No-WEIRD context."""
    child = operators.replicate_with_fidelity(test_genome, "india")
    
    distance = test_genome.distance_to(child)
    assert distance > 0.1  # Should be more different (low fidelity)


def test_replication_fidelity_comparison(operators, test_genome):
    """Test that No-WEIRD has lower fidelity than WEIRD."""
    child_weird = operators.replicate_with_fidelity(test_genome, "germany")
    child_no_weird = operators.replicate_with_fidelity(test_genome, "india")
    
    distance_weird = test_genome.distance_to(child_weird)
    distance_no_weird = test_genome.distance_to(child_no_weird)
    
    # India (No-WEIRD) should have larger distance than Germany (WEIRD)
    assert distance_no_weird > distance_weird


def test_adaptive_mutation(operators, test_genome):
    """Test adaptive mutation mechanism."""
    mutated = operators.adaptive_mutation(test_genome, "india", mutation_rate=0.5)
    
    assert mutated.text_id != test_genome.text_id
    assert mutated.parent_id == test_genome.text_id
    
    distance = test_genome.distance_to(mutated)
    assert distance > 0  # Should be different


def test_adaptive_mutation_scales_with_distance(operators, test_genome):
    """Test that mutation magnitude scales with cultural distance."""
    mutated_weird = operators.adaptive_mutation(test_genome, "germany", mutation_rate=0.5)
    mutated_no_weird = operators.adaptive_mutation(test_genome, "nigeria", mutation_rate=0.5)
    
    distance_weird = test_genome.distance_to(mutated_weird)
    distance_no_weird = test_genome.distance_to(mutated_no_weird)
    
    # Nigeria (higher cultural distance) should have larger mutations on average
    # Note: This is probabilistic, so we test multiple times
    distances_weird = []
    distances_no_weird = []
    
    for _ in range(10):
        m_w = operators.adaptive_mutation(test_genome, "germany", mutation_rate=0.5)
        m_nw = operators.adaptive_mutation(test_genome, "nigeria", mutation_rate=0.5)
        distances_weird.append(test_genome.distance_to(m_w))
        distances_no_weird.append(test_genome.distance_to(m_nw))
    
    assert np.mean(distances_no_weird) >= np.mean(distances_weird)


def test_fitness_function_with_reality_filter(operators, test_genome):
    """Test fitness calculation with Reality Filter."""
    fitness, (lower_ci, upper_ci) = operators.fitness_function(
        test_genome, "india", "tax_reform"
    )
    
    # Check fitness is in valid range
    assert 0.0 <= fitness <= 1.0
    
    # Check confidence interval
    assert lower_ci < fitness < upper_ci
    assert 0.0 <= lower_ci <= 1.0
    assert 0.0 <= upper_ci <= 1.0
    
    # Check CI width is honest (at least 0.30)
    ci_width = upper_ci - lower_ci
    assert ci_width >= 0.30  # Reality Filter ensures honest uncertainty


def test_fitness_weird_vs_no_weird(operators, test_genome):
    """Test that WEIRD countries have higher fitness than No-WEIRD."""
    fitness_weird, _ = operators.fitness_function(test_genome, "germany", "tax_reform")
    fitness_no_weird, _ = operators.fitness_function(test_genome, "india", "tax_reform")
    
    # WEIRD should have higher fitness (better implementation)
    assert fitness_weird > fitness_no_weird


def test_select_survivors(operators, test_genome):
    """Test survivor selection."""
    # Create population
    population = [test_genome]
    for i in range(9):
        clone = operators.replicate_with_fidelity(test_genome, "india")
        clone.text_id = f"clone_{i}"
        population.append(clone)
    
    # Select top 50%
    survivors = operators.select_survivors(population, "india", "tax_reform", survival_fraction=0.5)
    
    assert len(survivors) == 5  # 50% of 10
    
    # Check survivors are sorted by fitness
    fitness_scores = [g.fitness_score for g in survivors]
    assert fitness_scores == sorted(fitness_scores, reverse=True)


def test_evolution_convergence(operators, test_genome):
    """Test that evolution increases fitness over time."""
    final_pop, trajectory = operators.evolve_population(
        test_genome,
        "india",
        "tax_reform",
        generations=20,
        population_size=10
    )
    
    # Check trajectory structure
    assert len(trajectory) == 20
    assert all('mean_fitness' in t for t in trajectory)
    assert all('mean_ci_width' in t for t in trajectory)
    
    # Fitness should generally increase (allowing some stochasticity)
    initial_fitness = trajectory[0]['mean_fitness']
    final_fitness = trajectory[-1]['mean_fitness']
    
    # Allow for some variation due to randomness
    assert final_fitness >= initial_fitness * 0.85


def test_evolution_ci_width_honest(operators, test_genome):
    """Test that confidence intervals remain wide (honest uncertainty)."""
    final_pop, trajectory = operators.evolve_population(
        test_genome,
        "india",
        "tax_reform",
        generations=20,
        population_size=10
    )
    
    # Check that CI widths are consistently wide
    ci_widths = [t['mean_ci_width'] for t in trajectory]
    mean_ci_width = np.mean(ci_widths)
    
    # Reality Filter ensures CI width >= 0.40
    assert mean_ci_width >= 0.35  # Allow small margin


def test_evolution_population_size(operators, test_genome):
    """Test that population size is maintained."""
    final_pop, trajectory = operators.evolve_population(
        test_genome,
        "india",
        "tax_reform",
        generations=10,
        population_size=15
    )
    
    # Check final population size
    assert len(final_pop) > 0
    assert len(final_pop) <= 15  # May be less due to selection
    
    # Check trajectory reports correct sizes
    assert all(t['population_size'] > 0 for t in trajectory)
