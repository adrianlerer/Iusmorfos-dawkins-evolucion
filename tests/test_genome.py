"""
Unit tests for LegalGenome

Tests genome initialization, templates, distance calculations,
serialization, and integration with empirical cases.

Author: Adrian Lerer
Date: 2025-10-13
"""

import pytest
import numpy as np
from datetime import datetime
from iusmorfos.evolutionary.genome import LegalGenome


def test_genome_initialization():
    """Test basic genome creation."""
    context = {
        'jurisdiction': 'Test Country',
        'legal_family': 'civil_law',
        'enactment_date': datetime(2020, 1, 1),
        'cultural_indices': {'power_distance': 50}
    }
    
    genome = LegalGenome(
        text="Test legal text with articles and sections.",
        context=context,
        text_id="test_001"
    )
    
    assert genome.text_id == "test_001"
    assert genome.generation == 0
    assert genome.parent_id is None
    assert genome.features is not None
    assert len(genome.features) == 89


def test_genome_from_template():
    """Test template initialization."""
    genome = LegalGenome.from_template("oecd_gst_model")
    
    assert genome.features is not None
    assert len(genome.features) == 89
    assert "GST" in genome.text or "Tax" in genome.text
    assert genome.context['jurisdiction'] == 'OECD Model'


def test_all_templates():
    """Test all available templates."""
    templates = ["oecd_gst_model", "world_bank_regulatory", "constitutional_amendment"]
    
    for template in templates:
        genome = LegalGenome.from_template(template)
        assert genome.features is not None
        assert len(genome.features) == 89
        assert genome.text_id.startswith(template)


def test_genome_distance():
    """Test genetic distance calculation."""
    g1 = LegalGenome.from_template("oecd_gst_model", text_id="g1")
    g2 = LegalGenome.from_template("world_bank_regulatory", text_id="g2")
    
    distance = g1.distance_to(g2)
    assert distance > 0
    assert distance < 100  # Reasonable range for normalized features


def test_genome_cosine_similarity():
    """Test cosine similarity calculation."""
    g1 = LegalGenome.from_template("oecd_gst_model", text_id="g1")
    g2 = LegalGenome.from_template("oecd_gst_model", text_id="g2")
    
    similarity = g1.cosine_similarity(g2)
    assert 0.9 <= similarity <= 1.01  # Should be very similar (same template, allow tiny floating point error)


def test_genome_self_distance():
    """Test that genome has zero distance to itself."""
    genome = LegalGenome.from_template("oecd_gst_model")
    distance = genome.distance_to(genome)
    assert distance == 0.0


def test_genome_serialization():
    """Test to_dict / from_dict roundtrip."""
    genome = LegalGenome.from_template("oecd_gst_model")
    genome.fitness_score = 0.75
    genome.generation = 5
    genome.parent_id = "parent_001"
    
    serialized = genome.to_dict()
    reconstructed = LegalGenome.from_dict(serialized)
    
    assert reconstructed.text_id == genome.text_id
    assert reconstructed.fitness_score == genome.fitness_score
    assert reconstructed.generation == genome.generation
    assert reconstructed.parent_id == genome.parent_id
    np.testing.assert_array_almost_equal(reconstructed.features, genome.features)


def test_genome_from_vector():
    """Test genome reconstruction from feature vector."""
    features = np.random.rand(89)
    context = {'jurisdiction': 'Test', 'legal_family': 'civil_law'}
    
    genome = LegalGenome.from_vector(
        features=features,
        text_id="test_vec",
        context=context
    )
    
    assert genome.text_id == "test_vec"
    np.testing.assert_array_equal(genome.features, features)


def test_genome_feature_normalization():
    """Test that features are normalized to [0, 1]."""
    genome = LegalGenome.from_template("oecd_gst_model")
    
    assert np.all(genome.features >= 0.0)
    assert np.all(genome.features <= 1.0)


def test_genome_to_policy_genome():
    """Test conversion to RootFinder format."""
    genome = LegalGenome.from_template("oecd_gst_model")
    genome.fitness_score = 0.8
    
    policy_genome = genome.to_policy_genome()
    
    assert policy_genome['policy_id'] == genome.text_id
    assert policy_genome['generation'] == genome.generation
    assert policy_genome['metadata']['fitness_score'] == 0.8
    assert policy_genome['metadata']['feature_dim'] == 89


def test_genome_repr():
    """Test string representation."""
    genome = LegalGenome.from_template("oecd_gst_model")
    genome.fitness_score = 0.75
    
    repr_str = repr(genome)
    assert "LegalGenome" in repr_str
    assert "0.75" in repr_str or "0.750" in repr_str
