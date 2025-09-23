#!/usr/bin/env python3
"""
External Validation Framework for Iusmorfos
==========================================

Cross-country validation system for testing the generalizability of the Iusmorfos
framework across different legal systems and cultural contexts.

Target Countries:
- Chile (CL): Civil law system, similar to Argentina
- South Africa (ZA): Mixed legal system (civil + common law)
- Sweden (SE): Civil law system, Nordic model
- India (IN): Common law system with customary elements

Following FAIR principles and reproducibility best practices.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import warnings

# Statistical libraries
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans

# Configuration
from config import get_config

warnings.filterwarnings('ignore')


class LegalSystem(Enum):
    """Legal system classification."""
    CIVIL_LAW = "civil_law"
    COMMON_LAW = "common_law"
    MIXED_SYSTEM = "mixed_system"
    CUSTOMARY_LAW = "customary_law"
    RELIGIOUS_LAW = "religious_law"


@dataclass
class CountryProfile:
    """Country profile for validation."""
    code: str
    name: str
    legal_system: LegalSystem
    gdp_per_capita: float
    population_millions: float
    governance_index: float  # 0-1 scale
    cultural_dimensions: Dict[str, float]  # Hofstede dimensions
    expected_patterns: Dict[str, Any]


class ExternalValidationFramework:
    """
    Comprehensive external validation framework for cross-country analysis.
    
    Features:
    - Multi-country dataset integration
    - Cultural adaptation mechanisms
    - Cross-system transferability testing
    - Pattern generalization validation
    - Statistical significance testing
    """
    
    def __init__(self):
        """Initialize external validation framework."""
        self.config = get_config()
        self.logger = logging.getLogger('iusmorfos.external_validation')
        
        self.country_profiles = self._initialize_country_profiles()
        self.validation_results = {}
        
        self.logger.info("🌍 External validation framework initialized")
    
    def _initialize_country_profiles(self) -> Dict[str, CountryProfile]:
        """Initialize country profiles for validation."""
        
        profiles = {
            'CL': CountryProfile(
                code='CL',
                name='Chile',
                legal_system=LegalSystem.CIVIL_LAW,
                gdp_per_capita=15346,  # USD 2022
                population_millions=19.6,
                governance_index=0.78,  # World Bank estimate
                cultural_dimensions={
                    'power_distance': 63,
                    'individualism': 23,
                    'masculinity': 28,
                    'uncertainty_avoidance': 86,
                    'long_term_orientation': 31
                },
                expected_patterns={
                    'reform_frequency': 'medium',
                    'adoption_speed': 'slow',
                    'complexity_preference': 'high',
                    'crisis_response': 'institutional'
                }
            ),
            'ZA': CountryProfile(
                code='ZA',
                name='South Africa',
                legal_system=LegalSystem.MIXED_SYSTEM,
                gdp_per_capita=6994,
                population_millions=60.4,
                governance_index=0.62,
                cultural_dimensions={
                    'power_distance': 49,
                    'individualism': 65,
                    'masculinity': 63,
                    'uncertainty_avoidance': 49,
                    'long_term_orientation': 34
                },
                expected_patterns={
                    'reform_frequency': 'high',
                    'adoption_speed': 'medium',
                    'complexity_preference': 'medium',
                    'crisis_response': 'adaptive'
                }
            ),
            'SE': CountryProfile(
                code='SE',
                name='Sweden',
                legal_system=LegalSystem.CIVIL_LAW,
                gdp_per_capita=54608,
                population_millions=10.5,
                governance_index=0.94,
                cultural_dimensions={
                    'power_distance': 31,
                    'individualism': 71,
                    'masculinity': 5,
                    'uncertainty_avoidance': 29,
                    'long_term_orientation': 53
                },
                expected_patterns={
                    'reform_frequency': 'low',
                    'adoption_speed': 'fast',
                    'complexity_preference': 'low',
                    'crisis_response': 'consensus'
                }
            ),
            'IN': CountryProfile(
                code='IN',
                name='India',
                legal_system=LegalSystem.COMMON_LAW,
                gdp_per_capita=2256,
                population_millions=1417.2,
                governance_index=0.58,
                cultural_dimensions={
                    'power_distance': 77,
                    'individualism': 48,
                    'masculinity': 56,
                    'uncertainty_avoidance': 40,
                    'long_term_orientation': 51
                },
                expected_patterns={
                    'reform_frequency': 'high',
                    'adoption_speed': 'slow',
                    'complexity_preference': 'very_high',
                    'crisis_response': 'hierarchical'
                }
            )
        }
        
        return profiles
    
    def generate_synthetic_country_data(self, 
                                      country_code: str, 
                                      n_innovations: int = 300) -> pd.DataFrame:
        """
        Generate synthetic legal innovation data for a specific country.
        
        This simulates realistic data based on the country's legal system,
        cultural dimensions, and expected patterns.
        """
        self.logger.info(f"🏗️ Generating synthetic data for {country_code} ({n_innovations} innovations)")
        
        if country_code not in self.country_profiles:
            raise ValueError(f"Country {country_code} not configured")
        
        profile = self.country_profiles[country_code]
        
        # Set seed for reproducible synthetic data
        np.random.seed(hash(country_code) % 2**32)
        
        # Generate temporal distribution (1990-2023)
        years = np.random.choice(
            range(1990, 2024), 
            size=n_innovations, 
            p=self._get_temporal_weights(profile)
        )
        
        # Generate reform types based on legal system
        reform_types = self._generate_reform_types(profile, n_innovations)
        
        # Generate complexity scores influenced by cultural dimensions
        complexity_scores = self._generate_complexity_scores(profile, n_innovations)
        
        # Generate adoption success rates
        adoption_rates = self._generate_adoption_rates(profile, n_innovations)
        
        # Generate citation networks (influenced by legal system connectivity)
        citations = self._generate_citation_networks(profile, n_innovations)
        
        # Create crisis indicators
        crisis_indicators = self._generate_crisis_patterns(profile, years)
        
        # Calculate fitness scores
        fitness_scores = self._calculate_country_fitness(
            complexity_scores, adoption_rates, citations, profile
        )
        
        # Create IusSpace coordinates (9-dimensional)
        iuspace_coords = self._generate_iuspace_coordinates(
            complexity_scores, adoption_rates, citations, profile, n_innovations
        )
        
        # Assemble dataset
        data = pd.DataFrame({
            'country': country_code,
            'country_name': profile.name,
            'year': years,
            'reform_type': reform_types,
            'complexity_score': complexity_scores,
            'adoption_success': adoption_rates,
            'citation_count': citations,
            'fitness_score': fitness_scores,
            'in_crisis': crisis_indicators,
            'legal_system': profile.legal_system.value,
            'gdp_per_capita': profile.gdp_per_capita,
            'governance_index': profile.governance_index
        })
        
        # Add IusSpace dimensions
        for i in range(9):
            data[f'iuspace_dim_{i+1}'] = iuspace_coords[:, i]
        
        return data
    
    def _get_temporal_weights(self, profile: CountryProfile) -> np.ndarray:
        """Generate temporal weights based on country development pattern."""
        years = np.array(range(1990, 2024))
        
        if profile.gdp_per_capita > 30000:  # Developed countries
            # Steady, slight decline in recent years
            weights = np.exp(-0.02 * (years - 1990))
        elif profile.gdp_per_capita > 10000:  # Middle-income
            # Peak in 2000s-2010s
            weights = np.exp(-0.5 * ((years - 2005) / 10)**2)
        else:  # Developing countries
            # Increasing reform activity
            weights = np.exp(0.03 * (years - 1990))
        
        return weights / weights.sum()
    
    def _generate_reform_types(self, profile: CountryProfile, n: int) -> np.ndarray:
        """Generate reform types based on legal system characteristics."""
        
        if profile.legal_system == LegalSystem.CIVIL_LAW:
            types = ['constitutional', 'civil', 'administrative', 'commercial', 'criminal']
            weights = [0.15, 0.30, 0.25, 0.20, 0.10]
        elif profile.legal_system == LegalSystem.COMMON_LAW:
            types = ['judicial', 'statutory', 'administrative', 'commercial', 'criminal']
            weights = [0.25, 0.20, 0.20, 0.25, 0.10]
        elif profile.legal_system == LegalSystem.MIXED_SYSTEM:
            types = ['constitutional', 'civil', 'common', 'customary', 'administrative']
            weights = [0.20, 0.25, 0.20, 0.15, 0.20]
        else:
            types = ['constitutional', 'civil', 'administrative', 'commercial', 'criminal']
            weights = [0.20, 0.20, 0.20, 0.20, 0.20]
        
        return np.random.choice(types, size=n, p=weights)
    
    def _generate_complexity_scores(self, profile: CountryProfile, n: int) -> np.ndarray:
        """Generate complexity scores influenced by cultural dimensions."""
        
        # Base complexity influenced by uncertainty avoidance
        base_complexity = 3.0 + (profile.cultural_dimensions['uncertainty_avoidance'] / 100) * 4.0
        
        # Variance influenced by power distance
        complexity_std = 1.0 + (profile.cultural_dimensions['power_distance'] / 100) * 1.5
        
        scores = np.random.gamma(
            shape=base_complexity,
            scale=complexity_std,
            size=n
        )
        
        # Ensure 1-10 range
        return np.clip(scores, 1.0, 10.0)
    
    def _generate_adoption_rates(self, profile: CountryProfile, n: int) -> np.ndarray:
        """Generate adoption success rates based on governance and culture."""
        
        # Base success rate from governance index
        base_success = profile.governance_index
        
        # Adjust for individualism (higher individualism = lower adoption in some contexts)
        individualism_effect = 1.0 - (profile.cultural_dimensions['individualism'] / 200)
        
        # Beta distribution parameters
        alpha = base_success * individualism_effect * 5 + 1
        beta = (1 - base_success * individualism_effect) * 5 + 1
        
        return np.random.beta(alpha, beta, size=n)
    
    def _generate_citation_networks(self, profile: CountryProfile, n: int) -> np.ndarray:
        """Generate citation counts following power-law distribution."""
        
        # Scale parameter influenced by legal system connectivity
        if profile.legal_system == LegalSystem.COMMON_LAW:
            scale = 2.5  # Higher connectivity in common law
        elif profile.legal_system == LegalSystem.MIXED_SYSTEM:
            scale = 2.0
        else:
            scale = 1.5
        
        # Adjust for economic development
        scale *= (1 + np.log10(profile.gdp_per_capita / 1000) / 10)
        
        # Generate power-law distributed citations
        citations = np.random.pareto(1.3, size=n) * scale + 1
        
        return np.round(citations).astype(int)
    
    def _generate_crisis_patterns(self, profile: CountryProfile, years: np.ndarray) -> np.ndarray:
        """Generate crisis indicators based on country profile."""
        
        # Define crisis periods for each country (simplified)
        crisis_periods = {
            'CL': [(2008, 2009), (2019, 2020)],  # Financial crisis, social unrest
            'ZA': [(1994, 1996), (2008, 2009), (2015, 2017), (2020, 2021)],  # Transition, financial, political, COVID
            'SE': [(2008, 2009), (2020, 2021)],  # Financial crisis, COVID
            'IN': [(1991, 1992), (2008, 2009), (2016, 2017), (2020, 2021)]  # Economic reform, financial, demonetization, COVID
        }
        
        country_crises = crisis_periods.get(profile.code, [])
        
        crisis_indicators = np.zeros(len(years), dtype=bool)
        
        for start_year, end_year in country_crises:
            crisis_mask = (years >= start_year) & (years <= end_year)
            crisis_indicators |= crisis_mask
        
        return crisis_indicators
    
    def _calculate_country_fitness(self, 
                                 complexity: np.ndarray,
                                 adoption: np.ndarray,
                                 citations: np.ndarray,
                                 profile: CountryProfile) -> np.ndarray:
        """Calculate fitness scores adjusted for country context."""
        
        # Normalize components
        complexity_norm = complexity / 10.0
        adoption_norm = adoption
        citation_norm = np.minimum(np.log1p(citations) / 10.0, 1.0)
        
        # Country-specific weighting based on legal system
        if profile.legal_system == LegalSystem.COMMON_LAW:
            # Emphasize precedent (citations) more
            weights = [0.25, 0.35, 0.40]
        elif profile.legal_system == LegalSystem.CIVIL_LAW:
            # Emphasize systematic complexity
            weights = [0.40, 0.35, 0.25]
        else:
            # Balanced approach
            weights = [0.33, 0.34, 0.33]
        
        fitness = (weights[0] * complexity_norm + 
                  weights[1] * adoption_norm + 
                  weights[2] * citation_norm)
        
        return fitness
    
    def _generate_iuspace_coordinates(self,
                                    complexity: np.ndarray,
                                    adoption: np.ndarray,
                                    citations: np.ndarray,
                                    profile: CountryProfile,
                                    n: int) -> np.ndarray:
        """Generate 9-dimensional IusSpace coordinates."""
        
        # Base dimensions from core metrics
        dim1 = complexity  # Complexity
        dim2 = adoption * 10  # Adoption (scaled to 1-10)
        dim3 = np.minimum(citations / 10, 10)  # Citation impact (capped at 10)
        
        # Additional dimensions influenced by cultural and institutional factors
        dim4 = np.random.gamma(  # Institutional stability
            shape=profile.governance_index * 5 + 1,
            scale=1.0,
            size=n
        ) + 1
        
        dim5 = np.random.beta(  # Enforcement efficiency
            alpha=profile.governance_index * 3 + 1,
            beta=(1 - profile.governance_index) * 3 + 1,
            size=n
        ) * 9 + 1
        
        dim6 = np.random.poisson(  # Network connectivity
            lam=np.log10(profile.gdp_per_capita / 1000) + 1,
            size=n
        ) + 1
        
        dim7 = np.random.exponential(  # Adaptability index
            scale=profile.cultural_dimensions['uncertainty_avoidance'] / 20 + 1,
            size=n
        ) + 1
        
        dim8 = np.random.normal(  # Reform velocity
            loc=5.0,
            scale=profile.cultural_dimensions['power_distance'] / 50,
            size=n
        )
        
        dim9 = np.random.gamma(  # Social acceptance
            shape=profile.cultural_dimensions['individualism'] / 20 + 1,
            scale=1.5,
            size=n
        ) + 1
        
        # Ensure all dimensions are in 1-10 range
        coords = np.column_stack([dim1, dim2, dim3, dim4, dim5, dim6, dim7, dim8, dim9])
        coords = np.clip(coords, 1.0, 10.0)
        
        return coords
    
    def validate_argentina_model_on_country(self, 
                                          country_code: str,
                                          argentina_model_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate Argentina-trained model on another country's data.
        
        Args:
            country_code: Target country code
            argentina_model_params: Parameters learned from Argentina data
        
        Returns:
            Validation results including transferability metrics
        """
        self.logger.info(f"🎯 Validating Argentina model on {country_code}")
        
        # Generate country data
        country_data = self.generate_synthetic_country_data(country_code, n_innovations=400)
        
        # Extract features and target
        feature_cols = [f'iuspace_dim_{i+1}' for i in range(9)]
        X = country_data[feature_cols].values
        y = country_data['fitness_score'].values
        
        # Apply Argentina model (simplified simulation)
        # In real implementation, this would use the actual trained model
        argentina_predictions = self._simulate_argentina_model_predictions(X, argentina_model_params)
        
        # Calculate validation metrics
        mse = mean_squared_error(y, argentina_predictions)
        r2 = r2_score(y, argentina_predictions)
        mae = mean_absolute_error(y, argentina_predictions)
        
        # Calculate transferability metrics
        transferability = self._calculate_transferability_metrics(
            country_data, argentina_predictions, y
        )
        
        # Cultural adaptation analysis
        cultural_fit = self._analyze_cultural_adaptation(country_code, country_data)
        
        validation_result = {
            'country_code': country_code,
            'country_name': self.country_profiles[country_code].name,
            'performance_metrics': {
                'mse': float(mse),
                'r2_score': float(r2),
                'mae': float(mae),
                'rmse': float(np.sqrt(mse))
            },
            'transferability_metrics': transferability,
            'cultural_adaptation': cultural_fit,
            'data_characteristics': {
                'n_innovations': len(country_data),
                'year_range': [int(country_data['year'].min()), int(country_data['year'].max())],
                'legal_system': self.country_profiles[country_code].legal_system.value,
                'crisis_proportion': float(country_data['in_crisis'].mean())
            },
            'validation_timestamp': datetime.now().isoformat()
        }
        
        self.validation_results[country_code] = validation_result
        
        self.logger.info(f"✅ {country_code} validation complete - R²: {r2:.3f}")
        
        return validation_result
    
    def _simulate_argentina_model_predictions(self, 
                                            X: np.ndarray, 
                                            model_params: Dict[str, Any]) -> np.ndarray:
        """Simulate predictions from Argentina-trained model."""
        
        # Simple linear combination simulation
        # In reality, this would use the actual trained model
        weights = model_params.get('feature_weights', np.ones(X.shape[1]) / X.shape[1])
        bias = model_params.get('bias', 0.0)
        
        # Normalize features
        X_norm = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        
        # Linear prediction with some non-linearity
        predictions = np.dot(X_norm, weights) + bias
        predictions = 1 / (1 + np.exp(-predictions))  # Sigmoid to get 0-1 range
        
        # Add some noise to simulate model uncertainty
        predictions += np.random.normal(0, 0.05, size=predictions.shape)
        
        return np.clip(predictions, 0.0, 1.0)
    
    def _calculate_transferability_metrics(self, 
                                         country_data: pd.DataFrame,
                                         predictions: np.ndarray,
                                         actual: np.ndarray) -> Dict[str, Any]:
        """Calculate metrics specific to cross-country transferability."""
        
        # Performance by reform type
        reform_performance = {}
        for reform_type in country_data['reform_type'].unique():
            mask = country_data['reform_type'] == reform_type
            if mask.sum() > 5:  # Sufficient data
                type_r2 = r2_score(actual[mask], predictions[mask])
                reform_performance[reform_type] = float(type_r2)
        
        # Performance by time period
        temporal_performance = {}
        for decade in [1990, 2000, 2010, 2020]:
            decade_mask = (country_data['year'] >= decade) & (country_data['year'] < decade + 10)
            if decade_mask.sum() > 10:
                decade_r2 = r2_score(actual[decade_mask], predictions[decade_mask])
                temporal_performance[f'{decade}s'] = float(decade_r2)
        
        # Crisis vs normal period performance
        crisis_mask = country_data['in_crisis']
        crisis_r2 = r2_score(actual[crisis_mask], predictions[crisis_mask]) if crisis_mask.sum() > 5 else None
        normal_r2 = r2_score(actual[~crisis_mask], predictions[~crisis_mask]) if (~crisis_mask).sum() > 5 else None
        
        # Distribution similarity (KL divergence)
        from scipy.stats import entropy
        
        # Discretize for KL divergence calculation
        pred_hist, _ = np.histogram(predictions, bins=20, range=(0, 1))
        actual_hist, _ = np.histogram(actual, bins=20, range=(0, 1))
        
        # Add small epsilon to avoid log(0)
        pred_prob = (pred_hist + 1e-10) / (pred_hist.sum() + 1e-9)
        actual_prob = (actual_hist + 1e-10) / (actual_hist.sum() + 1e-9)
        
        kl_divergence = float(entropy(actual_prob, pred_prob))
        
        return {
            'reform_type_performance': reform_performance,
            'temporal_performance': temporal_performance,
            'crisis_performance': {
                'crisis_r2': crisis_r2,
                'normal_r2': normal_r2,
                'performance_gap': abs(crisis_r2 - normal_r2) if (crisis_r2 and normal_r2) else None
            },
            'distribution_similarity': {
                'kl_divergence': kl_divergence,
                'similar_distribution': kl_divergence < 1.0
            },
            'overall_transferability_score': float(np.mean([
                r2_score(actual, predictions),
                1.0 / (1.0 + kl_divergence),  # Convert KL to similarity score
                len(reform_performance) / 5.0  # Coverage score
            ]))
        }
    
    def _analyze_cultural_adaptation(self, 
                                   country_code: str, 
                                   country_data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze how well the model adapts to cultural differences."""
        
        profile = self.country_profiles[country_code]
        
        # Compare cultural dimensions with Argentina (baseline)
        argentina_profile = CountryProfile(
            code='AR', name='Argentina', legal_system=LegalSystem.CIVIL_LAW,
            gdp_per_capita=10937, population_millions=45.8, governance_index=0.65,
            cultural_dimensions={
                'power_distance': 49, 'individualism': 46, 'masculinity': 56,
                'uncertainty_avoidance': 86, 'long_term_orientation': 20
            },
            expected_patterns={}
        )
        
        cultural_distances = {}
        for dim, value in profile.cultural_dimensions.items():
            arg_value = argentina_profile.cultural_dimensions[dim]
            cultural_distances[dim] = abs(value - arg_value)
        
        # Overall cultural distance
        overall_distance = np.mean(list(cultural_distances.values()))
        
        # Legal system compatibility
        legal_compatibility = {
            LegalSystem.CIVIL_LAW: 1.0,      # Same as Argentina
            LegalSystem.MIXED_SYSTEM: 0.7,   # Partially compatible
            LegalSystem.COMMON_LAW: 0.4,     # Different paradigm
            LegalSystem.CUSTOMARY_LAW: 0.2,  # Very different
            LegalSystem.RELIGIOUS_LAW: 0.1   # Fundamentally different
        }
        
        legal_score = legal_compatibility.get(profile.legal_system, 0.5)
        
        # Economic development similarity
        gdp_ratio = min(profile.gdp_per_capita, argentina_profile.gdp_per_capita) / max(profile.gdp_per_capita, argentina_profile.gdp_per_capita)
        
        # Governance similarity
        governance_distance = abs(profile.governance_index - argentina_profile.governance_index)
        
        return {
            'cultural_distances': cultural_distances,
            'overall_cultural_distance': float(overall_distance),
            'legal_system_compatibility': float(legal_score),
            'economic_similarity': float(gdp_ratio),
            'governance_similarity': float(1.0 - governance_distance),
            'adaptation_challenges': self._identify_adaptation_challenges(profile, cultural_distances),
            'cultural_adaptation_score': float(np.mean([
                1.0 - (overall_distance / 100),  # Lower distance = better
                legal_score,
                gdp_ratio,
                1.0 - governance_distance
            ]))
        }
    
    def _identify_adaptation_challenges(self, 
                                      profile: CountryProfile,
                                      cultural_distances: Dict[str, float]) -> List[str]:
        """Identify specific cultural adaptation challenges."""
        
        challenges = []
        
        # High power distance difference
        if cultural_distances['power_distance'] > 30:
            challenges.append("Significant power distance differences may affect institutional reform patterns")
        
        # High individualism difference
        if cultural_distances['individualism'] > 30:
            challenges.append("Individualism differences may impact collective decision-making processes")
        
        # High uncertainty avoidance difference
        if cultural_distances['uncertainty_avoidance'] > 30:
            challenges.append("Uncertainty avoidance differences may affect innovation adoption rates")
        
        # Legal system differences
        if profile.legal_system != LegalSystem.CIVIL_LAW:
            challenges.append(f"Different legal system ({profile.legal_system.value}) requires model adaptation")
        
        # Economic development gap
        if profile.gdp_per_capita < 5000:
            challenges.append("Significant economic development gap may affect innovation patterns")
        elif profile.gdp_per_capita > 30000:
            challenges.append("Higher development level may show different innovation dynamics")
        
        # Governance quality difference
        if abs(profile.governance_index - 0.65) > 0.2:
            challenges.append("Governance quality differences may impact institutional effectiveness")
        
        return challenges
    
    def run_comprehensive_external_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive external validation across all target countries.
        
        Returns:
            Complete external validation results
        """
        self.logger.info("🌍 Starting comprehensive external validation")
        
        # Simulate Argentina model parameters
        argentina_params = {
            'feature_weights': np.array([0.12, 0.15, 0.08, 0.11, 0.13, 0.09, 0.10, 0.12, 0.10]),
            'bias': 0.1,
            'training_r2': 0.75,
            'training_rmse': 0.18
        }
        
        validation_summary = {
            'validation_timestamp': datetime.now().isoformat(),
            'argentina_baseline': argentina_params,
            'country_results': {},
            'comparative_analysis': {},
            'generalizability_assessment': {}
        }
        
        # Validate on each target country
        target_countries = ['CL', 'ZA', 'SE', 'IN']
        
        for country_code in target_countries:
            try:
                result = self.validate_argentina_model_on_country(country_code, argentina_params)
                validation_summary['country_results'][country_code] = result
                
            except Exception as e:
                self.logger.error(f"Validation failed for {country_code}: {e}")
                validation_summary['country_results'][country_code] = {
                    'error': str(e),
                    'validation_failed': True
                }
        
        # Comparative analysis
        validation_summary['comparative_analysis'] = self._perform_comparative_analysis()
        
        # Generalizability assessment
        validation_summary['generalizability_assessment'] = self._assess_generalizability()
        
        # Save results
        results_path = self.config.get_path('results_dir') / f'external_validation_results_{self.config.timestamp}.json'
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(validation_summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"💾 External validation results saved: {results_path}")
        self.logger.info("🏁 Comprehensive external validation complete")
        
        return validation_summary
    
    def _perform_comparative_analysis(self) -> Dict[str, Any]:
        """Perform comparative analysis across countries."""
        
        successful_validations = {k: v for k, v in self.validation_results.items() 
                                if 'error' not in v}
        
        if not successful_validations:
            return {'error': 'No successful validations for comparison'}
        
        # Performance comparison
        performance_comparison = {}
        for country, result in successful_validations.items():
            metrics = result['performance_metrics']
            performance_comparison[country] = {
                'r2_score': metrics['r2_score'],
                'rmse': metrics['rmse'],
                'transferability_score': result['transferability_metrics']['overall_transferability_score']
            }
        
        # Rank countries by model performance
        countries_by_performance = sorted(
            performance_comparison.keys(),
            key=lambda k: performance_comparison[k]['r2_score'],
            reverse=True
        )
        
        # Identify patterns
        legal_system_performance = {}
        for country, result in successful_validations.items():
            legal_sys = result['data_characteristics']['legal_system']
            r2 = result['performance_metrics']['r2_score']
            
            if legal_sys not in legal_system_performance:
                legal_system_performance[legal_sys] = []
            legal_system_performance[legal_sys].append(r2)
        
        # Average performance by legal system
        legal_system_avg = {sys: np.mean(scores) for sys, scores in legal_system_performance.items()}
        
        return {
            'performance_comparison': performance_comparison,
            'performance_ranking': countries_by_performance,
            'legal_system_performance': legal_system_avg,
            'best_performing_country': countries_by_performance[0] if countries_by_performance else None,
            'worst_performing_country': countries_by_performance[-1] if countries_by_performance else None,
            'performance_range': {
                'max_r2': max(p['r2_score'] for p in performance_comparison.values()),
                'min_r2': min(p['r2_score'] for p in performance_comparison.values())
            }
        }
    
    def _assess_generalizability(self) -> Dict[str, Any]:
        """Assess overall generalizability of the Iusmorfos framework."""
        
        successful_validations = {k: v for k, v in self.validation_results.items() 
                                if 'error' not in v}
        
        if not successful_validations:
            return {'generalizability': 'unknown', 'reason': 'No successful validations'}
        
        # Calculate generalizability metrics
        r2_scores = [v['performance_metrics']['r2_score'] for v in successful_validations.values()]
        transferability_scores = [v['transferability_metrics']['overall_transferability_score'] 
                                for v in successful_validations.values()]
        
        mean_r2 = np.mean(r2_scores)
        std_r2 = np.std(r2_scores)
        mean_transferability = np.mean(transferability_scores)
        
        # Generalizability criteria
        good_performance_threshold = 0.6  # R² > 0.6
        consistency_threshold = 0.15      # Std < 0.15
        transferability_threshold = 0.7   # Transfer score > 0.7
        
        good_performance = mean_r2 >= good_performance_threshold
        consistent_performance = std_r2 <= consistency_threshold
        high_transferability = mean_transferability >= transferability_threshold
        
        # Overall assessment
        criteria_met = sum([good_performance, consistent_performance, high_transferability])
        
        if criteria_met >= 3:
            generalizability_level = 'high'
        elif criteria_met >= 2:
            generalizability_level = 'moderate'
        else:
            generalizability_level = 'low'
        
        # Identify limitations
        limitations = []
        if not good_performance:
            limitations.append(f"Mean R² ({mean_r2:.3f}) below threshold ({good_performance_threshold})")
        if not consistent_performance:
            limitations.append(f"High variability (σ = {std_r2:.3f}) exceeds threshold ({consistency_threshold})")
        if not high_transferability:
            limitations.append(f"Low transferability ({mean_transferability:.3f}) below threshold ({transferability_threshold})")
        
        return {
            'generalizability_level': generalizability_level,
            'criteria_assessment': {
                'good_performance': good_performance,
                'consistent_performance': consistent_performance,
                'high_transferability': high_transferability,
                'criteria_met': criteria_met,
                'total_criteria': 3
            },
            'quantitative_metrics': {
                'mean_r2': float(mean_r2),
                'std_r2': float(std_r2),
                'mean_transferability': float(mean_transferability),
                'n_countries_validated': len(successful_validations)
            },
            'limitations': limitations,
            'recommendations': self._generate_generalizability_recommendations(
                generalizability_level, limitations
            )
        }
    
    def _generate_generalizability_recommendations(self, 
                                                 level: str, 
                                                 limitations: List[str]) -> List[str]:
        """Generate recommendations based on generalizability assessment."""
        
        recommendations = []
        
        if level == 'high':
            recommendations.extend([
                "✅ Framework shows strong cross-country generalizability",
                "📈 Consider expanding to additional legal systems",
                "🔬 Implement in production with confidence monitoring"
            ])
        elif level == 'moderate':
            recommendations.extend([
                "⚠️ Framework shows partial generalizability",
                "🎯 Focus on improving performance in underperforming countries",
                "📊 Collect additional validation data"
            ])
        else:
            recommendations.extend([
                "🚨 Framework shows limited generalizability",
                "🔄 Significant model adaptation required for cross-country use",
                "📚 Conduct deeper analysis of cultural and legal differences"
            ])
        
        # Specific recommendations based on limitations
        for limitation in limitations:
            if "Mean R²" in limitation:
                recommendations.append("🎯 Improve core model performance through feature engineering")
            elif "variability" in limitation:
                recommendations.append("🎲 Develop country-specific adaptation mechanisms")
            elif "transferability" in limitation:
                recommendations.append("🌍 Create cultural adaptation layers for the model")
        
        return recommendations


def main():
    """Main external validation function."""
    print("🌍 Iusmorfos External Validation Framework")
    print("=" * 50)
    
    # Initialize framework
    validator = ExternalValidationFramework()
    
    # Run comprehensive validation
    results = validator.run_comprehensive_external_validation()
    
    # Display summary
    print(f"\n📊 Validation Summary:")
    print(f"Countries tested: {len(results['country_results'])}")
    
    successful = sum(1 for r in results['country_results'].values() if 'error' not in r)
    print(f"Successful validations: {successful}")
    
    if 'generalizability_assessment' in results:
        assessment = results['generalizability_assessment']
        print(f"\n🎯 Generalizability Assessment:")
        print(f"Level: {assessment.get('generalizability_level', 'unknown').upper()}")
        
        if 'quantitative_metrics' in assessment:
            metrics = assessment['quantitative_metrics']
            print(f"Mean R²: {metrics['mean_r2']:.3f}")
            print(f"Mean transferability: {metrics['mean_transferability']:.3f}")
    
    if 'comparative_analysis' in results:
        comparison = results['comparative_analysis']
        if 'performance_ranking' in comparison and comparison['performance_ranking']:
            print(f"\n🏆 Performance Ranking:")
            for i, country in enumerate(comparison['performance_ranking'], 1):
                r2 = results['country_results'][country]['performance_metrics']['r2_score']
                print(f"{i}. {country}: R² = {r2:.3f}")


if __name__ == "__main__":
    main()