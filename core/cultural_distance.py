"""
IUSMORFOS V4.0 - CULTURAL DISTANCE CALCULATOR
Determines adaptive coefficients based on distance from WEIRD characteristics

🎯 OBJETIVO: Automatic calculation of implementation gaps based on cultural/institutional factors
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, asdict
import json
from pathlib import Path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.adaptive_coefficients_global import (
    CulturalMetrics, SocietyType, WEIRD_CHARACTERISTICS_THRESHOLDS,
    calculate_cultural_distance_from_weird, get_adaptive_coefficient
)

@dataclass 
class CountryProfile:
    """Complete country profile for cultural distance analysis"""
    country_name: str
    iso_code: str
    cultural_metrics: CulturalMetrics
    adaptive_coefficient: float
    society_type: SocietyType
    recent_reforms: List[str]
    validation_cases: List[Dict]

class CulturalDistanceCalculator:
    """
    Calculate and manage cultural distances from WEIRD baseline.
    Predict implementation gaps based on institutional characteristics.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        """Initialize with cultural metrics database"""
        self.data_path = Path(data_path) if data_path else Path(__file__).parent.parent / "data"
        self.country_profiles = {}
        self.load_cultural_database()
    
    def load_cultural_database(self):
        """Load cultural metrics from JSON database"""
        cultural_data_file = self.data_path / "cultural_metrics.json"
        
        if cultural_data_file.exists():
            with open(cultural_data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self._parse_cultural_data(data)
        else:
            print(f"⚠️  Cultural database not found at {cultural_data_file}")
            self._create_sample_database()
    
    def _parse_cultural_data(self, data: Dict):
        """Parse loaded cultural data into CountryProfile objects"""
        for country, metrics_data in data.items():
            try:
                metrics = CulturalMetrics(**metrics_data['cultural_metrics'])
                coef, society_type = calculate_cultural_distance_from_weird(metrics)
                
                profile = CountryProfile(
                    country_name=metrics_data.get('country_name', country),
                    iso_code=metrics_data.get('iso_code', country.upper()[:3]),
                    cultural_metrics=metrics,
                    adaptive_coefficient=coef,
                    society_type=society_type,
                    recent_reforms=metrics_data.get('recent_reforms', []),
                    validation_cases=metrics_data.get('validation_cases', [])
                )
                
                self.country_profiles[country.lower()] = profile
                
            except Exception as e:
                print(f"❌ Error parsing data for {country}: {e}")
    
    def _create_sample_database(self):
        """Create sample cultural database for testing"""
        sample_data = {
            'india': {
                'country_name': 'India',
                'iso_code': 'IND', 
                'cultural_metrics': {
                    'rule_of_law_index': 0.56,
                    'institutional_quality': -0.12,
                    'individualism_score': 48,
                    'historical_continuity': 76,
                    'colonial_legacy': True,
                    'informal_institutions_strength': 0.65
                },
                'recent_reforms': ['GST_2017', 'Farm_Laws_2020', 'Labor_Code_2020'],
                'validation_cases': [
                    {
                        'reform': 'GST_2017',
                        'passage_success': 0.95,
                        'implementation_success': 0.65,
                        'gap': 0.30,
                        'description': 'Legal passage successful, GSTN portal issues, compliance gaps'
                    }
                ]
            },
            'germany': {
                'country_name': 'Germany', 
                'iso_code': 'DEU',
                'cultural_metrics': {
                    'rule_of_law_index': 0.86,
                    'institutional_quality': 1.64,
                    'individualism_score': 67,
                    'historical_continuity': 74,
                    'colonial_legacy': False,
                    'informal_institutions_strength': 0.25
                },
                'recent_reforms': ['Immigration_Reform_2016', 'Digital_Services_Act_2021'],
                'validation_cases': []
            },
            'argentina': {
                'country_name': 'Argentina',
                'iso_code': 'ARG',
                'cultural_metrics': {
                    'rule_of_law_index': 0.52,
                    'institutional_quality': -0.35,
                    'individualism_score': 46,
                    'historical_continuity': 40,  # Democratic continuity since 1983
                    'colonial_legacy': True,
                    'informal_institutions_strength': 0.70
                },
                'recent_reforms': ['Ley_Bases_2024', 'Tax_Reform_Milei_2024'],
                'validation_cases': []
            }
        }
        
        # Save sample data
        os.makedirs(self.data_path, exist_ok=True)
        with open(self.data_path / "cultural_metrics.json", 'w', encoding='utf-8') as f:
            json.dump(sample_data, f, indent=2, ensure_ascii=False)
        
        self._parse_cultural_data(sample_data)
    
    def calculate_distance(self, country: str, metrics: Optional[CulturalMetrics] = None) -> Tuple[float, SocietyType, Dict]:
        """
        Calculate cultural distance and adaptive coefficient for a country.
        
        Args:
            country: Country name or ISO code
            metrics: Optional custom metrics, otherwise use database
            
        Returns:
            Tuple of (adaptive_coefficient, society_type, analysis_details)
        """
        country_key = country.lower()
        
        if metrics is None:
            if country_key in self.country_profiles:
                metrics = self.country_profiles[country_key].cultural_metrics
            else:
                raise ValueError(f"No cultural metrics found for {country}. Please provide metrics.")
        
        adaptive_coef, society_type = calculate_cultural_distance_from_weird(metrics)
        
        # Detailed analysis
        analysis = self._analyze_characteristics(metrics)
        analysis.update({
            'adaptive_coefficient': adaptive_coef,
            'society_type': society_type.value,
            'country': country,
            'weird_classification': society_type in [SocietyType.WEIRD_CORE, SocietyType.WEIRD_RECENT]
        })
        
        return adaptive_coef, society_type, analysis
    
    def _analyze_characteristics(self, metrics: CulturalMetrics) -> Dict:
        """Analyze individual WEIRD characteristics"""
        analysis = {}
        
        # Rule of Law
        rol_threshold = WEIRD_CHARACTERISTICS_THRESHOLDS['rule_of_law_index']['weird_threshold']
        analysis['rule_of_law'] = {
            'value': metrics.rule_of_law_index,
            'threshold': rol_threshold,
            'meets_weird_criteria': metrics.rule_of_law_index >= rol_threshold,
            'gap': rol_threshold - metrics.rule_of_law_index
        }
        
        # Institutional Quality
        iq_threshold = WEIRD_CHARACTERISTICS_THRESHOLDS['institutional_quality']['weird_threshold'] 
        normalized_iq = (metrics.institutional_quality + 2.5) / 5.0
        analysis['institutional_quality'] = {
            'value': metrics.institutional_quality,
            'normalized': normalized_iq,
            'threshold': iq_threshold,
            'meets_weird_criteria': normalized_iq >= iq_threshold,
            'gap': iq_threshold - normalized_iq
        }
        
        # Individualism
        ind_threshold = WEIRD_CHARACTERISTICS_THRESHOLDS['individualism_score']['weird_threshold']
        analysis['individualism'] = {
            'value': metrics.individualism_score,
            'threshold': ind_threshold,
            'meets_weird_criteria': metrics.individualism_score >= ind_threshold,
            'gap': ind_threshold - metrics.individualism_score
        }
        
        # Historical Continuity  
        hc_threshold = WEIRD_CHARACTERISTICS_THRESHOLDS['historical_continuity']['weird_threshold']
        analysis['historical_continuity'] = {
            'value': metrics.historical_continuity,
            'threshold': hc_threshold,
            'meets_weird_criteria': metrics.historical_continuity >= hc_threshold,
            'gap': hc_threshold - metrics.historical_continuity
        }
        
        # Colonial Legacy
        analysis['colonial_legacy'] = {
            'value': metrics.colonial_legacy,
            'penalty': WEIRD_CHARACTERISTICS_THRESHOLDS['colonial_legacy']['penalty'] if metrics.colonial_legacy else 0
        }
        
        # Informal Institutions
        inf_threshold = WEIRD_CHARACTERISTICS_THRESHOLDS['informal_institutions_strength']['weird_threshold']
        analysis['informal_institutions'] = {
            'value': metrics.informal_institutions_strength,
            'threshold': inf_threshold,
            'meets_weird_criteria': metrics.informal_institutions_strength <= inf_threshold,
            'excess': max(0, metrics.informal_institutions_strength - inf_threshold)
        }
        
        return analysis
    
    def compare_countries(self, countries: List[str]) -> pd.DataFrame:
        """
        Compare cultural distances and adaptive coefficients across countries.
        
        Args:
            countries: List of country names/codes
            
        Returns:
            DataFrame with comparison results
        """
        comparisons = []
        
        for country in countries:
            try:
                coef, society_type, analysis = self.calculate_distance(country)
                
                comparison = {
                    'Country': country.title(),
                    'Society_Type': society_type.value,
                    'Adaptive_Coefficient': coef,
                    'Rule_of_Law': analysis['rule_of_law']['value'],
                    'Institutional_Quality': analysis['institutional_quality']['value'], 
                    'Individualism': analysis['individualism']['value'],
                    'Historical_Continuity': analysis['historical_continuity']['value'],
                    'Colonial_Legacy': analysis['colonial_legacy']['value'],
                    'Informal_Institutions': analysis['informal_institutions']['value'],
                    'WEIRD_Criteria_Met': sum([
                        analysis['rule_of_law']['meets_weird_criteria'],
                        analysis['institutional_quality']['meets_weird_criteria'],
                        analysis['individualism']['meets_weird_criteria'], 
                        analysis['historical_continuity']['meets_weird_criteria'],
                        not analysis['colonial_legacy']['value'],
                        analysis['informal_institutions']['meets_weird_criteria']
                    ])
                }
                
                comparisons.append(comparison)
                
            except Exception as e:
                print(f"❌ Error analyzing {country}: {e}")
        
        return pd.DataFrame(comparisons)
    
    def predict_implementation_gap(self, country: str, passage_probability: float) -> Dict:
        """
        Predict implementation success based on passage probability and cultural distance.
        
        Args:
            country: Country name/code
            passage_probability: Predicted probability of legal passage (0-1)
            
        Returns:
            Dictionary with implementation predictions
        """
        try:
            coef, society_type, analysis = self.calculate_distance(country)
            
            # Apply adaptive coefficient to predict implementation
            implementation_probability = passage_probability + coef
            implementation_probability = max(0.05, min(0.95, implementation_probability))  # Bound between 5-95%
            
            gap = passage_probability - implementation_probability
            
            prediction = {
                'country': country,
                'society_type': society_type.value,
                'passage_probability': passage_probability,
                'implementation_probability': implementation_probability,
                'expected_gap': gap,
                'adaptive_coefficient': coef,
                'gap_magnitude': self._classify_gap_magnitude(gap),
                'cultural_factors': self._identify_key_factors(analysis),
                'recommendations': self._generate_recommendations(gap, analysis)
            }
            
            return prediction
            
        except Exception as e:
            return {'error': f"Prediction failed for {country}: {e}"}
    
    def _classify_gap_magnitude(self, gap: float) -> str:
        """Classify the magnitude of implementation gap"""
        if gap < 0.10:
            return "Small gap - Minimal implementation challenges expected"
        elif gap < 0.25:
            return "Moderate gap - Significant adaptation required" 
        elif gap < 0.40:
            return "Large gap - Major implementation challenges"
        else:
            return "Critical gap - High risk of implementation failure"
    
    def _identify_key_factors(self, analysis: Dict) -> List[str]:
        """Identify key cultural factors contributing to gaps"""
        factors = []
        
        if not analysis['rule_of_law']['meets_weird_criteria']:
            factors.append(f"Weak rule of law (index: {analysis['rule_of_law']['value']:.2f})")
        
        if not analysis['institutional_quality']['meets_weird_criteria']:
            factors.append(f"Low institutional quality ({analysis['institutional_quality']['value']:.2f})")
        
        if not analysis['individualism']['meets_weird_criteria']:
            factors.append(f"Collectivist culture (score: {analysis['individualism']['value']})")
        
        if analysis['colonial_legacy']['value']:
            factors.append("Post-colonial institutional legacy")
        
        if not analysis['informal_institutions']['meets_weird_criteria']:
            factors.append(f"Strong informal institutions ({analysis['informal_institutions']['value']:.2f})")
        
        return factors
    
    def _generate_recommendations(self, gap: float, analysis: Dict) -> List[str]:
        """Generate recommendations based on gap size and cultural factors"""
        recommendations = []
        
        if gap > 0.20:
            recommendations.append("Phase implementation gradually to allow adaptation")
            recommendations.append("Engage informal leaders and networks early")
        
        if not analysis['rule_of_law']['meets_weird_criteria']:
            recommendations.append("Strengthen enforcement mechanisms and judicial capacity")
        
        if not analysis['institutional_quality']['meets_weird_criteria']:
            recommendations.append("Build administrative capacity before implementation")
        
        if analysis['informal_institutions']['value'] > 0.60:
            recommendations.append("Design formal rules that align with informal norms")
            recommendations.append("Use existing social networks for implementation")
        
        if analysis['colonial_legacy']['value']:
            recommendations.append("Consider historical institutional patterns in design")
        
        return recommendations

def main():
    """Test the cultural distance calculator"""
    calculator = CulturalDistanceCalculator()
    
    # Test countries
    test_countries = ['india', 'germany', 'argentina']
    
    print("🌍 CULTURAL DISTANCE ANALYSIS")
    print("=" * 50)
    
    for country in test_countries:
        print(f"\n📍 {country.upper()}")
        coef, society_type, analysis = calculator.calculate_distance(country)
        
        print(f"   Adaptive Coefficient: {coef:.3f}")
        print(f"   Society Type: {society_type.value}")
        print(f"   WEIRD Criteria Met: {sum([
            analysis['rule_of_law']['meets_weird_criteria'],
            analysis['institutional_quality']['meets_weird_criteria'], 
            analysis['individualism']['meets_weird_criteria'],
            analysis['historical_continuity']['meets_weird_criteria'],
            not analysis['colonial_legacy']['value'],
            analysis['informal_institutions']['meets_weird_criteria']
        ])}/6")
        
        # Prediction example
        passage_prob = 0.80
        prediction = calculator.predict_implementation_gap(country, passage_prob)
        print(f"   Example: {passage_prob:.0%} passage → {prediction['implementation_probability']:.0%} implementation")
        print(f"   Gap: {prediction['expected_gap']:.3f} ({prediction['gap_magnitude']})")
    
    # Comparison table
    print(f"\n📊 COMPARATIVE ANALYSIS")
    comparison_df = calculator.compare_countries(test_countries)
    print(comparison_df.to_string(index=False))

if __name__ == "__main__":
    main()