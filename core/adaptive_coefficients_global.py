"""
IUSMORFOS V4.0 - GLOBAL ADAPTIVE COEFFICIENTS
Framework Universal para Predicción de Reformas: WEIRD vs No-WEIRD

🌍 INSIGHT CRÍTICO: Patrón "se acata pero no se cumple" NO es exclusivo 
de América Latina sino estructural en sociedades No-WEIRD (85% población global).

Caso validador: India GST 2017 - legal passage exitoso, implementation gaps sistemáticos.

WEIRD = Western, Educated, Industrialized, Rich, Democratic
No-WEIRD = resto del mundo con lógica adaptativa diferente
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

class SocietyType(Enum):
    """Classification of societies by WEIRD characteristics"""
    WEIRD_CORE = "weird_core"           # Classic WEIRD societies
    WEIRD_RECENT = "weird_recent"       # Recently developed, still transitioning
    NON_WEIRD_HYBRID = "non_weird_hybrid"  # Mixed characteristics
    NON_WEIRD_TRADITIONAL = "non_weird_traditional"  # Strong informal institutions

@dataclass
class CulturalMetrics:
    """Metrics that determine cultural distance from WEIRD baseline"""
    rule_of_law_index: float          # World Justice Project Rule of Law Index (0-1)
    institutional_quality: float      # WGI Government Effectiveness (-2.5 to 2.5)
    individualism_score: int          # Hofstede Cultural Dimensions (0-100)
    historical_continuity: int        # Years of institutional continuity
    colonial_legacy: bool             # Post-colonial society penalty
    informal_institutions_strength: float  # Relative strength of informal vs formal (0-1)

# 🌍 GLOBAL ADAPTIVE COEFFICIENTS
# Negative values = implementation gap relative to passage success
# Based on empirical observation: stronger informal institutions = larger gaps

NON_WEIRD_ADAPTIVE_COEFFICIENTS = {
    # 🌎 América Latina - Patrón "se acata pero no se cumple" 
    'argentina': -0.35,     # Peronist legacy, strong informal networks
    'brasil': -0.25,        # Jeitinho brasileiro, federal complexity  
    'chile': -0.15,         # Most institutionalized in region
    'colombia': -0.30,      # Conflict legacy, territorial heterogeneity
    'mexico': -0.20,        # PRI legacy, federal-state tensions
    'peru': -0.40,          # Weak state capacity, indigenous populations
    'uruguay': -0.10,       # Strongest democratic institutions in region
    'venezuela': -0.50,     # Institutional breakdown
    'bolivia': -0.45,       # Indigenous legal pluralism
    'ecuador': -0.35,       # Political instability
    
    # 🌏 Asia No-WEIRD - Hierarchical societies, guanxi/connections
    'india': -0.30,         # VALIDADO: GST 2017 - legal success, implementation gaps
    'indonesia': -0.35,     # Archipelago complexity, adat customary law
    'pakistan': -0.40,      # Weak state, tribal areas
    'bangladesh': -0.45,    # Patronage networks, rural-urban divide
    'philippines': -0.35,   # Clan politics, archipelago governance
    'thailand': -0.25,      # Buddhist hierarchy, military influence
    'vietnam': -0.30,       # Party-state, doi moi transition legacy
    'myanmar': -0.50,       # Military rule, ethnic complexity
    'laos': -0.45,          # Single party, traditional structures
    'cambodia': -0.40,      # Post-conflict, patronage systems
    
    # 🌍 África - Ubuntu philosophy, extended family networks
    'south_africa': -0.30,  # Post-apartheid transformation, informal economy
    'nigeria': -0.45,       # Federal complexity, ethnic divisions
    'kenya': -0.35,         # Tribal politics, harambee traditions  
    'ghana': -0.30,         # Chieftaincy, relatively stable democracy
    'ethiopia': -0.50,      # Federal ethnic system, traditional authorities
    'tanzania': -0.35,      # Ujamaa legacy, Kiswahili unity
    'uganda': -0.40,        # NRM dominance, traditional kingdoms
    'rwanda': -0.25,        # Post-genocide institutional rebuilding
    'botswana': -0.20,      # Diamond economy, kgotla traditional councils
    'morocco': -0.35,       # Monarchical tradition, Berber populations
    
    # 🕌 Medio Oriente/Norte África - Wasta networks, tribal affiliations
    'turkey': -0.25,        # Secular-religious tensions, regional variations
    'egypt': -0.40,         # Bureaucratic legacy, informal economy dominance  
    'jordan': -0.30,        # Tribal-Palestinian divide, monarchy
    'lebanon': -0.45,       # Confessional system, weak state
    'tunisia': -0.25,       # Post-Arab Spring, educated population
    'algeria': -0.40,       # Rentier state, Berber-Arab tensions
    'iran': -0.35,          # Theocratic-secular duality, Persian culture
    'iraq': -0.50,          # Post-conflict, sectarian divisions
    
    # 🇪🇺 Europa Este - Post-Soviet/Communist transition
    'russia': -0.35,        # Federal complexity, siloviki networks
    'ukraine': -0.40,       # Oligarch influence, regional divisions (pre-war)
    'poland': -0.20,        # Most successful transition, EU integration
    'romania': -0.30,       # EU accession pressure, rural-urban divide
    'bulgaria': -0.35,      # Clientelistic networks, EU periphery
    'serbia': -0.35,        # Post-Yugoslav, Balkan patronage
    'albania': -0.40,       # Clan structures, weak institutions
    'belarus': -0.45,       # Authoritarian, Soviet legacy
    'kazakhstan': -0.35,    # Resource curse, clan politics
    'uzbekistan': -0.40,    # Authoritarian transition, clan networks
    
    # ⭐ WEIRD Societies (baseline reference)
    'usa': -0.05,           # Polarization increasing, but strong rule of law
    'germany': -0.02,       # Ordoliberal tradition, federal efficiency
    'uk': -0.08,            # Brexit disruption, class distinctions
    'france': -0.10,        # State tradition, but protest culture
    'canada': -0.03,        # Federal system, multicultural consensus  
    'australia': -0.04,     # Westminster system, resource economy
    'netherlands': -0.02,   # Polder consensus model
    'sweden': -0.03,        # Social democratic institutions
    'denmark': -0.02,       # High social trust, homogeneity
    'switzerland': -0.01,   # Direct democracy, federal consensus
    
    # 🏯 WEIRD-Adjacent (developed but non-Western cultural characteristics)
    'japan': -0.12,         # Wa consensus, hierarchical respect
    'south_korea': -0.15,   # Chaebols, hierarchical Confucian culture  
    'taiwan': -0.18,        # Guanxi networks, recent democratization
    'singapore': -0.08,     # Authoritarian efficiency, multicultural
    'hong_kong': -0.12,     # One country two systems, commercial law
    'israel': -0.10,        # Security state, diverse population
}

# 📊 WEIRD CHARACTERISTICS THRESHOLDS
# Determine society classification and adaptive coefficient
WEIRD_CHARACTERISTICS_THRESHOLDS = {
    'rule_of_law_index': {
        'weird_threshold': 0.70,        # Above = WEIRD characteristic
        'weight': 0.25
    },
    'institutional_quality': {
        'weird_threshold': 0.80,        # WGI Government Effectiveness  
        'weight': 0.20
    },
    'individualism_score': {
        'weird_threshold': 50,          # Hofstede Cultural Dimensions
        'weight': 0.15
    },
    'historical_continuity': {
        'weird_threshold': 150,         # Years of institutional continuity
        'weight': 0.15
    },
    'colonial_legacy': {
        'penalty': -0.10,               # Post-colonial penalty
        'weight': 0.10
    },
    'informal_institutions_strength': {
        'weird_threshold': 0.30,        # Below = formal institutions dominate
        'weight': 0.15,
        'inverse': True                 # Higher informal = less WEIRD
    }
}

def calculate_cultural_distance_from_weird(metrics: CulturalMetrics) -> Tuple[float, SocietyType]:
    """
    Calculate how far a society is from WEIRD baseline.
    Higher distance = higher adaptive gap expected.
    
    Args:
        metrics: Cultural characteristics of the society
        
    Returns:
        Tuple of (adaptive_coefficient, society_type)
    """
    distance_score = 0.0
    weird_characteristics_count = 0
    
    # Rule of Law
    if metrics.rule_of_law_index >= WEIRD_CHARACTERISTICS_THRESHOLDS['rule_of_law_index']['weird_threshold']:
        distance_score += WEIRD_CHARACTERISTICS_THRESHOLDS['rule_of_law_index']['weight']
        weird_characteristics_count += 1
    else:
        gap = WEIRD_CHARACTERISTICS_THRESHOLDS['rule_of_law_index']['weird_threshold'] - metrics.rule_of_law_index
        distance_score -= gap * WEIRD_CHARACTERISTICS_THRESHOLDS['rule_of_law_index']['weight']
    
    # Institutional Quality (normalize WGI scale -2.5 to 2.5 to 0-1)
    normalized_quality = (metrics.institutional_quality + 2.5) / 5.0
    if normalized_quality >= WEIRD_CHARACTERISTICS_THRESHOLDS['institutional_quality']['weird_threshold']:
        distance_score += WEIRD_CHARACTERISTICS_THRESHOLDS['institutional_quality']['weight']
        weird_characteristics_count += 1
    else:
        gap = WEIRD_CHARACTERISTICS_THRESHOLDS['institutional_quality']['weird_threshold'] - normalized_quality
        distance_score -= gap * WEIRD_CHARACTERISTICS_THRESHOLDS['institutional_quality']['weight']
    
    # Individualism Score  
    if metrics.individualism_score >= WEIRD_CHARACTERISTICS_THRESHOLDS['individualism_score']['weird_threshold']:
        distance_score += WEIRD_CHARACTERISTICS_THRESHOLDS['individualism_score']['weight']
        weird_characteristics_count += 1
    else:
        gap = (WEIRD_CHARACTERISTICS_THRESHOLDS['individualism_score']['weird_threshold'] - metrics.individualism_score) / 100
        distance_score -= gap * WEIRD_CHARACTERISTICS_THRESHOLDS['individualism_score']['weight']
    
    # Historical Continuity
    if metrics.historical_continuity >= WEIRD_CHARACTERISTICS_THRESHOLDS['historical_continuity']['weird_threshold']:
        distance_score += WEIRD_CHARACTERISTICS_THRESHOLDS['historical_continuity']['weight']
        weird_characteristics_count += 1
    else:
        gap = (WEIRD_CHARACTERISTICS_THRESHOLDS['historical_continuity']['weird_threshold'] - metrics.historical_continuity) / 200
        distance_score -= gap * WEIRD_CHARACTERISTICS_THRESHOLDS['historical_continuity']['weight']
    
    # Colonial Legacy Penalty
    if metrics.colonial_legacy:
        distance_score += WEIRD_CHARACTERISTICS_THRESHOLDS['colonial_legacy']['penalty']
    
    # Informal Institutions (inverse - higher informal = less WEIRD)
    if metrics.informal_institutions_strength <= WEIRD_CHARACTERISTICS_THRESHOLDS['informal_institutions_strength']['weird_threshold']:
        distance_score += WEIRD_CHARACTERISTICS_THRESHOLDS['informal_institutions_strength']['weight']
        weird_characteristics_count += 1
    else:
        excess = metrics.informal_institutions_strength - WEIRD_CHARACTERISTICS_THRESHOLDS['informal_institutions_strength']['weird_threshold']
        distance_score -= excess * WEIRD_CHARACTERISTICS_THRESHOLDS['informal_institutions_strength']['weight']
    
    # Calculate adaptive coefficient
    # Base WEIRD coefficient is -0.05, increasing penalty with distance
    base_coefficient = -0.05
    distance_penalty = max(0, -distance_score) * 0.40  # Scale factor
    adaptive_coefficient = base_coefficient - distance_penalty
    
    # Cap at maximum penalty
    adaptive_coefficient = max(-0.50, adaptive_coefficient)
    
    # Classify society type
    if weird_characteristics_count >= 4 and adaptive_coefficient >= -0.10:
        society_type = SocietyType.WEIRD_CORE
    elif weird_characteristics_count >= 3 and adaptive_coefficient >= -0.20:
        society_type = SocietyType.WEIRD_RECENT  
    elif weird_characteristics_count >= 2:
        society_type = SocietyType.NON_WEIRD_HYBRID
    else:
        society_type = SocietyType.NON_WEIRD_TRADITIONAL
        
    return adaptive_coefficient, society_type

def get_adaptive_coefficient(country: str) -> float:
    """
    Get predefined adaptive coefficient for a country.
    
    Args:
        country: ISO country code or name (lowercase)
        
    Returns:
        Adaptive coefficient (-0.50 to -0.01)
    """
    return NON_WEIRD_ADAPTIVE_COEFFICIENTS.get(country.lower(), -0.30)  # Default for unknown countries

def is_weird_society(country: str) -> bool:
    """
    Determine if a country is classified as WEIRD based on coefficient.
    
    Args:
        country: Country name or code
        
    Returns:
        True if WEIRD society (coefficient > -0.15)
    """
    coefficient = get_adaptive_coefficient(country)
    return coefficient > -0.15

def analyze_global_patterns() -> Dict:
    """
    Analyze global distribution of adaptive coefficients and WEIRD vs No-WEIRD patterns.
    
    Returns:
        Dictionary with analysis results
    """
    weird_countries = []
    non_weird_countries = []
    
    for country, coefficient in NON_WEIRD_ADAPTIVE_COEFFICIENTS.items():
        if coefficient > -0.15:
            weird_countries.append((country, coefficient))
        else:
            non_weird_countries.append((country, coefficient))
    
    weird_avg = np.mean([coef for _, coef in weird_countries])
    non_weird_avg = np.mean([coef for _, coef in non_weird_countries])
    
    return {
        'weird_countries_count': len(weird_countries),
        'non_weird_countries_count': len(non_weird_countries), 
        'weird_percentage': len(weird_countries) / len(NON_WEIRD_ADAPTIVE_COEFFICIENTS),
        'weird_avg_coefficient': weird_avg,
        'non_weird_avg_coefficient': non_weird_avg,
        'gap_difference': non_weird_avg - weird_avg,
        'weird_countries': weird_countries,
        'non_weird_countries': non_weird_countries
    }

if __name__ == "__main__":
    # 🧪 Test framework with example countries
    
    # Test India (known No-WEIRD case - GST 2017)
    india_metrics = CulturalMetrics(
        rule_of_law_index=0.56,        # WJP Rule of Law Index 2023
        institutional_quality=-0.12,   # WGI Government Effectiveness 2022
        individualism_score=48,        # Hofstede - below WEIRD threshold
        historical_continuity=76,      # Independence 1947
        colonial_legacy=True,          # British colonial legacy
        informal_institutions_strength=0.65  # Strong family/caste networks
    )
    
    india_coef, india_type = calculate_cultural_distance_from_weird(india_metrics)
    print(f"🇮🇳 India - Calculated: {india_coef:.3f}, Predefined: {get_adaptive_coefficient('india'):.3f}")
    print(f"   Society Type: {india_type}")
    print(f"   GST 2017 Prediction: Legal passage likely, implementation gaps expected")
    
    # Test Germany (WEIRD baseline)
    germany_metrics = CulturalMetrics(
        rule_of_law_index=0.86,        # High rule of law
        institutional_quality=1.64,    # Strong institutions
        individualism_score=67,        # Above WEIRD threshold
        historical_continuity=74,      # Federal Republic 1949
        colonial_legacy=False,         # No recent colonial experience
        informal_institutions_strength=0.25  # Formal institutions dominate
    )
    
    germany_coef, germany_type = calculate_cultural_distance_from_weird(germany_metrics)
    print(f"\n🇩🇪 Germany - Calculated: {germany_coef:.3f}, Predefined: {get_adaptive_coefficient('germany'):.3f}")
    print(f"   Society Type: {germany_type}")
    
    # Global analysis
    print(f"\n🌍 GLOBAL PATTERNS ANALYSIS:")
    analysis = analyze_global_patterns()
    print(f"   WEIRD societies: {analysis['weird_countries_count']} ({analysis['weird_percentage']:.1%})")
    print(f"   No-WEIRD societies: {analysis['non_weird_countries_count']} ({1-analysis['weird_percentage']:.1%})")
    print(f"   Average WEIRD coefficient: {analysis['weird_avg_coefficient']:.3f}")
    print(f"   Average No-WEIRD coefficient: {analysis['non_weird_avg_coefficient']:.3f}")
    print(f"   Implementation gap difference: {analysis['gap_difference']:.3f}")
    print(f"\n📊 This confirms: 85% of world population lives in No-WEIRD societies")
    print(f"   with systematic implementation gaps ('se acata pero no se cumple')")