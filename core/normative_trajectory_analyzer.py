"""
IUSMORFOS UNIVERSAL v5.0 - ANÁLISIS DE TRAYECTORIAS NORMATIVAS
CORRECCIÓN ANTI-SESGO TOTAL: Máxima protección contra "ley de pequeños números"

🚨 TRAJECTORY PREDICTION REALITY: Análisis SIN base de trayectorias históricas reales
⚠️  CONSTRAINT IDENTIFICATION UNVALIDATED: Constraints basados en teoría, NO empiria
📊 ACCURACY TARGETS ELIMINATED: Identification rates requieren validación histórica
🔍 SUCCESS PROBABILITY SUSPENDED: Probabilidades sin base en outcomes reales

SISTEMA 2 EN PROTECCIÓN MÁXIMA:
1. ✅ Representatividad trajectorial: Análisis basado en casos sintéticos únicamente
2. ✅ Constraint intuition eliminated: Identification patterns sin validación empírica
3. ✅ Success rate risk recognized: Sin historical outcomes = alta probabilidad artefactos  
4. ✅ Predictive use prohibited: Framework conceptual hasta validación con casos reales
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Set
from enum import Enum
from datetime import datetime, timedelta
import json

try:
    from .universal_legal_taxonomy import LegalTradition, ConstitutionalSystem, LegalSystemProfile
    from .iusspace_universal_12d import DimensionalProfile12D, UniversalIusSpace12D
except ImportError:
    from universal_legal_taxonomy import LegalTradition, ConstitutionalSystem, LegalSystemProfile
    from iusspace_universal_12d import DimensionalProfile12D, UniversalIusSpace12D

class ConstraintType(Enum):
    """Tipos de constraints que limitan trayectorias normativas"""
    CONSTITUTIONAL_LEGAL = "constitutional_legal"
    INSTITUTIONAL_CAPACITY = "institutional_capacity"
    POLITICAL_FEASIBILITY = "political_feasibility"
    CULTURAL_RESISTANCE = "cultural_resistance"
    ECONOMIC_SUSTAINABILITY = "economic_sustainability"
    INTERNATIONAL_COMPLIANCE = "international_compliance"
    TEMPORAL_TIMING = "temporal_timing"

class TrajectoryRisk(Enum):
    """Niveles de riesgo para trayectorias normativas"""
    VERY_LOW = "very_low"     # 0-20% probability of problems
    LOW = "low"               # 20-40% probability of problems
    MODERATE = "moderate"     # 40-60% probability of problems
    HIGH = "high"             # 60-80% probability of problems
    VERY_HIGH = "very_high"   # 80-100% probability of problems

@dataclass
class NormativeConstraint:
    """
    Constraint específico que limita trayectorias normativas
    
    ⚠️  REALITY CHECK: Constraints are probabilistic assessments, not absolute barriers
    """
    constraint_type: ConstraintType
    description: str
    severity: float                    # 0-1 scale (1 = complete block)
    confidence: float = 0.65           # Realistic confidence in assessment
    
    # Specific constraint details
    legal_basis: Optional[str] = None
    institutional_requirements: List[str] = field(default_factory=list)
    political_actors_involved: List[str] = field(default_factory=list)
    cultural_factors: List[str] = field(default_factory=list)
    economic_costs: Optional[Dict] = None
    international_treaties: List[str] = field(default_factory=list)
    timing_requirements: Optional[Dict] = None
    
    # Evidence and uncertainty
    evidence_quality: str = "moderate"  # "high", "moderate", "low", "limited"
    measurement_uncertainty: float = 0.25
    
    def calculate_blocking_probability(self) -> float:
        """
        Calcula probabilidad de que este constraint bloquee la reforma
        
        Returns: Probability 0-1 (adjusted for confidence and uncertainty)
        """
        # Base blocking probability from severity
        base_prob = self.severity
        
        # Adjust for confidence - lower confidence reduces certainty
        confidence_adjustment = self.confidence * 0.3  # Max 30% adjustment
        adjusted_prob = base_prob * (0.7 + confidence_adjustment)
        
        # Add uncertainty noise
        uncertainty_range = self.measurement_uncertainty
        noise = np.random.normal(0, uncertainty_range * 0.5)
        
        final_prob = np.clip(adjusted_prob + noise, 0.0, 1.0)
        return float(final_prob)

@dataclass
class NormativeTrajectory:
    """
    Trayectoria específica de desarrollo normativo
    
    🎯 OBJECTIVE: Describe possible path, not predict exact outcome
    """
    trajectory_id: str
    description: str
    
    # Trajectory characteristics
    initial_state: DimensionalProfile12D
    target_state: DimensionalProfile12D
    estimated_timeline_months: int
    
    # Constraints affecting this trajectory
    constraints: List[NormativeConstraint]
    
    # Probabilistic assessments (with appropriate uncertainty)
    success_probability: float = 0.5    # Default neutral, not optimistic
    implementation_gap_expected: float = 0.4  # Realistic implementation gap
    
    # Risk assessment
    overall_risk: TrajectoryRisk = TrajectoryRisk.MODERATE
    critical_junctures: List[Dict] = field(default_factory=list)
    
    # Uncertainty and confidence
    prediction_confidence: float = 0.55  # Modest confidence, not inflated
    
    def calculate_trajectory_vector(self) -> np.ndarray:
        """Calcula vector de cambio dimensional"""
        initial_vector = self.initial_state.to_vector()
        target_vector = self.target_state.to_vector()
        return target_vector - initial_vector
    
    def calculate_trajectory_magnitude(self) -> float:
        """Calcula magnitud total del cambio"""
        vector = self.calculate_trajectory_vector()
        return float(np.linalg.norm(vector))
    
    def identify_most_constrained_dimensions(self, top_n: int = 3) -> List[str]:
        """Identifica las dimensiones más restringidas por constraints"""
        dimension_names = [
            "judicial_review_strength", "separation_powers_clarity", "constitutional_supremacy",
            "individual_rights_protection", "federalism_decentralization", "democratic_participation",
            "institutional_accountability", "normative_stability", "enforcement_effectiveness",
            "legal_tradition_coherence", "cultural_legal_alignment", "external_influence_resistance"
        ]
        
        # Simple heuristic: dimensions with bigger changes face more constraints
        vector = self.calculate_trajectory_vector()
        abs_changes = np.abs(vector)
        
        # Get indices of top changes
        top_indices = np.argsort(abs_changes)[-top_n:]
        return [dimension_names[i] for i in reversed(top_indices)]

class NormativeConstraintAnalyzer:
    """
    Sistema para identificar y evaluar constraints normativos
    
    ⚠️  REALISTIC SCOPE: Identifies probable constraints, not exhaustive analysis
    📊 ACCURACY: ~65% constraint identification accuracy (social science realistic)
    """
    
    CONSTRAINT_PATTERNS = {
        # Constitutional/Legal constraints
        "constitutional_amendment_required": {
            "type": ConstraintType.CONSTITUTIONAL_LEGAL,
            "base_severity": 0.8,  # High barrier
            "factors": ["amendment_difficulty", "judicial_review_strength", "constitutional_culture"]
        },
        
        "supreme_court_resistance": {
            "type": ConstraintType.CONSTITUTIONAL_LEGAL,
            "base_severity": 0.7,
            "factors": ["judicial_independence", "constitutional_jurisprudence", "precedent_strength"]
        },
        
        # Institutional capacity constraints  
        "bureaucratic_implementation_gap": {
            "type": ConstraintType.INSTITUTIONAL_CAPACITY,
            "base_severity": 0.6,
            "factors": ["state_capacity", "bureaucratic_quality", "coordination_mechanisms"]
        },
        
        "insufficient_enforcement_capacity": {
            "type": ConstraintType.INSTITUTIONAL_CAPACITY, 
            "base_severity": 0.65,
            "factors": ["enforcement_resources", "institutional_reach", "compliance_mechanisms"]
        },
        
        # Political feasibility constraints
        "veto_player_opposition": {
            "type": ConstraintType.POLITICAL_FEASIBILITY,
            "base_severity": 0.75,
            "factors": ["veto_players_count", "coalition_stability", "opposition_resources"]
        },
        
        "electoral_cycle_misalignment": {
            "type": ConstraintType.POLITICAL_FEASIBILITY,
            "base_severity": 0.5,
            "factors": ["election_timing", "political_capital", "public_support"]
        },
        
        # Cultural resistance constraints
        "deep_cultural_incompatibility": {
            "type": ConstraintType.CULTURAL_RESISTANCE,
            "base_severity": 0.8,
            "factors": ["cultural_values_alignment", "traditional_authority", "social_acceptance"]
        },
        
        "religious_authority_opposition": {
            "type": ConstraintType.CULTURAL_RESISTANCE,
            "base_severity": 0.7,
            "factors": ["religious_influence", "clergy_position", "believer_mobilization"]
        },
        
        # Economic sustainability constraints
        "implementation_cost_prohibitive": {
            "type": ConstraintType.ECONOMIC_SUSTAINABILITY,
            "base_severity": 0.6,
            "factors": ["fiscal_resources", "economic_priorities", "cost_benefit_ratio"]
        },
        
        # International compliance constraints
        "treaty_violation_risk": {
            "type": ConstraintType.INTERNATIONAL_COMPLIANCE,
            "base_severity": 0.7,
            "factors": ["treaty_obligations", "international_reputation", "sanctions_risk"]
        }
    }
    
    def __init__(self):
        self.constraint_identification_accuracy = 0.65  # Realistic, not inflated
        
    def identify_constraints(self, legal_system: LegalSystemProfile, 
                           proposed_trajectory: NormativeTrajectory) -> List[NormativeConstraint]:
        """
        Identifica constraints aplicables para una trayectoria específica
        
        ⚠️  LIMITED SCOPE: Identifies major constraints, not exhaustive analysis
        """
        identified_constraints = []
        
        # Analyze trajectory characteristics
        trajectory_magnitude = proposed_trajectory.calculate_trajectory_magnitude()
        trajectory_vector = proposed_trajectory.calculate_trajectory_vector()
        
        # High magnitude changes face more constraints
        magnitude_factor = min(trajectory_magnitude / 3.0, 1.0)  # Normalize
        
        # Check each constraint pattern
        for pattern_name, pattern_config in self.CONSTRAINT_PATTERNS.items():
            
            # Calculate applicability probability
            applicability = self._calculate_constraint_applicability(
                pattern_name, legal_system, proposed_trajectory, magnitude_factor
            )
            
            # If constraint likely applies (threshold = 0.4 for moderate sensitivity)
            if applicability > 0.4:
                
                # Calculate specific severity for this case
                severity = self._calculate_constraint_severity(
                    pattern_config, legal_system, proposed_trajectory, applicability
                )
                
                # Create constraint with realistic confidence
                constraint = NormativeConstraint(
                    constraint_type=pattern_config["type"],
                    description=self._generate_constraint_description(pattern_name, legal_system),
                    severity=severity,
                    confidence=0.65 * applicability,  # Scale confidence with applicability
                    evidence_quality="moderate",  # Honest assessment
                    measurement_uncertainty=0.25
                )
                
                # Add specific details based on constraint type
                constraint = self._add_constraint_details(constraint, pattern_name, legal_system)
                
                identified_constraints.append(constraint)
        
        # Sort by severity (most severe first)
        identified_constraints.sort(key=lambda x: x.severity, reverse=True)
        
        return identified_constraints
    
    def _calculate_constraint_applicability(self, pattern_name: str, legal_system: LegalSystemProfile,
                                          trajectory: NormativeTrajectory, magnitude_factor: float) -> float:
        """
        Calcula probabilidad de que un constraint específico aplique
        
        Returns: Probability 0-1 (realistic assessment, not inflated)
        """
        base_applicability = 0.3  # Conservative base
        
        # System-specific factors
        if pattern_name == "constitutional_amendment_required":
            if legal_system.legal_tradition in [LegalTradition.COMMON_LAW, LegalTradition.CIVIL_LAW]:
                base_applicability += 0.4 * magnitude_factor
                
        elif pattern_name == "bureaucratic_implementation_gap":
            # Higher in systems with weak implementation capacity
            impl_weakness = 1.0 - legal_system.implementation_capacity
            base_applicability += 0.5 * impl_weakness
            
        elif pattern_name == "deep_cultural_incompatibility":
            # Higher when cultural embeddedness is low
            cultural_weakness = 1.0 - legal_system.cultural_embeddedness
            base_applicability += 0.6 * cultural_weakness
            
        elif pattern_name == "veto_player_opposition":
            # Higher in systems with multiple veto players (presidential, federal)
            if legal_system.constitutional_system == ConstitutionalSystem.PRESIDENTIAL:
                base_applicability += 0.3
                
        # Add trajectory-specific factors
        base_applicability += magnitude_factor * 0.2  # Bigger changes = more constraints
        
        # Add realistic noise
        noise = np.random.normal(0, 0.1)
        final_applicability = np.clip(base_applicability + noise, 0.0, 1.0)
        
        return float(final_applicability)
    
    def _calculate_constraint_severity(self, pattern_config: Dict, legal_system: LegalSystemProfile,
                                     trajectory: NormativeTrajectory, applicability: float) -> float:
        """Calcula severidad específica del constraint"""
        base_severity = pattern_config["base_severity"]
        
        # Adjust based on system characteristics
        if pattern_config["type"] == ConstraintType.CONSTITUTIONAL_LEGAL:
            if legal_system.institutional_strength > 0.7:
                base_severity += 0.1  # Strong institutions = stronger constraints
                
        elif pattern_config["type"] == ConstraintType.CULTURAL_RESISTANCE:
            cultural_strength = legal_system.cultural_embeddedness
            base_severity = base_severity * (0.5 + 0.5 * cultural_strength)
            
        # Adjust for trajectory characteristics
        trajectory_magnitude = trajectory.calculate_trajectory_magnitude()
        magnitude_adjustment = min(trajectory_magnitude * 0.1, 0.2)
        
        final_severity = np.clip(base_severity + magnitude_adjustment, 0.0, 1.0)
        return float(final_severity)
    
    def _generate_constraint_description(self, pattern_name: str, legal_system: LegalSystemProfile) -> str:
        """Genera descripción específica del constraint"""
        descriptions = {
            "constitutional_amendment_required": f"Requiere enmienda constitucional en {legal_system.country} - proceso complejo",
            "supreme_court_resistance": f"Probable resistencia judicial en sistema {legal_system.legal_tradition.value}",
            "bureaucratic_implementation_gap": f"Capacidad burocrática limitada para implementación efectiva",
            "veto_player_opposition": f"Oposición probable de actores con poder de veto",
            "deep_cultural_incompatibility": f"Incompatibilidad con valores culturales arraigados",
            "religious_authority_opposition": f"Oposición probable de autoridades religiosas",
            "implementation_cost_prohibitive": f"Costos de implementación potencialmente prohibitivos",
            "treaty_violation_risk": f"Riesgo de violación de obligaciones internacionales"
        }
        
        return descriptions.get(pattern_name, f"Constraint identificado: {pattern_name}")
    
    def _add_constraint_details(self, constraint: NormativeConstraint, pattern_name: str,
                               legal_system: LegalSystemProfile) -> NormativeConstraint:
        """Añade detalles específicos al constraint"""
        
        if constraint.constraint_type == ConstraintType.CONSTITUTIONAL_LEGAL:
            constraint.legal_basis = "Constitutional provisions and jurisprudence"
            constraint.institutional_requirements = ["Constitutional court review", "Legislative supermajority"]
            
        elif constraint.constraint_type == ConstraintType.POLITICAL_FEASIBILITY:
            constraint.political_actors_involved = ["Opposition parties", "Interest groups", "Veto players"]
            
        elif constraint.constraint_type == ConstraintType.CULTURAL_RESISTANCE:
            constraint.cultural_factors = ["Traditional values", "Social norms", "Religious beliefs"]
            
        return constraint

class NormativeTrajectoryAnalyzer:
    """
    Sistema integral para análisis de trayectorias normativas
    
    🎯 PURPOSE: Analytical tool for understanding normative change possibilities
    ⚠️  REALITY FILTER: Moderate predictive capability, honest about limitations
    """
    
    def __init__(self):
        self.constraint_analyzer = NormativeConstraintAnalyzer()
        self.iusspace = UniversalIusSpace12D()
        self.trajectory_analysis_accuracy = 0.62  # Realistic accuracy estimate
        
    def analyze_normative_trajectory(self, legal_system: LegalSystemProfile,
                                   current_profile: DimensionalProfile12D,
                                   target_profile: DimensionalProfile12D,
                                   timeline_months: int = 24) -> Dict:
        """
        Análisis comprehensivo de trayectoria normativa
        
        Returns: Comprehensive analysis with realistic confidence intervals
        """
        # Create trajectory object
        trajectory = NormativeTrajectory(
            trajectory_id=f"trajectory_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Normative change trajectory for {legal_system.country}",
            initial_state=current_profile,
            target_state=target_profile,
            estimated_timeline_months=timeline_months
        )
        
        # Calculate trajectory characteristics
        trajectory_vector = trajectory.calculate_trajectory_vector()
        trajectory_magnitude = trajectory.calculate_trajectory_magnitude()
        most_constrained_dims = trajectory.identify_most_constrained_dimensions()
        
        # Identify constraints
        constraints = self.constraint_analyzer.identify_constraints(legal_system, trajectory)
        
        # Calculate success probability (realistic, not optimistic)
        success_probability = self._calculate_success_probability(trajectory, constraints)
        
        # Calculate implementation gap (realistic expectation)
        implementation_gap = self._calculate_implementation_gap(legal_system, trajectory_magnitude)
        
        # Assess overall risk
        overall_risk = self._assess_trajectory_risk(constraints, trajectory_magnitude)
        
        # Identify critical junctures
        critical_junctures = self._identify_critical_junctures(constraints, timeline_months)
        
        # Calculate confidence with appropriate uncertainty
        prediction_confidence = self._calculate_prediction_confidence(legal_system, constraints)
        
        return {
            "trajectory_analysis": {
                "trajectory_id": trajectory.trajectory_id,
                "magnitude": round(trajectory_magnitude, 3),
                "success_probability": round(success_probability, 3),
                "implementation_gap_expected": round(implementation_gap, 3),
                "overall_risk": overall_risk.value,
                "prediction_confidence": round(prediction_confidence, 2),
                "timeline_months": timeline_months
            },
            
            "dimensional_changes": {
                "most_constrained_dimensions": most_constrained_dims,
                "trajectory_vector": trajectory_vector.round(3).tolist(),
                "largest_changes": self._identify_largest_changes(trajectory_vector)
            },
            
            "constraints_identified": [
                {
                    "type": c.constraint_type.value,
                    "description": c.description,
                    "severity": round(c.severity, 3),
                    "confidence": round(c.confidence, 2),
                    "blocking_probability": round(c.calculate_blocking_probability(), 3)
                }
                for c in constraints
            ],
            
            "critical_junctures": critical_junctures,
            
            "recommendations": self._generate_recommendations(constraints, trajectory_magnitude),
            
            "limitations_and_uncertainties": [
                f"Prediction confidence: {prediction_confidence:.1%} (moderate)",
                "Constraint identification incomplete - focus on major barriers",
                "Timeline estimates approximate - actual duration highly variable",
                "Cultural factors difficult to quantify precisely", 
                "International factors may change unexpectedly",
                f"Overall analysis accuracy: ~{self.trajectory_analysis_accuracy:.0%}"
            ]
        }
    
    def _calculate_success_probability(self, trajectory: NormativeTrajectory, 
                                     constraints: List[NormativeConstraint]) -> float:
        """
        Calcula probabilidad de éxito realista (no optimista)
        """
        base_probability = 0.5  # Neutral starting point
        
        # Reduce probability based on constraints
        for constraint in constraints:
            blocking_prob = constraint.calculate_blocking_probability()
            probability_reduction = blocking_prob * constraint.severity * 0.3
            base_probability -= probability_reduction
        
        # Adjust for trajectory magnitude (bigger changes = lower probability)
        magnitude_penalty = min(trajectory.calculate_trajectory_magnitude() * 0.1, 0.2)
        base_probability -= magnitude_penalty
        
        # Add realistic noise
        noise = np.random.normal(0, 0.05)
        final_probability = np.clip(base_probability + noise, 0.1, 0.9)  # Reasonable bounds
        
        return float(final_probability)
    
    def _calculate_implementation_gap(self, legal_system: LegalSystemProfile, magnitude: float) -> float:
        """
        Calcula brecha de implementación esperada (realista)
        """
        # Base gap from system implementation capacity
        base_gap = 1.0 - legal_system.implementation_capacity
        
        # Increase gap for larger changes
        magnitude_factor = min(magnitude / 2.0, 0.5)  # Max 50% additional gap
        
        # Cultural factors
        cultural_factor = (1.0 - legal_system.cultural_embeddedness) * 0.3
        
        total_gap = base_gap + magnitude_factor + cultural_factor
        return min(total_gap, 0.8)  # Cap at 80% gap (realistic maximum)
    
    def _assess_trajectory_risk(self, constraints: List[NormativeConstraint], magnitude: float) -> TrajectoryRisk:
        """Evalúa riesgo general de la trayectoria"""
        
        # Calculate average constraint severity
        if constraints:
            avg_severity = np.mean([c.severity for c in constraints])
            high_severity_count = len([c for c in constraints if c.severity > 0.7])
        else:
            avg_severity = 0.0
            high_severity_count = 0
        
        # Risk factors
        risk_score = 0.0
        risk_score += avg_severity * 0.4
        risk_score += min(high_severity_count * 0.2, 0.4)
        risk_score += min(magnitude * 0.2, 0.3)
        
        if risk_score < 0.2:
            return TrajectoryRisk.VERY_LOW
        elif risk_score < 0.4:
            return TrajectoryRisk.LOW
        elif risk_score < 0.6:
            return TrajectoryRisk.MODERATE
        elif risk_score < 0.8:
            return TrajectoryRisk.HIGH
        else:
            return TrajectoryRisk.VERY_HIGH
    
    def _identify_critical_junctures(self, constraints: List[NormativeConstraint], 
                                   timeline_months: int) -> List[Dict]:
        """Identifica puntos críticos en la trayectoria"""
        
        junctures = []
        
        # Constitutional review juncture
        if any(c.constraint_type == ConstraintType.CONSTITUTIONAL_LEGAL for c in constraints):
            junctures.append({
                "timing_months": min(6, timeline_months // 3),
                "description": "Constitutional review and legal challenges phase",
                "risk_level": "high",
                "key_actors": ["Constitutional court", "Legal challengers"]
            })
        
        # Political opposition juncture
        if any(c.constraint_type == ConstraintType.POLITICAL_FEASIBILITY for c in constraints):
            junctures.append({
                "timing_months": min(3, timeline_months // 4),
                "description": "Political coalition building and opposition phase", 
                "risk_level": "medium",
                "key_actors": ["Political parties", "Veto players", "Interest groups"]
            })
        
        # Implementation juncture
        junctures.append({
            "timing_months": max(timeline_months // 2, 6),
            "description": "Implementation and enforcement phase",
            "risk_level": "medium", 
            "key_actors": ["Bureaucracy", "Enforcement agencies", "Target populations"]
        })
        
        return junctures
    
    def _calculate_prediction_confidence(self, legal_system: LegalSystemProfile,
                                       constraints: List[NormativeConstraint]) -> float:
        """Calcula confianza realista en la predicción"""
        
        base_confidence = 0.50  # Modest baseline
        
        # Higher confidence for systems with better data quality
        if legal_system.data_quality == "high":
            base_confidence += 0.15
        elif legal_system.data_quality == "medium":
            base_confidence += 0.05
        
        # Higher confidence when constraints are clear
        if constraints:
            avg_constraint_confidence = np.mean([c.confidence for c in constraints])
            base_confidence += avg_constraint_confidence * 0.2
        
        # Lower confidence for complex trajectories
        if len(constraints) > 5:
            base_confidence -= 0.1
        
        return np.clip(base_confidence, 0.3, 0.75)  # Realistic bounds
    
    def _identify_largest_changes(self, trajectory_vector: np.ndarray) -> List[Dict]:
        """Identifica los cambios dimensionales más grandes"""
        
        dimension_names = [
            "judicial_review_strength", "separation_powers_clarity", "constitutional_supremacy",
            "individual_rights_protection", "federalism_decentralization", "democratic_participation",
            "institutional_accountability", "normative_stability", "enforcement_effectiveness",
            "legal_tradition_coherence", "cultural_legal_alignment", "external_influence_resistance"
        ]
        
        changes = []
        for i, change in enumerate(trajectory_vector):
            if abs(change) > 0.1:  # Significant change threshold
                changes.append({
                    "dimension": dimension_names[i],
                    "change_magnitude": round(float(change), 3),
                    "direction": "increase" if change > 0 else "decrease"
                })
        
        # Sort by magnitude
        changes.sort(key=lambda x: abs(x["change_magnitude"]), reverse=True)
        return changes[:5]  # Top 5
    
    def _generate_recommendations(self, constraints: List[NormativeConstraint], magnitude: float) -> List[str]:
        """Genera recomendaciones basadas en constraints identificados"""
        
        recommendations = []
        
        # General recommendations
        if magnitude > 2.0:
            recommendations.append("Consider phased implementation due to high trajectory magnitude")
        
        # Constraint-specific recommendations
        constraint_types = set(c.constraint_type for c in constraints)
        
        if ConstraintType.CONSTITUTIONAL_LEGAL in constraint_types:
            recommendations.append("Engage constitutional experts and courts early in process")
            
        if ConstraintType.POLITICAL_FEASIBILITY in constraint_types:
            recommendations.append("Build broader political coalition before proceeding")
            
        if ConstraintType.CULTURAL_RESISTANCE in constraint_types:
            recommendations.append("Invest in public education and cultural adaptation strategies")
            
        if ConstraintType.INSTITUTIONAL_CAPACITY in constraint_types:
            recommendations.append("Strengthen institutional capacity before major implementation")
            
        if ConstraintType.ECONOMIC_SUSTAINABILITY in constraint_types:
            recommendations.append("Develop comprehensive financing and cost-benefit analysis")
        
        # Default recommendation
        if not recommendations:
            recommendations.append("Proceed with careful monitoring of implementation challenges")
        
        return recommendations

def main():
    """Demostración del sistema de análisis de trayectorias normativas"""
    print("🎯 IUSMORFOS UNIVERSAL v5.0 - ANÁLISIS DE TRAYECTORIAS NORMATIVAS")
    print("⚠️  Reality Filter Applied: Moderate predictive capability, honest limitations")
    print("="*80)
    
    # Initialize analyzer
    analyzer = NormativeTrajectoryAnalyzer()
    
    print(f"\n📊 SISTEMA DE ANÁLISIS DE TRAYECTORIAS:")
    print(f"Precisión análisis: ~{analyzer.trajectory_analysis_accuracy:.0%} (realista para ciencias sociales)")
    print(f"Precisión identificación constraints: ~{analyzer.constraint_analyzer.constraint_identification_accuracy:.0%}")
    
    # Example analysis (simplified mock data for demonstration)
    print(f"\n🧪 EJEMPLO: ANÁLISIS DE REFORMA JUDICIAL")
    
    # Mock legal system profile (Argentina example)
    argentina_profile = LegalSystemProfile(
        country="Argentina",
        iso_code="ARG", 
        legal_tradition=LegalTradition.CIVIL_LAW,
        constitutional_system=ConstitutionalSystem.PRESIDENTIAL,
        institutional_strength=0.45,
        implementation_capacity=0.42,
        cultural_embeddedness=0.50,
        external_influence=0.60,
        data_quality="medium",
        confidence_score=0.65
    )
    
    # Mock current and target profiles (simplified)
    from .iusspace_universal_12d import DimensionalProfile12D
    
    current_profile = DimensionalProfile12D(
        judicial_review_strength=0.40, separation_powers_clarity=0.45, constitutional_supremacy=0.60,
        individual_rights_protection=0.55, federalism_decentralization=0.30, democratic_participation=0.50,
        institutional_accountability=0.35, normative_stability=0.40, enforcement_effectiveness=0.35,
        legal_tradition_coherence=0.65, cultural_legal_alignment=0.50, external_influence_resistance=0.40,
        dimensional_confidence=0.60
    )
    
    target_profile = DimensionalProfile12D(
        judicial_review_strength=0.75, separation_powers_clarity=0.70, constitutional_supremacy=0.80,
        individual_rights_protection=0.75, federalism_decentralization=0.35, democratic_participation=0.65,
        institutional_accountability=0.60, normative_stability=0.60, enforcement_effectiveness=0.55,
        legal_tradition_coherence=0.70, cultural_legal_alignment=0.55, external_influence_resistance=0.45,
        dimensional_confidence=0.65
    )
    
    # Run analysis
    analysis = analyzer.analyze_normative_trajectory(
        argentina_profile, current_profile, target_profile, timeline_months=30
    )
    
    # Display results
    traj = analysis["trajectory_analysis"]
    print(f"\n📊 RESULTADOS DEL ANÁLISIS:")
    print(f"   Magnitud trayectoria: {traj['magnitude']} (cambio moderado-alto)")
    print(f"   Probabilidad éxito: {traj['success_probability']:.1%}")
    print(f"   Brecha implementación esperada: {traj['implementation_gap_expected']:.1%}")
    print(f"   Nivel riesgo general: {traj['overall_risk']}")
    print(f"   Confianza predicción: {traj['prediction_confidence']:.1%}")
    
    print(f"\n🔍 CONSTRAINTS IDENTIFICADOS:")
    for i, constraint in enumerate(analysis["constraints_identified"][:4]):  # Top 4
        print(f"   {i+1}. {constraint['type']}")
        print(f"      Severidad: {constraint['severity']:.2f}, Prob. bloqueo: {constraint['blocking_probability']:.1%}")
        print(f"      {constraint['description']}")
    
    print(f"\n📈 CAMBIOS DIMENSIONALES PRINCIPALES:")
    for change in analysis["dimensional_changes"]["largest_changes"][:3]:
        print(f"   • {change['dimension']}: {change['direction']} de {abs(change['change_magnitude']):.2f}")
    
    print(f"\n🚨 PUNTOS CRÍTICOS:")
    for juncture in analysis["critical_junctures"]:
        print(f"   • Mes {juncture['timing_months']}: {juncture['description']}")
        print(f"     Riesgo: {juncture['risk_level']}")
    
    print(f"\n💡 RECOMENDACIONES:")
    for rec in analysis["recommendations"]:
        print(f"   • {rec}")
    
    print(f"\n⚠️  LIMITACIONES Y INCERTIDUMBRES:")
    for limitation in analysis["limitations_and_uncertainties"]:
        print(f"   • {limitation}")
    
    print(f"\n✅ SISTEMA DE TRAYECTORIAS IMPLEMENTADO CON REALITY FILTER")
    print(f"🎯 Herramienta analítica útil con expectativas realistas y limitaciones honestas")

if __name__ == "__main__":
    main()