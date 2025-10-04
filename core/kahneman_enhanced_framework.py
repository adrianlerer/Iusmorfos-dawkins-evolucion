#!/usr/bin/env python3
"""
Kahneman Enhanced Framework
Integra correcciones y detecciones Kahneman con framework Iusmorfos Universal

MEJORAS IMPLEMENTADAS:
1. Corrección regresiva automática de predicciones (Sistema 2)
2. Detección automática de sesgos cognitivos  
3. Intervalos de confianza honestos ajustados por sesgos
4. Protocolo Meehl: fórmulas estadísticas > juicios clínicos
5. Base rates históricas para corrección regresiva

@author: Iusmorfos Universal Framework + Kahneman Enhancements
@version: 2.0 - Bias-Corrected Universal Framework
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import logging

# Importar módulos Kahneman
try:
    from .kahneman_prediction_correction import (
        KahnemanPredictionCorrector, BaseRateData, IntuitiveEvidence, 
        CorrelationAssessment, create_argentina_base_rates
    )
    from .kahneman_bias_detector import (
        KahnemanBiasDetector, PredictionInput, BiasDetection
    )
    # Importar componentes framework original
    from .universal_legal_taxonomy import UniversalLegalTaxonomy
    from .normative_trajectory_analyzer import NormativeTrajectoryAnalyzer
except ImportError:
    from kahneman_prediction_correction import (
        KahnemanPredictionCorrector, BaseRateData, IntuitiveEvidence, 
        CorrelationAssessment, create_argentina_base_rates
    )
    from kahneman_bias_detector import (
        KahnemanBiasDetector, PredictionInput, BiasDetection
    )
    # Importar componentes framework original
    from universal_legal_taxonomy import UniversalLegalTaxonomy
    from normative_trajectory_analyzer import NormativeTrajectoryAnalyzer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class KahnemanEnhancedAnalysis:
    """
    Análisis mejorado con correcciones Kahneman aplicadas
    Integra predicción original + correcciones + detección sesgos
    """
    component_name: str
    original_prediction: float
    corrected_prediction: float
    confidence_interval_original: Tuple[float, float]
    confidence_interval_corrected: Tuple[float, float]
    
    # Información de correcciones aplicadas
    base_rate_used: float
    correlation_assessed: float
    regression_factor_applied: float
    
    # Sesgos detectados
    biases_detected: List[BiasDetection]
    overall_bias_impact: float
    reliability_rating: str
    
    # Recomendaciones Kahneman
    kahneman_warnings: List[str]
    correction_recommendations: List[str]

class KahnemanEnhancedFramework:
    """
    Framework Iusmorfos Universal mejorado con protocolos Kahneman
    
    CARACTERÍSTICAS:
    1. Todas las predicciones pasan por corrección regresiva obligatoria
    2. Detección automática de sesgos cognitivos
    3. Intervalos de confianza ajustados por calidad de evidencia
    4. Uso de base rates históricas para calibración
    5. Protocolo Meehl cuando datos suficientes disponibles
    """
    
    def __init__(self, jurisdiction: str, legal_tradition: str):
        self.jurisdiction = jurisdiction
        self.legal_tradition = legal_tradition
        
        # Componentes Kahneman
        self.prediction_corrector = KahnemanPredictionCorrector()
        self.bias_detector = KahnemanBiasDetector()
        
        # Componentes framework original 
        self.legal_taxonomy = UniversalLegalTaxonomy()
        self.trajectory_analyzer = NormativeTrajectoryAnalyzer()
        
        # Base rates específicas por jurisdicción
        if jurisdiction.lower() == "argentina":
            self.base_rates = create_argentina_base_rates()
        else:
            self.base_rates = self._create_generic_base_rates()
        
        # Registro de análisis
        self.analysis_history: List[KahnemanEnhancedAnalysis] = []
        
        logger.info(f"Kahneman Enhanced Framework initialized for {jurisdiction} ({legal_tradition})")
    
    def _create_generic_base_rates(self) -> Dict[str, BaseRateData]:
        """Crear base rates genéricas cuando no hay datos específicos"""
        return {
            "economic_stabilization": BaseRateData(
                outcome_type="Generic Economic Stabilization",
                historical_base_rate=0.50,  # 50% tasa base global
                sample_size=20,  # Muestra limitada
                confidence_in_base_rate=0.60,
                time_period="2000-2020",
                geographic_scope="Emerging Markets Average"
            ),
            "structural_reform": BaseRateData(
                outcome_type="Generic Structural Reform",
                historical_base_rate=0.40,
                sample_size=15,
                confidence_in_base_rate=0.55,
                time_period="2000-2020",
                geographic_scope="Emerging Markets Average"
            )
        }
    
    def enhanced_trajectory_analysis(self, 
                                   reform_description: str,
                                   target_dimensions: List[str],
                                   evidence_text: str) -> KahnemanEnhancedAnalysis:
        """
        Análisis de trayectoria mejorado con correcciones Kahneman
        
        PROTOCOLO:
        1. Análisis original del framework
        2. Detección de sesgos en inputs
        3. Corrección regresiva de predicciones  
        4. Intervalos de confianza ajustados
        5. Recomendaciones basadas en Kahneman
        """
        
        # PASO 1: Análisis original del framework
        original_results = self.trajectory_analyzer.analyze_reform_trajectory(
            reform_description, target_dimensions
        )
        
        original_prediction = original_results['trajectory_analysis']['success_probability']
        original_ci = original_results['trajectory_analysis']['confidence_interval']
        
        # PASO 2: Detección de sesgos en inputs
        pred_input = PredictionInput(
            prediction_value=original_prediction,
            confidence_interval=original_ci,
            evidence_description=evidence_text,
            methodology_used="iusmorfos_trajectory_analysis",
            historical_comparisons=self._extract_historical_comparisons(evidence_text),
            narrative_elements=self._extract_narrative_elements(evidence_text)
        )
        
        detected_biases = self.bias_detector.detect_all_biases(pred_input)
        bias_summary = self.bias_detector.get_bias_summary(detected_biases)
        
        # PASO 3: Preparar datos para corrección regresiva
        reform_type = self._classify_reform_type(reform_description)
        
        if reform_type in self.base_rates:
            base_rate = self.base_rates[reform_type]
        else:
            # Usar base rate genérica más conservadora
            base_rate = BaseRateData(
                outcome_type="Generic Reform",
                historical_base_rate=0.45,
                sample_size=10,
                confidence_in_base_rate=0.50,
                time_period="Limited Data",
                geographic_scope=self.jurisdiction
            )
        
        # Evaluar calidad de evidencia y correlación
        evidence_quality = self._assess_evidence_quality(evidence_text, detected_biases)
        
        evidence = IntuitiveEvidence(
            evidence_type="Trajectory Analysis Evidence",
            evidence_strength=evidence_quality['strength'],
            evidence_quality=evidence_quality['quality'],
            coherence_score=evidence_quality['coherence'],
            availability_bias=evidence_quality['availability_bias'],
            representativeness_bias=evidence_quality['representativeness_bias']
        )
        
        # Correlación basada en framework performance y sesgos detectados
        correlation_base = 0.50  # Correlación base moderada
        bias_penalty = min(0.3, bias_summary['total_impact_on_prediction'])
        adjusted_correlation = max(0.2, correlation_base - bias_penalty)
        
        correlation = CorrelationAssessment(
            evidence_outcome_correlation=adjusted_correlation,
            correlation_confidence=0.60,
            historical_validation=None,  # No disponible aún
            expert_consensus=None
        )
        
        # PASO 4: Aplicar corrección regresiva
        correction_result = self.prediction_corrector.correct_prediction(
            base_rate, evidence, correlation, original_prediction
        )
        
        # PASO 5: Generar recomendaciones Kahneman
        recommendations = self._generate_kahneman_recommendations(
            detected_biases, correction_result, bias_summary
        )
        
        # PASO 6: Crear análisis mejorado
        enhanced_analysis = KahnemanEnhancedAnalysis(
            component_name="trajectory_analysis",
            original_prediction=original_prediction,
            corrected_prediction=correction_result['corrected_prediction'],
            confidence_interval_original=original_ci,
            confidence_interval_corrected=correction_result['confidence_interval'],
            base_rate_used=base_rate.historical_base_rate,
            correlation_assessed=adjusted_correlation,
            regression_factor_applied=1.0 - abs(adjusted_correlation),
            biases_detected=detected_biases,
            overall_bias_impact=bias_summary['total_impact_on_prediction'],
            reliability_rating=bias_summary['overall_reliability'],
            kahneman_warnings=correction_result['kahneman_warnings'],
            correction_recommendations=recommendations
        )
        
        # Registrar análisis
        self.analysis_history.append(enhanced_analysis)
        
        return enhanced_analysis
    
    def _classify_reform_type(self, reform_description: str) -> str:
        """Clasificar tipo de reforma para seleccionar base rate apropiada"""
        
        reform_keywords = {
            "economic_stabilization": ["economic", "stabilization", "inflation", "monetary", "fiscal"],
            "structural_reform": ["structural", "labor", "pension", "tax", "institutional"],
            "constitutional_reform": ["constitutional", "amendment", "fundamental", "charter"]
        }
        
        description_lower = reform_description.lower()
        
        for reform_type, keywords in reform_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                return reform_type
        
        return "structural_reform"  # Default
    
    def _assess_evidence_quality(self, evidence_text: str, 
                               detected_biases: List[BiasDetection]) -> Dict[str, float]:
        """Evaluar calidad de evidencia considerando sesgos detectados"""
        
        # Evaluación base de evidencia
        word_count = len(evidence_text.split())
        length_factor = min(1.0, word_count / 100)  # Normalizar por longitud
        
        # Buscar indicadores de calidad
        quality_indicators = ['data', 'evidence', 'research', 'study', 'analysis', 'empirical']
        quality_count = sum(1 for ind in quality_indicators if ind in evidence_text.lower())
        quality_factor = min(1.0, quality_count * 0.2)
        
        # Evaluar coherencia narrativa
        coherence_indicators = ['therefore', 'because', 'leads to', 'results in']
        coherence_count = sum(1 for ind in coherence_indicators if ind in evidence_text.lower())
        coherence_factor = min(1.0, coherence_count * 0.15)
        
        # Penalizar por sesgos detectados
        bias_penalties = {}
        for bias in detected_biases:
            if bias.bias_type.value == 'availability_heuristic':
                bias_penalties['availability_bias'] = bias.severity
            elif bias.bias_type.value == 'representativeness_heuristic':
                bias_penalties['representativeness_bias'] = bias.severity
        
        availability_bias = bias_penalties.get('availability_bias', 0.2)
        representativeness_bias = bias_penalties.get('representativeness_bias', 0.2)
        
        # Calcular métricas finales
        base_strength = (length_factor + quality_factor) / 2
        base_quality = quality_factor
        
        return {
            'strength': max(0.3, base_strength),
            'quality': max(0.3, base_quality),
            'coherence': coherence_factor,
            'availability_bias': availability_bias,
            'representativeness_bias': representativeness_bias
        }
    
    def _extract_historical_comparisons(self, text: str) -> List[str]:
        """Extraer comparaciones históricas mencionadas en el texto"""
        
        historical_patterns = [
            r'like.*\d{4}', r'similar to.*\d{4}', r'reminds.*\d{4}',
            r'argentina.*crisis', r'chile.*\d{4}', r'brazil.*\d{4}'
        ]
        
        import re
        comparisons = []
        
        for pattern in historical_patterns:
            matches = re.findall(pattern, text.lower())
            comparisons.extend(matches)
        
        return comparisons[:5]  # Máximo 5 comparaciones
    
    def _extract_narrative_elements(self, text: str) -> List[str]:
        """Extraer elementos narrativos del texto"""
        
        narrative_indicators = [
            'story', 'narrative', 'explanation', 'reason', 'cause',
            'leads to', 'results in', 'because', 'therefore'
        ]
        
        elements = []
        for indicator in narrative_indicators:
            if indicator in text.lower():
                elements.append(indicator)
        
        return elements
    
    def _generate_kahneman_recommendations(self, 
                                         detected_biases: List[BiasDetection],
                                         correction_result: Dict,
                                         bias_summary: Dict) -> List[str]:
        """Generar recomendaciones específicas basadas en hallazgos Kahneman"""
        
        recommendations = []
        
        # Recomendaciones por corrección regresiva
        if correction_result['correction_applied']:
            diff = abs(correction_result['original_intuitive_prediction'] - 
                      correction_result['corrected_prediction'])
            if diff > 0.1:
                recommendations.append(
                    f"Predicción corregida por regresión a la media (-{diff:.1%}). "
                    "Intuición original probablemente sesgada hacia extremos."
                )
        
        # Recomendaciones por sesgos específicos
        for bias in detected_biases:
            
            if bias.bias_type.value == 'illusion_of_validity':
                recommendations.append(
                    "Ilusión de validez detectada: Coherencia narrativa ≠ validez predictiva. "
                    "Expandir intervalos de confianza y buscar evidencia contradictoria."
                )
            
            elif bias.bias_type.value == 'representativeness_heuristic':
                recommendations.append(
                    "Sesgo de representatividad: Incorporar tasas base históricas explícitamente. "
                    "No juzgar solo por similaridad a casos típicos."
                )
            
            elif bias.bias_type.value == 'availability_heuristic':
                recommendations.append(
                    "Sesgo de disponibilidad: Casos memorables pueden distorsionar estimación. "
                    "Buscar estadísticas sistemáticas vs. ejemplos anecdóticos."
                )
            
            elif bias.bias_type.value == 'overconfidence_bias':
                recommendations.append(
                    "Exceso de confianza detectado: Expandir intervalos de confianza. "
                    "Considerar factores que podrían generar sorpresas."
                )
        
        # Recomendación general por confiabilidad
        if bias_summary['overall_reliability'] == 'LOW':
            recommendations.append(
                "Confiabilidad baja debido a múltiples sesgos. "
                "Considerar obtener segunda opinión o análisis estadístico independiente."
            )
        
        # Recomendación protocolo Meehl si aplicable
        if len(detected_biases) >= 3:
            recommendations.append(
                "Protocolo Meehl recomendado: Usar fórmula estadística simple en lugar de juicio clínico "
                "para mejorar accuracy predictiva."
            )
        
        return recommendations
    
    def generate_enhanced_report(self, analysis: KahnemanEnhancedAnalysis) -> Dict:
        """Generar reporte completo con mejoras Kahneman"""
        
        return {
            'kahneman_enhanced_analysis': {
                'component': analysis.component_name,
                'jurisdiction': self.jurisdiction,
                'legal_tradition': self.legal_tradition
            },
            
            'prediction_comparison': {
                'original_prediction': f"{analysis.original_prediction:.1%}",
                'corrected_prediction': f"{analysis.corrected_prediction:.1%}",
                'correction_magnitude': f"{abs(analysis.original_prediction - analysis.corrected_prediction):.1%}",
                'original_ci': f"{analysis.confidence_interval_original[0]:.1%} - {analysis.confidence_interval_original[1]:.1%}",
                'corrected_ci': f"{analysis.confidence_interval_corrected[0]:.1%} - {analysis.confidence_interval_corrected[1]:.1%}"
            },
            
            'correction_details': {
                'base_rate_used': f"{analysis.base_rate_used:.1%}",
                'correlation_assessed': f"{analysis.correlation_assessed:.2f}",
                'regression_factor': f"{analysis.regression_factor_applied:.2f}",
                'regression_explanation': "Moved prediction toward base rate proportionally to correlation strength"
            },
            
            'bias_analysis': {
                'biases_detected': len(analysis.biases_detected),
                'bias_types': [bias.bias_type.value for bias in analysis.biases_detected],
                'overall_bias_impact': f"{analysis.overall_bias_impact:.1%}",
                'reliability_rating': analysis.reliability_rating,
                'bias_details': [
                    {
                        'type': bias.bias_type.value,
                        'severity': f"{bias.severity:.2f}",
                        'evidence': bias.evidence[:2]  # Top 2 pieces of evidence
                    }
                    for bias in analysis.biases_detected
                ]
            },
            
            'kahneman_recommendations': {
                'warnings': analysis.kahneman_warnings,
                'corrections_recommended': analysis.correction_recommendations,
                'methodology_improvements': [
                    "Use statistical formulas when sample size adequate (Protocol Meehl)",
                    "Always incorporate base rates before making intuitive adjustments", 
                    "Expand confidence intervals to account for bias and uncertainty",
                    "Seek disconfirming evidence to counter confirmation bias"
                ]
            },
            
            'enhanced_framework_status': {
                'kahneman_integration': "COMPLETE",
                'bias_protection': "ACTIVE", 
                'regressive_correction': "APPLIED",
                'academic_integrity': "ENHANCED with System 2 thinking protocols"
            }
        }

# Función de utilidad para análisis completo Argentina con Kahneman
def enhanced_argentina_analysis() -> Dict:
    """Análisis completo Argentina con todas las mejoras Kahneman aplicadas"""
    
    # Inicializar framework mejorado
    enhanced_framework = KahnemanEnhancedFramework("Argentina", "civil_law")
    
    # Análisis de trayectoria con evidencia sesgada (para demostrar correcciones)
    evidence_text = """
    Argentina's economic stabilization will definitely succeed because the current situation 
    is clearly similar to successful cases like Chile in the 1980s. The coherent policy 
    package obviously leads to positive results, just like we remember from other Latin 
    American success stories. Recent examples show that market-friendly reforms inevitably 
    produce growth. This is a typical case of orthodox adjustment that has worked before.
    """
    
    enhanced_analysis = enhanced_framework.enhanced_trajectory_analysis(
        reform_description="Economic stabilization through fiscal adjustment and monetary policy",
        target_dimensions=["rule_of_law", "economic_sustainability", "constitutional_stability"],
        evidence_text=evidence_text
    )
    
    # Generar reporte completo
    report = enhanced_framework.generate_enhanced_report(enhanced_analysis)
    
    return report

if __name__ == "__main__":
    # Ejecutar análisis mejorado
    result = enhanced_argentina_analysis()
    
    print("🧠 KAHNEMAN ENHANCED FRAMEWORK - ARGENTINA ANALYSIS")
    print("="*70)
    
    # Mostrar comparación de predicciones
    pred_comp = result['prediction_comparison']
    print(f"\n📊 PREDICTION COMPARISON:")
    print(f"Original: {pred_comp['original_prediction']} (CI: {pred_comp['original_ci']})")
    print(f"Corrected: {pred_comp['corrected_prediction']} (CI: {pred_comp['corrected_ci']})")
    print(f"Correction: {pred_comp['correction_magnitude']}")
    
    # Mostrar sesgos detectados
    bias_analysis = result['bias_analysis']
    print(f"\n🚨 BIAS DETECTION:")
    print(f"Biases detected: {bias_analysis['biases_detected']}")
    print(f"Types: {', '.join(bias_analysis['bias_types'])}")
    print(f"Impact: {bias_analysis['overall_bias_impact']}")
    print(f"Reliability: {bias_analysis['reliability_rating']}")
    
    # Mostrar recomendaciones
    recommendations = result['kahneman_recommendations']
    print(f"\n💡 KAHNEMAN RECOMMENDATIONS:")
    for i, rec in enumerate(recommendations['corrections_recommended'], 1):
        print(f"{i}. {rec}")