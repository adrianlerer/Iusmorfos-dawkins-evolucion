#!/usr/bin/env python3
"""
Kahneman Bias Detection Module
Detecta automáticamente sesgos cognitivos en predicciones del framework

Sesgos implementados basados en "Thinking, Fast and Slow":
1. Ilusión de Validez (coherencia ≠ validez)
2. Heurística de Representatividad (ignorar tasas base)
3. Heurística de Disponibilidad (ejemplos memorables)
4. Sesgo de Anclaje (valores iniciales)
5. Exceso de Confianza (intervalos demasiado estrechos)
6. Falacia Narrativa (historias simples sobre eventos complejos)

@author: Iusmorfos Universal Framework + Kahneman Bias Detection
@version: 1.0 - Comprehensive Bias Detection System
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import logging
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BiasType(Enum):
    """Tipos de sesgos cognitivos según Kahneman"""
    ILLUSION_OF_VALIDITY = "illusion_of_validity"
    REPRESENTATIVENESS = "representativeness_heuristic" 
    AVAILABILITY = "availability_heuristic"
    ANCHORING = "anchoring_bias"
    OVERCONFIDENCE = "overconfidence_bias"
    NARRATIVE_FALLACY = "narrative_fallacy"
    PLANNING_FALLACY = "planning_fallacy"
    OUTCOME_BIAS = "outcome_bias"

@dataclass
class BiasDetection:
    """Resultado de detección de sesgo específico"""
    bias_type: BiasType
    severity: float  # 0-1, severidad del sesgo detectado
    confidence: float  # 0-1, confianza en detección
    evidence: List[str]  # Evidencia específica del sesgo
    correction_needed: bool
    impact_on_prediction: float  # Impacto estimado en accuracy predicción

@dataclass
class PredictionInput:
    """Input para análisis de sesgos en predicción"""
    prediction_value: float
    confidence_interval: Tuple[float, float] 
    evidence_description: str
    methodology_used: str
    historical_comparisons: List[str] = field(default_factory=list)
    narrative_elements: List[str] = field(default_factory=list)
    anchor_values: List[float] = field(default_factory=list)

class KahnemanBiasDetector:
    """
    Detector automático de sesgos cognitivos en predicciones
    Basado en taxonomía completa de Kahneman de sesgos Sistema 1
    """
    
    def __init__(self):
        self.detection_thresholds = self._initialize_thresholds()
        self.bias_patterns = self._initialize_bias_patterns()
        self.detection_history: List[Dict] = []
        
    def _initialize_thresholds(self) -> Dict[BiasType, Dict]:
        """Umbrales para detección automática de sesgos"""
        return {
            BiasType.ILLUSION_OF_VALIDITY: {
                'min_coherence_for_detection': 0.7,
                'max_evidence_quality_threshold': 0.5,
                'confidence_inflation_threshold': 0.3
            },
            BiasType.REPRESENTATIVENESS: {
                'stereotype_language_threshold': 3,  # Número palabras estereotípicas
                'base_rate_mention_required': True,
                'similarity_focus_threshold': 0.6
            }, 
            BiasType.AVAILABILITY: {
                'memorable_example_threshold': 2,  # Ejemplos memorables mencionados
                'recent_event_weight_threshold': 0.7,
                'media_coverage_bias_threshold': 0.5
            },
            BiasType.ANCHORING: {
                'anchor_influence_threshold': 0.3,
                'initial_value_correlation_min': 0.4,
                'adjustment_insufficiency_threshold': 0.5
            },
            BiasType.OVERCONFIDENCE: {
                'confidence_interval_width_max': 0.2,  # CI muy estrecho
                'prediction_precision_threshold': 0.05,  # Predicciones muy precisas
                'uncertainty_acknowledgment_min': 0.3
            }
        }
    
    def _initialize_bias_patterns(self) -> Dict[BiasType, List[str]]:
        """Patrones de lenguaje que indican sesgos específicos"""
        return {
            BiasType.ILLUSION_OF_VALIDITY: [
                "obviously", "clearly", "definitely", "certainly",
                "coherent story", "makes perfect sense", "logical conclusion"
            ],
            BiasType.REPRESENTATIVENESS: [
                "typical", "stereotype", "looks like", "similar to", 
                "reminds me of", "classic case", "textbook example"
            ],
            BiasType.AVAILABILITY: [
                "remember when", "recent example", "everyone knows", 
                "just like", "reminds me", "similar situation"
            ],
            BiasType.NARRATIVE_FALLACY: [
                "simple explanation", "clear cause", "obvious reason",
                "story makes sense", "logical sequence", "inevitable outcome"
            ],
            BiasType.OVERCONFIDENCE: [
                "certain", "guaranteed", "definitely", "without doubt",
                "precisely", "exactly", "specifically"
            ]
        }
    
    def detect_all_biases(self, prediction_input: PredictionInput) -> List[BiasDetection]:
        """Detectar todos los sesgos potenciales en una predicción"""
        
        detected_biases = []
        
        # 1. Detectar Ilusión de Validez
        illusion_bias = self._detect_illusion_of_validity(prediction_input)
        if illusion_bias:
            detected_biases.append(illusion_bias)
        
        # 2. Detectar Heurística de Representatividad
        representativeness_bias = self._detect_representativeness_heuristic(prediction_input)
        if representativeness_bias:
            detected_biases.append(representativeness_bias)
        
        # 3. Detectar Heurística de Disponibilidad
        availability_bias = self._detect_availability_heuristic(prediction_input)
        if availability_bias:
            detected_biases.append(availability_bias)
        
        # 4. Detectar Sesgo de Anclaje
        anchoring_bias = self._detect_anchoring_bias(prediction_input)
        if anchoring_bias:
            detected_biases.append(anchoring_bias)
        
        # 5. Detectar Exceso de Confianza
        overconfidence_bias = self._detect_overconfidence_bias(prediction_input)
        if overconfidence_bias:
            detected_biases.append(overconfidence_bias)
        
        # 6. Detectar Falacia Narrativa
        narrative_bias = self._detect_narrative_fallacy(prediction_input)
        if narrative_bias:
            detected_biases.append(narrative_bias)
        
        # Registrar detección
        self.detection_history.append({
            'timestamp': pd.Timestamp.now(),
            'prediction_value': prediction_input.prediction_value,
            'biases_detected': [b.bias_type.value for b in detected_biases],
            'total_bias_severity': sum(b.severity for b in detected_biases)
        })
        
        return detected_biases
    
    def _detect_illusion_of_validity(self, pred_input: PredictionInput) -> Optional[BiasDetection]:
        """
        Detectar Ilusión de Validez: alta confianza basada en coherencia vs. calidad evidencia
        Kahneman: "La confianza subjetiva refleja coherencia, no calidad de evidencia"
        """
        
        # Analizar patrones de lenguaje de alta confianza
        high_confidence_patterns = self.bias_patterns[BiasType.ILLUSION_OF_VALIDITY]
        confidence_language_count = sum(
            1 for pattern in high_confidence_patterns
            if pattern.lower() in pred_input.evidence_description.lower()
        )
        
        # Analizar anchura de intervalo de confianza
        ci_width = pred_input.confidence_interval[1] - pred_input.confidence_interval[0]
        
        # Detectar coherencia narrativa alta
        narrative_coherence = self._assess_narrative_coherence(pred_input.evidence_description)
        
        # Criterios para ilusión de validez
        coherence_threshold = self.detection_thresholds[BiasType.ILLUSION_OF_VALIDITY]['min_coherence_for_detection']
        
        if (narrative_coherence > coherence_threshold and 
            ci_width < 0.2 and  # Intervalo muy estrecho
            confidence_language_count >= 2):  # Lenguaje de alta confianza
            
            severity = min(1.0, narrative_coherence + (3 - ci_width * 5) * 0.1)
            
            return BiasDetection(
                bias_type=BiasType.ILLUSION_OF_VALIDITY,
                severity=severity,
                confidence=0.8,
                evidence=[
                    f"High narrative coherence ({narrative_coherence:.2f}) with narrow CI ({ci_width:.2f})",
                    f"Confidence language patterns: {confidence_language_count}",
                    "Coherence may be mistaken for validity"
                ],
                correction_needed=True,
                impact_on_prediction=severity * 0.3
            )
        
        return None
    
    def _detect_representativeness_heuristic(self, pred_input: PredictionInput) -> Optional[BiasDetection]:
        """
        Detectar uso de Heurística de Representatividad
        Kahneman: Juzgar probabilidad por similaridad a estereotipo, ignorando tasas base
        """
        
        # Buscar lenguaje estereotípico
        stereotype_patterns = self.bias_patterns[BiasType.REPRESENTATIVENESS]
        stereotype_count = sum(
            1 for pattern in stereotype_patterns
            if pattern.lower() in pred_input.evidence_description.lower()
        )
        
        # Verificar si se mencionan tasas base
        base_rate_mentions = any(
            term in pred_input.evidence_description.lower() 
            for term in ['base rate', 'historical rate', 'average', 'typical rate', 'usual outcome']
        )
        
        # Buscar comparaciones por similaridad
        similarity_language = any(
            term in pred_input.evidence_description.lower()
            for term in ['similar to', 'like', 'resembles', 'typical of', 'characteristic of']
        )
        
        if (stereotype_count >= 2 and 
            not base_rate_mentions and 
            similarity_language):
            
            severity = min(1.0, stereotype_count * 0.2 + (0.5 if similarity_language else 0))
            
            return BiasDetection(
                bias_type=BiasType.REPRESENTATIVENESS,
                severity=severity,
                confidence=0.75,
                evidence=[
                    f"Stereotype language count: {stereotype_count}",
                    f"Base rate mentioned: {base_rate_mentions}",
                    f"Similarity-based reasoning detected: {similarity_language}",
                    "Prediction may ignore statistical base rates"
                ],
                correction_needed=True,
                impact_on_prediction=severity * 0.4
            )
        
        return None
    
    def _detect_availability_heuristic(self, pred_input: PredictionInput) -> Optional[BiasDetection]:
        """
        Detectar Heurística de Disponibilidad
        Kahneman: Estimar frecuencia por facilidad de recordar ejemplos
        """
        
        # Buscar referencias a ejemplos específicos memorables
        availability_patterns = self.bias_patterns[BiasType.AVAILABILITY]
        availability_count = sum(
            1 for pattern in availability_patterns
            if pattern.lower() in pred_input.evidence_description.lower()
        )
        
        # Detectar referencias a eventos recientes
        recent_indicators = ['recent', 'lately', 'just happened', 'last year', 'recently']
        recent_focus = sum(
            1 for indicator in recent_indicators
            if indicator in pred_input.evidence_description.lower()
        )
        
        # Detectar énfasis en casos específicos vs. estadísticas
        specific_examples = len(re.findall(r'\b(example|case|instance|situation)\b', 
                                         pred_input.evidence_description.lower()))
        
        if (availability_count >= 2 or 
            recent_focus >= 2 or 
            specific_examples >= 3):
            
            severity = min(1.0, (availability_count + recent_focus + specific_examples * 0.3) * 0.2)
            
            return BiasDetection(
                bias_type=BiasType.AVAILABILITY,
                severity=severity,
                confidence=0.7,
                evidence=[
                    f"Availability language patterns: {availability_count}",
                    f"Recent event focus: {recent_focus}",
                    f"Specific examples emphasized: {specific_examples}",
                    "Prediction may overweight memorable/recent examples"
                ],
                correction_needed=True,
                impact_on_prediction=severity * 0.3
            )
        
        return None
    
    def _detect_anchoring_bias(self, pred_input: PredictionInput) -> Optional[BiasDetection]:
        """
        Detectar Sesgo de Anclaje
        Kahneman: Influencia excesiva de valores iniciales en estimaciones
        """
        
        # Si hay valores de anclaje disponibles
        if pred_input.anchor_values:
            
            # Calcular correlación entre prediction y anchors
            correlations = []
            for anchor in pred_input.anchor_values:
                if anchor != 0:  # Evitar división por cero
                    correlation = abs(pred_input.prediction_value - anchor) / max(anchor, pred_input.prediction_value)
                    correlations.append(1 - correlation)  # Invertir para medir similaridad
            
            avg_correlation = np.mean(correlations) if correlations else 0
            
            # Detectar ajuste insuficiente desde anchor
            insufficient_adjustment = any(
                abs(pred_input.prediction_value - anchor) / max(anchor, 0.1) < 0.2
                for anchor in pred_input.anchor_values
            )
            
            if (avg_correlation > 0.6 or insufficient_adjustment):
                
                severity = min(1.0, avg_correlation + (0.3 if insufficient_adjustment else 0))
                
                return BiasDetection(
                    bias_type=BiasType.ANCHORING,
                    severity=severity,
                    confidence=0.65,
                    evidence=[
                        f"High correlation with anchor values: {avg_correlation:.2f}",
                        f"Insufficient adjustment detected: {insufficient_adjustment}",
                        f"Anchor values: {pred_input.anchor_values}",
                        "Prediction may be unduly influenced by initial values"
                    ],
                    correction_needed=True,
                    impact_on_prediction=severity * 0.25
                )
        
        return None
    
    def _detect_overconfidence_bias(self, pred_input: PredictionInput) -> Optional[BiasDetection]:
        """
        Detectar Exceso de Confianza
        Kahneman: Intervalos de confianza demasiado estrechos, precisión excesiva
        """
        
        ci_width = pred_input.confidence_interval[1] - pred_input.confidence_interval[0]
        
        # Detectar intervalo muy estrecho
        narrow_ci = ci_width < 0.15
        
        # Detectar lenguaje de exceso de confianza
        overconfidence_patterns = self.bias_patterns[BiasType.OVERCONFIDENCE]
        overconfidence_count = sum(
            1 for pattern in overconfidence_patterns
            if pattern.lower() in pred_input.evidence_description.lower()
        )
        
        # Detectar precisión excesiva en predicción
        excessive_precision = (pred_input.prediction_value * 100) % 1 != 0  # Decimales precisos
        
        # Verificar mención de incertidumbre
        uncertainty_acknowledgment = any(
            term in pred_input.evidence_description.lower()
            for term in ['uncertain', 'may vary', 'could be', 'approximately', 'roughly']
        )
        
        if ((narrow_ci and overconfidence_count >= 1) or 
            (overconfidence_count >= 3) or 
            (excessive_precision and not uncertainty_acknowledgment)):
            
            severity = min(1.0, 
                          (0.5 if narrow_ci else 0) + 
                          overconfidence_count * 0.2 + 
                          (0.3 if excessive_precision and not uncertainty_acknowledgment else 0))
            
            return BiasDetection(
                bias_type=BiasType.OVERCONFIDENCE,
                severity=severity,
                confidence=0.8,
                evidence=[
                    f"Narrow confidence interval: {ci_width:.3f}",
                    f"Overconfidence language: {overconfidence_count}",
                    f"Excessive precision: {excessive_precision}",
                    f"Uncertainty acknowledged: {uncertainty_acknowledgment}",
                    "Prediction shows signs of overconfidence"
                ],
                correction_needed=True,
                impact_on_prediction=severity * 0.35
            )
        
        return None
    
    def _detect_narrative_fallacy(self, pred_input: PredictionInput) -> Optional[BiasDetection]:
        """
        Detectar Falacia Narrativa
        Kahneman: Explicaciones simples para eventos complejos, subestimando rol del azar
        """
        
        # Buscar lenguaje causal simple
        causal_language = ['because', 'caused by', 'due to', 'result of', 'leads to', 'therefore']
        causal_count = sum(
            1 for term in causal_language
            if term in pred_input.evidence_description.lower()
        )
        
        # Buscar narrativa de certeza
        certainty_narrative = ['inevitable', 'obvious', 'clear', 'simple', 'straightforward']
        certainty_count = sum(
            1 for term in certainty_narrative
            if term in pred_input.evidence_description.lower()
        )
        
        # Verificar mención de factores aleatorios/complejos
        complexity_acknowledgment = any(
            term in pred_input.evidence_description.lower()
            for term in ['complex', 'random', 'unpredictable', 'multiple factors', 'uncertain']
        )
        
        # Contar elementos narrativos
        narrative_elements_count = len(pred_input.narrative_elements)
        
        if ((causal_count >= 3 and certainty_count >= 2 and not complexity_acknowledgment) or
            narrative_elements_count >= 5):
            
            severity = min(1.0, 
                          causal_count * 0.1 + 
                          certainty_count * 0.15 + 
                          narrative_elements_count * 0.1 +
                          (0.3 if not complexity_acknowledgment else 0))
            
            return BiasDetection(
                bias_type=BiasType.NARRATIVE_FALLACY,
                severity=severity,
                confidence=0.7,
                evidence=[
                    f"Causal language frequency: {causal_count}",
                    f"Certainty narrative: {certainty_count}",
                    f"Complexity acknowledged: {complexity_acknowledgment}",
                    f"Narrative elements: {narrative_elements_count}",
                    "Prediction may oversimplify complex causation"
                ],
                correction_needed=True,
                impact_on_prediction=severity * 0.25
            )
        
        return None
    
    def _assess_narrative_coherence(self, text: str) -> float:
        """Evaluar coherencia narrativa de un texto (0-1)"""
        
        # Indicadores de coherencia narrativa
        coherence_indicators = [
            'therefore', 'thus', 'consequently', 'as a result',
            'clearly', 'obviously', 'naturally', 'logically',
            'leads to', 'causes', 'results in', 'explains'
        ]
        
        coherence_score = sum(
            1 for indicator in coherence_indicators
            if indicator.lower() in text.lower()
        )
        
        # Normalizar por longitud del texto
        words = len(text.split())
        normalized_score = min(1.0, coherence_score / max(words / 50, 1))
        
        return normalized_score
    
    def get_bias_summary(self, detected_biases: List[BiasDetection]) -> Dict:
        """Generar resumen de sesgos detectados"""
        
        if not detected_biases:
            return {"biases_detected": 0, "overall_reliability": "HIGH"}
        
        total_severity = sum(bias.severity for bias in detected_biases)
        avg_severity = total_severity / len(detected_biases)
        total_impact = sum(bias.impact_on_prediction for bias in detected_biases)
        
        reliability_levels = {
            (0, 0.3): "HIGH",
            (0.3, 0.6): "MODERATE", 
            (0.6, 1.0): "LOW"
        }
        
        reliability = next(
            level for (min_val, max_val), level in reliability_levels.items()
            if min_val <= total_impact < max_val
        )
        
        return {
            "biases_detected": len(detected_biases),
            "bias_types": [bias.bias_type.value for bias in detected_biases],
            "average_severity": avg_severity,
            "total_impact_on_prediction": total_impact,
            "overall_reliability": reliability,
            "corrections_recommended": sum(1 for bias in detected_biases if bias.correction_needed),
            "detailed_biases": [
                {
                    "type": bias.bias_type.value,
                    "severity": bias.severity,
                    "evidence": bias.evidence
                }
                for bias in detected_biases
            ]
        }

# Función de utilidad para aplicar detección a análisis Argentina
def detect_biases_in_argentina_analysis(prediction_text: str, 
                                       prediction_value: float,
                                       confidence_interval: Tuple[float, float]) -> Dict:
    """Aplicar detección de sesgos al análisis de Argentina"""
    
    detector = KahnemanBiasDetector()
    
    pred_input = PredictionInput(
        prediction_value=prediction_value,
        confidence_interval=confidence_interval,
        evidence_description=prediction_text,
        methodology_used="iusmorfos_universal_framework",
        historical_comparisons=["Argentina 2001 crisis", "Argentina 1989 hyperinflation"],
        narrative_elements=["economic stabilization story", "political reform narrative"],
        anchor_values=[0.4, 0.6]  # Valores de referencia históricos
    )
    
    detected_biases = detector.detect_all_biases(pred_input)
    bias_summary = detector.get_bias_summary(detected_biases)
    
    return bias_summary

if __name__ == "__main__":
    # Test del detector con ejemplo
    example_text = """
    Argentina will definitely achieve economic stabilization because the current situation 
    is clearly similar to successful cases like Chile in the 1980s. The coherent policy 
    package obviously leads to positive results, just like we remember from other Latin 
    American success stories. This is a typical case of market-friendly reforms that 
    inevitably produce growth.
    """
    
    result = detect_biases_in_argentina_analysis(
        example_text, 
        0.75,  # 75% prediction
        (0.70, 0.80)  # Narrow confidence interval
    )
    
    print("🧠 KAHNEMAN BIAS DETECTION RESULTS:")
    print("="*50)
    for key, value in result.items():
        print(f"{key}: {value}")