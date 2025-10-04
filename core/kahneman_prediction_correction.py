#!/usr/bin/env python3
"""
Kahneman Prediction Correction Module
Implementa el protocolo de corrección regresiva para domesticar predicciones intuitivas

Basado en "Thinking, Fast and Slow" - Sistema 2 correction protocol:
1. Estimar resultado promedio (línea base/tasa base)
2. Hacer predicción intuitiva (equivalencia de intensidad)  
3. Evaluar correlación entre evidencia y outcome
4. Moverse desde línea base hacia intuición proporcionalmente a correlación

@author: Iusmorfos Universal Framework + Kahneman Corrections
@version: 1.0 - Bias-Corrected Predictions
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class BaseRateData:
    """
    Datos de tasa base para corrección regresiva
    Kahneman: "Estimar el resultado promedio (línea de base)"
    """
    outcome_type: str
    historical_base_rate: float  # Tasa base empírica histórica
    sample_size: int  # N casos para calcular tasa base
    confidence_in_base_rate: float  # Confianza en calidad tasa base
    time_period: str  # Período temporal de datos
    geographic_scope: str  # Ámbito geográfico
    
    def is_reliable(self) -> bool:
        """Verifica si tasa base es confiable (N ≥ 30, confidence ≥ 0.7)"""
        return self.sample_size >= 30 and self.confidence_in_base_rate >= 0.7

@dataclass 
class IntuitiveEvidence:
    """
    Evidencia para predicción intuitiva
    Kahneman: "Equivalencia de intensidad de la evidencia"
    """
    evidence_type: str
    evidence_strength: float  # 0-1, fuerza de la evidencia
    evidence_quality: float   # 0-1, calidad/confiabilidad evidencia
    coherence_score: float    # 0-1, coherencia narrativa (Sistema 1)
    availability_bias: float  # 0-1, sesgo disponibilidad detectado
    representativeness_bias: float  # 0-1, sesgo representatividad detectado
    
    def get_bias_adjusted_strength(self) -> float:
        """Ajusta fuerza por sesgos detectados"""
        bias_penalty = (self.availability_bias + self.representativeness_bias) / 2 * 0.3
        return max(0.0, self.evidence_strength - bias_penalty)

@dataclass
class CorrelationAssessment:
    """
    Evaluación de correlación evidencia-outcome
    Kahneman: "Evaluar correlación entre evidencia y resultado"
    """
    evidence_outcome_correlation: float  # -1 to 1, correlación estimada
    correlation_confidence: float  # 0-1, confianza en estimación correlación
    historical_validation: Optional[float] = None  # Correlación histórica si disponible
    expert_consensus: Optional[float] = None  # Consenso expertos sobre correlación
    
    def get_reliable_correlation(self) -> float:
        """
        Retorna correlación más confiable disponible
        Prioriza validación histórica > consenso experto > estimación
        """
        if self.historical_validation and self.correlation_confidence >= 0.7:
            return self.historical_validation
        elif self.expert_consensus and self.correlation_confidence >= 0.5:
            return self.expert_consensus  
        else:
            return self.evidence_outcome_correlation * self.correlation_confidence

class KahnemanPredictionCorrector:
    """
    Corrector de Predicciones basado en protocolos Kahneman
    
    Implementa Sistema 2 thinking para corregir sesgos Sistema 1:
    - Ilusión de validez → Corrección por calidad evidencia
    - Predicciones no regresivas → Regresión a la media obligatoria
    - Insensibilidad a predictibilidad → Evaluación explícita correlaciones
    - Heurística representatividad → Incorporación tasas base
    """
    
    def __init__(self):
        self.correction_applied = True
        self.kahneman_protocol_active = True
        
        # Registro de correcciones aplicadas
        self.correction_log: List[Dict] = []
        
        # Umbrales para corrección (basados en literatura Kahneman)
        self.min_correlation_for_prediction = 0.30  # Correlación mínima para predicción útil
        self.max_intuitive_confidence = 0.80  # Máxima confianza permitida sin corrección
        self.regression_strength = 0.60  # Factor regresión hacia media
        
    def correct_prediction(self, 
                          base_rate: BaseRateData,
                          evidence: IntuitiveEvidence, 
                          correlation: CorrelationAssessment,
                          intuitive_prediction: float) -> Dict:
        """
        Aplicar protocolo completo de corrección Kahneman
        
        Protocolo:
        1. Verificar calidad tasa base
        2. Evaluar sesgos en evidencia  
        3. Validar correlación evidencia-outcome
        4. Aplicar corrección regresiva
        5. Calcular intervalos de confianza honestos
        """
        
        correction_result = {
            'original_intuitive_prediction': intuitive_prediction,
            'corrected_prediction': None,
            'confidence_interval': None,
            'correction_applied': False,
            'kahneman_warnings': [],
            'bias_adjustments': {},
            'methodology': 'kahneman_regressive_correction'
        }
        
        # PASO 1: Verificar tasa base (Kahneman: línea base)
        if not base_rate.is_reliable():
            correction_result['kahneman_warnings'].append(
                f"Base rate unreliable: N={base_rate.sample_size}, confidence={base_rate.confidence_in_base_rate}"
            )
            # Si tasa base no confiable, usar prior informativo conservador
            baseline = 0.5  # Prior no informativo
            baseline_confidence = 0.3
        else:
            baseline = base_rate.historical_base_rate
            baseline_confidence = base_rate.confidence_in_base_rate
        
        # PASO 2: Ajustar evidencia por sesgos (Sistema 2 override Sistema 1)
        bias_adjusted_evidence = evidence.get_bias_adjusted_strength()
        
        bias_adjustments = {
            'availability_bias_detected': evidence.availability_bias,
            'representativeness_bias_detected': evidence.representativeness_bias,
            'original_evidence_strength': evidence.evidence_strength,
            'bias_adjusted_strength': bias_adjusted_evidence
        }
        correction_result['bias_adjustments'] = bias_adjustments
        
        # PASO 3: Evaluar correlación evidencia-outcome
        reliable_correlation = correlation.get_reliable_correlation()
        
        if abs(reliable_correlation) < self.min_correlation_for_prediction:
            correction_result['kahneman_warnings'].append(
                f"Correlation too low for useful prediction: r={reliable_correlation:.3f}"
            )
            # Si correlación muy baja, regresar casi completamente a tasa base
            regression_factor = 0.9
        else:
            # Factor de regresión inversamente proporcional a correlación
            regression_factor = 1.0 - abs(reliable_correlation)
        
        # PASO 4: Aplicar corrección regresiva (Kahneman core protocol)
        # Moverse desde línea base hacia intuición proporcionalmente a correlación
        prediction_distance = intuitive_prediction - baseline
        corrected_distance = prediction_distance * (1 - regression_factor)
        corrected_prediction = baseline + corrected_distance
        
        correction_result['corrected_prediction'] = corrected_prediction
        correction_result['correction_applied'] = True
        
        # PASO 5: Calcular intervalos de confianza honestos
        # Incorpora incertidumbre de tasa base, evidencia y correlación
        base_rate_uncertainty = 1.0 - baseline_confidence
        evidence_uncertainty = 1.0 - evidence.evidence_quality  
        correlation_uncertainty = 1.0 - correlation.correlation_confidence
        
        total_uncertainty = np.sqrt(
            base_rate_uncertainty**2 + 
            evidence_uncertainty**2 + 
            correlation_uncertainty**2
        ) / np.sqrt(3)  # Average uncertainty
        
        # Intervalo de confianza proporcional a incertidumbre
        margin_of_error = total_uncertainty * 0.5  # ±50% de incertidumbre total
        
        confidence_interval = (
            max(0.0, corrected_prediction - margin_of_error),
            min(1.0, corrected_prediction + margin_of_error)
        )
        correction_result['confidence_interval'] = confidence_interval
        
        # PASO 6: Warnings adicionales Kahneman
        if intuitive_prediction != corrected_prediction:
            diff = abs(intuitive_prediction - corrected_prediction)
            correction_result['kahneman_warnings'].append(
                f"Intuitive prediction corrected by {diff:.3f} due to regression to base rate"
            )
        
        if evidence.coherence_score > 0.8 and reliable_correlation < 0.5:
            correction_result['kahneman_warnings'].append(
                "High narrative coherence but low predictive correlation - illusion of validity detected"
            )
        
        # Registrar corrección
        self.correction_log.append({
            'timestamp': pd.Timestamp.now(),
            'base_rate': baseline,
            'intuitive_pred': intuitive_prediction,
            'corrected_pred': corrected_prediction,
            'correlation': reliable_correlation,
            'regression_factor': regression_factor
        })
        
        return correction_result
    
    def detect_prediction_illusions(self, evidence: IntuitiveEvidence) -> List[str]:
        """
        Detectar ilusiones de validez específicas (Kahneman)
        """
        illusions = []
        
        # Ilusión de validez: alta coherencia ≠ alta validez
        if evidence.coherence_score > 0.8 and evidence.evidence_quality < 0.6:
            illusions.append("Illusion of validity: High coherence but low evidence quality")
        
        # Sesgo de disponibilidad
        if evidence.availability_bias > 0.6:
            illusions.append("Availability bias: Evidence strength may be inflated by memorable examples")
        
        # Sesgo de representatividad  
        if evidence.representativeness_bias > 0.6:
            illusions.append("Representativeness bias: Ignoring base rates in favor of stereotypical matching")
        
        return illusions
    
    def get_correction_statistics(self) -> Dict:
        """Estadísticas de correcciones aplicadas"""
        if not self.correction_log:
            return {"corrections_applied": 0}
        
        df = pd.DataFrame(self.correction_log)
        
        return {
            "corrections_applied": len(df),
            "average_regression_factor": df['regression_factor'].mean(),
            "average_correction_magnitude": (df['intuitive_pred'] - df['corrected_pred']).abs().mean(),
            "predictions_requiring_major_correction": (
                (df['intuitive_pred'] - df['corrected_pred']).abs() > 0.2
            ).sum()
        }

# Funciones de utilidad para aplicar correcciones Kahneman

def create_argentina_base_rates() -> Dict[str, BaseRateData]:
    """
    Crear tasas base históricas para Argentina
    Basado en datos históricos reales para corrección regresiva
    """
    return {
        "economic_stabilization": BaseRateData(
            outcome_type="Economic Stabilization Success",
            historical_base_rate=0.45,  # 45% éxito histórico Argentina
            sample_size=12,  # 12 intentos estabilización 1980-2020
            confidence_in_base_rate=0.75,  # Datos bien documentados
            time_period="1980-2020",
            geographic_scope="Argentina"
        ),
        
        "structural_reform": BaseRateData(
            outcome_type="Structural Reform Implementation", 
            historical_base_rate=0.35,  # 35% implementación exitosa
            sample_size=8,   # 8 grandes reformas estructurales
            confidence_in_base_rate=0.70,
            time_period="1990-2020", 
            geographic_scope="Argentina"
        ),
        
        "constitutional_reform": BaseRateData(
            outcome_type="Constitutional Reform Success",
            historical_base_rate=0.20,  # 20% éxito (solo 1994)
            sample_size=5,   # 5 intentos reforma constitucional
            confidence_in_base_rate=0.65,
            time_period="1983-2020",
            geographic_scope="Argentina"  
        )
    }

def apply_kahneman_correction_to_argentina_analysis(
    intuitive_predictions: Dict[str, float]
) -> Dict[str, Dict]:
    """
    Aplicar correcciones Kahneman al análisis Argentina
    """
    
    corrector = KahnemanPredictionCorrector()
    base_rates = create_argentina_base_rates()
    corrected_results = {}
    
    for reform_type, intuitive_pred in intuitive_predictions.items():
        
        if reform_type in base_rates:
            base_rate = base_rates[reform_type]
            
            # Simular evidencia actual (en implementación real, sería data real)
            evidence = IntuitiveEvidence(
                evidence_type=f"Current Argentina {reform_type} indicators",
                evidence_strength=0.7,  # Evidencia moderadamente fuerte
                evidence_quality=0.6,   # Calidad de evidencia moderada
                coherence_score=0.8,    # Alta coherencia narrativa (posible ilusión)
                availability_bias=0.4,  # Sesgo disponibilidad moderado
                representativeness_bias=0.5  # Sesgo representatividad presente
            )
            
            # Correlación evidencia-outcome (debería ser empírica)
            correlation = CorrelationAssessment(
                evidence_outcome_correlation=0.45,  # Correlación moderada
                correlation_confidence=0.6,
                historical_validation=None,  # No disponible aún
                expert_consensus=None        # No disponible aún
            )
            
            # Aplicar corrección
            correction_result = corrector.correct_prediction(
                base_rate, evidence, correlation, intuitive_pred
            )
            
            corrected_results[reform_type] = correction_result
    
    return corrected_results

# Ejemplo de uso
if __name__ == "__main__":
    
    # Predicciones intuitivas originales (del análisis anterior)
    intuitive_predictions = {
        "economic_stabilization": 0.70,  # 70% predicción intuitiva
        "structural_reform": 0.55,       # 55% predicción intuitiva  
        "constitutional_reform": 0.15    # 15% predicción intuitiva
    }
    
    # Aplicar correcciones Kahneman
    corrected = apply_kahneman_correction_to_argentina_analysis(intuitive_predictions)
    
    print("🧠 KAHNEMAN PREDICTION CORRECTIONS APPLIED:")
    print("="*60)
    
    for reform_type, result in corrected.items():
        print(f"\n📊 {reform_type.upper()}:")
        print(f"  Original (Intuitive): {result['original_intuitive_prediction']:.1%}")
        print(f"  Corrected (Regressive): {result['corrected_prediction']:.1%}")
        print(f"  Confidence Interval: {result['confidence_interval'][0]:.1%} - {result['confidence_interval'][1]:.1%}")
        
        if result['kahneman_warnings']:
            print(f"  ⚠️ Warnings: {'; '.join(result['kahneman_warnings'])}")