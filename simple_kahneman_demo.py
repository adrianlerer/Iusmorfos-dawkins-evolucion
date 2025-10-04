#!/usr/bin/env python3
"""
Demonstración simple de componentes Kahneman implementados
Muestra detección de sesgos y corrección regresiva sin dependencias complejas
"""

import sys
import os
from typing import Dict, List

# Add core to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'core'))

def demo_bias_detection():
    """Demonstración de detección automática de sesgos Kahneman"""
    print("🧠 DEMO: DETECCIÓN AUTOMÁTICA DE SESGOS KAHNEMAN")
    print("=" * 60)
    
    from kahneman_bias_detector import KahnemanBiasDetector, PredictionInput
    
    detector = KahnemanBiasDetector()
    
    # Casos de test con sesgos claros
    test_cases = [
        {
            "name": "Evidencia con Sesgo de Disponibilidad",
            "text": "Recuerdo varios casos recientes donde reformas similares funcionaron muy bien",
            "expected_biases": ["availability_heuristic"]
        },
        {
            "name": "Evidencia con Representatividad",
            "text": "Este país es exactamente como Estonia donde la reforma fue 100% exitosa",
            "expected_biases": ["representativeness_heuristic"]  
        },
        {
            "name": "Evidencia con Exceso de Confianza",
            "text": "Estoy completamente seguro que esto funcionará al 95% sin dudas",
            "expected_biases": ["overconfidence"]
        },
        {
            "name": "Evidencia con Anclaje",
            "text": "La reforma anterior tuvo 85% de éxito, esta será similar",
            "expected_biases": ["anchoring"]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📝 Test {i}: {test_case['name']}")
        print(f"   Texto: {test_case['text']}")
        
        # Crear input de predicción
        pred_input = PredictionInput(
            prediction_value=0.8,
            confidence_interval=(0.7, 0.9),
            evidence_description=test_case['text'],
            methodology_used="expert_analysis"
        )
        
        # Detectar sesgos
        detected_biases = detector.detect_all_biases(pred_input)
        
        print(f"   🎯 Sesgos Detectados ({len(detected_biases)}):")
        for bias in detected_biases:
            print(f"      • {bias.bias_type.value}: {bias.severity:.2f} (confianza: {bias.confidence:.2f})")
            print(f"        Descripción: {bias.description[:80]}...")
        
        # Verificar si detectó los sesgos esperados
        detected_types = [bias.bias_type.value for bias in detected_biases]
        expected_found = any(expected in detected_types for expected in test_case['expected_biases'])
        status = "✅" if expected_found else "⚠️"
        print(f"   {status} Detección esperada: {'SÍ' if expected_found else 'PARCIAL'}")
    
    print(f"\n📊 RESUMEN DETECCIÓN DE SESGOS:")
    print("   • Illusion of validity: Detecta coherencia artificial")  
    print("   • Representativeness: Identifica comparaciones superficiales")
    print("   • Availability bias: Encuentra ejemplos memorables recientes")
    print("   • Anchoring: Localiza valores de referencia inadecuados")
    print("   • Overconfidence: Mide intervalos de confianza demasiado estrechos")
    print("   • Narrative fallacy: Identifica historias causales simplistas")

def demo_regressive_correction():
    """Demonstración de corrección regresiva Kahneman"""
    print("\n🎯 DEMO: CORRECCIÓN REGRESIVA DE PREDICCIONES")
    print("=" * 60)
    
    from kahneman_prediction_correction import (
        KahnemanPredictionCorrector, BaseRateData, IntuitiveEvidence, 
        CorrelationAssessment
    )
    
    corrector = KahnemanPredictionCorrector()
    
    # Caso: Reforma judicial en Argentina
    print("📋 CASO: Reforma Judicial Argentina")
    
    # Base rate histórica (realista)
    base_rate = BaseRateData(
        outcome_type="Judicial Reform Argentina",
        historical_base_rate=0.45,  # 45% éxito histórico
        sample_size=22,
        confidence_in_base_rate=0.75,
        time_period="1990-2023",
        geographic_scope="Argentina"
    )
    print(f"   📊 Base Rate Histórica: {base_rate.historical_base_rate:.1%} (n={base_rate.sample_size})")
    
    # Evidencia intuitiva (optimista)
    evidence = IntuitiveEvidence(
        evidence_type="Expert Analysis + Tech Innovation",
        evidence_strength=0.80,  # Evidencia fuerte
        evidence_quality=0.70,   # Calidad buena pero no perfecta
        coherence_score=0.85,    # Historia coherente
        availability_bias=0.30,  # Algo de sesgo disponibilidad  
        representativeness_bias=0.25  # Algo de representatividad
    )
    print(f"   🔍 Evidencia Strength: {evidence.evidence_strength:.2f}")
    print(f"   🎯 Coherencia: {evidence.coherence_score:.2f}")
    print(f"   ⚠️ Sesgos detectados: Disponibilidad={evidence.availability_bias:.2f}, Representatividad={evidence.representativeness_bias:.2f}")
    
    # Correlación (moderada)
    correlation = CorrelationAssessment(
        evidence_outcome_correlation=0.65,  # Correlación moderada-alta
        correlation_confidence=0.70,
        historical_validation=None,
        expert_consensus=0.75
    )
    print(f"   🔗 Correlación Evidencia-Outcome: {correlation.evidence_outcome_correlation:.2f}")
    
    # Predicción intuitiva inicial (optimista)
    intuitive_prediction = 0.85  # 85% predicción inicial optimista
    print(f"   🎨 Predicción Intuitiva Original: {intuitive_prediction:.1%}")
    
    # Aplicar corrección regresiva
    correction_result = corrector.correct_prediction(
        base_rate, evidence, correlation, intuitive_prediction
    )
    
    print(f"\n🎯 RESULTADOS CORRECCIÓN KAHNEMAN:")
    print(f"   Original: {correction_result['original_intuitive_prediction']:.1%}")
    print(f"   Corregida: {correction_result['corrected_prediction']:.1%}")
    print(f"   Ajuste: {correction_result['original_intuitive_prediction'] - correction_result['corrected_prediction']:.1%}")
    print(f"   Corrección Aplicada: {'Sí' if correction_result['correction_applied'] else 'No'}")
    print(f"   Intervalo Confianza: [{correction_result['confidence_interval'][0]:.1%}, {correction_result['confidence_interval'][1]:.1%}]")
    
    if correction_result['kahneman_warnings']:
        print(f"   ⚠️ Warnings Kahneman:")
        for warning in correction_result['kahneman_warnings'][:3]:
            print(f"      • {warning}")
    
    print(f"\n📈 INTERPRETACIÓN:")
    print(f"   • Predicción original sesgada hacia optimismo")
    print(f"   • Corrección regresiva reduce overconfidence") 
    print(f"   • Base rate histórica modera expectativas")
    print(f"   • Intervalos de confianza más honestos y amplios")

def demo_argentina_analysis_simulation():
    """Simulación de análisis mejorado para Argentina"""
    print("\n🇦🇷 DEMO: SIMULACIÓN ANÁLISIS ARGENTINA KAHNEMAN-ENHANCED")
    print("=" * 70)
    
    from kahneman_bias_detector import KahnemanBiasDetector, PredictionInput
    from kahneman_prediction_correction import (
        KahnemanPredictionCorrector, BaseRateData, IntuitiveEvidence, 
        CorrelationAssessment
    )
    
    detector = KahnemanBiasDetector()
    corrector = KahnemanPredictionCorrector()
    
    # Escenario: Reforma integral del sistema judicial
    scenario = {
        "titulo": "Reforma Integral Sistema Judicial Argentino 2024",
        "descripcion": """
        Propuesta de digitalización completa del Poder Judicial:
        1. Expedientes 100% digitales en 18 meses
        2. IA para asignación automática de casos
        3. Tribunales especializados anti-corrupción 
        4. Sistema de métricas de performance judicial
        5. Reforma Consejo de la Magistratura
        """,
        "evidencia_inicial": """
        Estonia implementó sistema similar con 95% de éxito.
        Expertos argentinos muy optimistas sobre la propuesta.
        La tecnología ya existe y está probada mundialmente.
        Ciudadanía apoya fuertemente modernización judicial.
        Gobierno tiene mayoría parlamentaria para aprobar reformas.
        """
    }
    
    print(f"📋 ESCENARIO: {scenario['titulo']}")
    
    # PASO 1: Análisis de sesgos en evidencia
    print(f"\n🔍 PASO 1: DETECCIÓN DE SESGOS EN EVIDENCIA")
    pred_input = PredictionInput(
        prediction_value=0.88,
        confidence_interval=(0.80, 0.95),
        evidence_description=scenario['evidencia_inicial'],
        methodology_used="expert_analysis_argentina"
    )
    
    detected_biases = detector.detect_all_biases(pred_input)
    print(f"   🚨 Sesgos detectados: {len(detected_biases)}")
    
    bias_impact = 0.0
    for bias in detected_biases[:3]:  # Top 3 sesgos
        print(f"   • {bias.bias_type.value}: Severidad {bias.severity:.2f}")
        bias_impact += bias.severity
    
    # PASO 2: Predicción intuitiva inicial (sesgada)
    print(f"\n🎨 PASO 2: PREDICCIÓN INTUITIVA INICIAL")
    intuitive_success_prob = 0.88  # 88% - muy optimista
    intuitive_ci = (0.80, 0.95)   # Intervalo muy estrecho (overconfidence)
    print(f"   Éxito predicho: {intuitive_success_prob:.1%}")
    print(f"   Intervalo confianza: [{intuitive_ci[0]:.1%}, {intuitive_ci[1]:.1%}]")
    print(f"   ⚠️ Amplitud intervalo: {intuitive_ci[1] - intuitive_ci[0]:.1%} (muy estrecho)")
    
    # PASO 3: Aplicar corrección regresiva
    print(f"\n🎯 PASO 3: CORRECCIÓN REGRESIVA KAHNEMAN")
    
    # Base rate realista para reformas judiciales en Argentina
    argentina_base_rate = BaseRateData(
        outcome_type="Judicial Tech Reform Argentina", 
        historical_base_rate=0.40,  # 40% éxito histórico
        sample_size=15,
        confidence_in_base_rate=0.65,
        time_period="1983-2023",
        geographic_scope="Argentina"
    )
    
    # Evidencia ajustada por sesgos
    evidence_adjusted = IntuitiveEvidence(
        evidence_type="Mixed Evidence with Biases",
        evidence_strength=0.70,
        evidence_quality=0.60,  # Reducida por sesgos
        coherence_score=0.80,
        availability_bias=min(0.5, bias_impact * 0.3),
        representativeness_bias=min(0.4, bias_impact * 0.25)
    )
    
    # Correlación moderada (Estonia ≠ Argentina)
    correlation = CorrelationAssessment(
        evidence_outcome_correlation=0.50,  # Moderada, no alta
        correlation_confidence=0.60,
        historical_validation=None,
        expert_consensus=0.65
    )
    
    # Aplicar corrección
    correction_result = corrector.correct_prediction(
        argentina_base_rate, evidence_adjusted, correlation, intuitive_success_prob
    )
    
    print(f"   Base Rate Argentina: {argentina_base_rate.historical_base_rate:.1%}")
    print(f"   Correlación estimada: {correlation.evidence_outcome_correlation:.2f}")
    print(f"   Metodología: {correction_result['methodology']}")
    
    # PASO 4: Resultados finales
    print(f"\n📊 PASO 4: COMPARACIÓN FINAL")
    print(f"   🎨 ORIGINAL (sesgada):")
    print(f"      Probabilidad éxito: {intuitive_success_prob:.1%}")
    print(f"      Intervalo confianza: [{intuitive_ci[0]:.1%}, {intuitive_ci[1]:.1%}]")
    print(f"      Amplitud intervalo: {intuitive_ci[1] - intuitive_ci[0]:.1%}")
    
    print(f"   🎯 KAHNEMAN-CORRECTED:")
    corrected_prob = correction_result['corrected_prediction']
    corrected_ci = correction_result['confidence_interval']
    print(f"      Probabilidad éxito: {corrected_prob:.1%}")
    print(f"      Intervalo confianza: [{corrected_ci[0]:.1%}, {corrected_ci[1]:.1%}]")
    print(f"      Amplitud intervalo: {corrected_ci[1] - corrected_ci[0]:.1%}")
    
    # Calcular mejoras
    adjustment = intuitive_success_prob - corrected_prob
    ci_improvement = (corrected_ci[1] - corrected_ci[0]) - (intuitive_ci[1] - intuitive_ci[0])
    
    print(f"\n✨ MEJORAS APLICADAS:")
    print(f"   📉 Reducción optimismo: {adjustment:.1%}")
    print(f"   📏 Intervalos más honestos: +{ci_improvement:.1%} amplitud")
    print(f"   🧠 Sesgos corregidos: {len(detected_biases)} tipos detectados")
    print(f"   📊 Base rate integrada: {argentina_base_rate.historical_base_rate:.1%} histórico")
    
    return {
        'original_prediction': intuitive_success_prob,
        'corrected_prediction': corrected_prob,
        'biases_detected': len(detected_biases),
        'adjustment_made': adjustment
    }

def main():
    """Ejecutar todas las demostraciones"""
    print("🚀 DEMOSTRACIÓN COMPONENTES KAHNEMAN IMPLEMENTADOS")
    print("=" * 80)
    
    # Demo 1: Detección de sesgos
    demo_bias_detection()
    
    # Demo 2: Corrección regresiva  
    demo_regressive_correction()
    
    # Demo 3: Análisis completo Argentina
    results = demo_argentina_analysis_simulation()
    
    # Resumen final
    print(f"\n🎉 RESUMEN FINAL - KAHNEMAN ENHANCEMENTS OPERATIVOS")
    print("=" * 80)
    print("✅ IMPLEMENTACIONES COMPLETADAS:")
    print("   1. 🧠 Detección automática 6 tipos sesgos cognitivos")
    print("   2. 🎯 Corrección regresiva con base rates históricas") 
    print("   3. 📊 Intervalos de confianza ajustados por calidad evidencia")
    print("   4. ⚠️ Sistema alertas anti-overconfidence")
    print("   5. 🔧 Protocolo Meehl: fórmulas estadísticas > juicios clínicos")
    
    print(f"\n📈 RESULTADOS ARGENTINA TEST:")
    print(f"   • Predicción original: {results['original_prediction']:.1%}")
    print(f"   • Predicción corregida: {results['corrected_prediction']:.1%}")
    print(f"   • Sesgos detectados: {results['biases_detected']}")
    print(f"   • Ajuste aplicado: {results['adjustment_made']:.1%}")
    
    print(f"\n💡 VALOR AGREGADO KAHNEMAN:")
    print("   • Protección contra overconfidence sistemático")
    print("   • Base rates históricas anclan predicciones en realidad")
    print("   • Detección automática elimina sesgos inconscientes") 
    print("   • Intervalos honestos mejoran toma de decisiones")
    print("   • Sistema 2 thinking aplicado automáticamente")

if __name__ == "__main__":
    main()