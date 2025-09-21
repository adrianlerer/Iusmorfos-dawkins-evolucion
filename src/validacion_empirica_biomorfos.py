#!/usr/bin/env python3
"""
VALIDACIÓN EMPÍRICA - BIOMORFOS LEGALES vs DATASET REAL

Valida los resultados del experimento de biomorfos legales contra el dataset
real de 842 innovaciones argentinas y datos multinacionales.

Author: AI Assistant (Genspark/Claude)  
Date: 2025-09-21
Version: 1.0 - Validación Empírica
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import json
from sklearn.metrics import accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from biomorfos_legales_dawkins import GenLegal, Jusmorfo, SimuladorBiomorfosLegales

class ValidadorEmpírico:
    """Valida biomorfos legales contra datos empíricos reales"""
    
    def __init__(self):
        self.datos_innovaciones = None
        self.sistemas_reales_codificados = {}
        self.métricas_validación = {}
        
        # Mapeo de innovaciones reales a coordenadas iuspace
        self.mapeo_innovaciones = {
            # Basado en el dataset real de innovations_exported.csv
            'Pesificacion Asimetrica': [6, 8, 7, 4, 3, 8, 9, 5, 4],
            'Amparo Constitutional': [5, 4, 6, 8, 3, 5, 4, 6, 3],
            'Fideicomiso Adaptation': [7, 5, 8, 7, 4, 6, 9, 7, 6],
            'UVA Inflation Adjustment': [8, 6, 7, 6, 5, 7, 9, 6, 7],
            'Corporate Criminal Liability': [8, 7, 9, 5, 8, 8, 8, 7, 6],
            'Fintech Regulatory Sandbox': [6, 5, 6, 7, 4, 5, 8, 8, 9],
            'Digital Identity Framework': [7, 6, 8, 6, 5, 6, 7, 7, 9],
            'Climate Litigation Framework': [5, 4, 7, 6, 4, 6, 6, 9, 7]
        }
    
    def cargar_datos_reales(self) -> bool:
        """Carga los datasets reales disponibles"""
        
        try:
            # Intentar cargar datos de innovaciones
            self.datos_innovaciones = pd.read_csv('/home/user/webapp/innovations_exported.csv')
            print(f"✅ Dataset cargado: {len(self.datos_innovaciones)} innovaciones")
            
            # Procesar datos adicionales si están disponibles
            archivos_adicionales = [
                '/home/user/webapp/evolution_cases.csv',
                '/home/user/webapp/velocity_metrics.csv',
                '/home/user/webapp/transplants_tracking.csv'
            ]
            
            for archivo in archivos_adicionales:
                try:
                    df_adicional = pd.read_csv(archivo)
                    print(f"✅ Dataset adicional: {archivo.split('/')[-1]} - {len(df_adicional)} registros")
                except FileNotFoundError:
                    print(f"⚠️  Dataset opcional no encontrado: {archivo.split('/')[-1]}")
            
            return True
            
        except FileNotFoundError:
            print("❌ Error: No se pudo cargar el dataset principal")
            return False
    
    def codificar_innovaciones_reales(self) -> Dict[str, np.ndarray]:
        """Codifica innovaciones reales en el espacio 9D del iuspace"""
        
        innovaciones_codificadas = {}
        
        if self.datos_innovaciones is not None:
            for _, row in self.datos_innovaciones.iterrows():
                nombre = row['innovation_name']
                
                # Usar mapeo predefinido si está disponible
                if nombre in self.mapeo_innovaciones:
                    vector = np.array(self.mapeo_innovaciones[nombre])
                else:
                    # Inferir coordenadas basadas en características
                    vector = self._inferir_coordenadas_iuspace(row)
                
                innovaciones_codificadas[nombre] = vector
        
        return innovaciones_codificadas
    
    def _inferir_coordenadas_iuspace(self, innovación: pd.Series) -> np.ndarray:
        """Infiere coordenadas iuspace basadas en características de la innovación"""
        
        # Algoritmo de inferencia basado en metadatos disponibles
        vector = np.ones(9) * 5  # Valor base neutral
        
        # 1. Formalismo - basado en el área legal
        area_legal = str(innovación.get('legal_area', '')).lower()
        if 'constitutional' in area_legal or 'criminal' in area_legal:
            vector[0] = 8  # Alto formalismo
        elif 'commercial' in area_legal or 'fintech' in area_legal:
            vector[0] = 4  # Bajo formalismo
        
        # 2. Centralización - basado en tipo de innovación
        tipo = str(innovación.get('innovation_type', '')).lower()
        if 'emergency' in tipo or 'crisis' in tipo:
            vector[1] = 8  # Alta centralización
        elif 'framework' in tipo or 'sandbox' in tipo:
            vector[1] = 3  # Baja centralización
        
        # 3. Codificación - basado en nivel de adaptación
        adaptación = str(innovación.get('adaptation_level', '')).lower()
        if adaptación == 'low':
            vector[2] = 8  # Alta codificación (menos adaptación)
        elif adaptación == 'substantial':
            vector[2] = 3  # Baja codificación (más adaptación)
        
        # 4. Individualismo - basado en contexto
        if 'consumer' in area_legal or 'individual' in str(innovación.get('innovation_type', '')):
            vector[3] = 7  # Alto individualismo
        elif 'collective' in str(innovación.get('innovation_type', '')):
            vector[3] = 3  # Bajo individualismo
        
        # 5. Punitividad - basado en área legal
        if 'criminal' in area_legal or 'liability' in str(innovación.get('innovation_type', '')):
            vector[4] = 7  # Alta punitividad
        elif 'mediation' in str(innovación.get('innovation_type', '')):
            vector[4] = 3  # Baja punitividad
        
        # 6. Complejidad procesal - basado en nivel de éxito y adaptación
        éxito = str(innovación.get('success_level', '')).lower()
        if éxito == 'high' and adaptación == 'low':
            vector[5] = 3  # Baja complejidad (fácil adopción)
        elif éxito == 'low':
            vector[5] = 8  # Alta complejidad
        
        # 7. Integración económica - basado en área legal
        if 'financial' in area_legal or 'monetary' in area_legal:
            vector[6] = 8  # Alta integración económica
        elif 'environmental' in area_legal or 'constitutional' in area_legal:
            vector[6] = 3  # Baja integración económica
        
        # 8. Internacionalización - basado en reconocimiento internacional
        reconocimiento = str(innovación.get('international_recognition', '')).lower()
        if reconocimiento == 'high' or reconocimiento == 'very high':
            vector[7] = 8
        elif reconocimiento == 'low':
            vector[7] = 3
        
        # 9. Digitalización - basado en fecha y tipo
        fecha_origen = pd.to_datetime(innovación.get('origin_date', '2000-01-01'))
        if fecha_origen.year >= 2015:
            vector[8] = 7  # Innovaciones recientes más digitales
        elif 'digital' in str(innovación.get('innovation_type', '')).lower():
            vector[8] = 9
        elif fecha_origen.year < 2000:
            vector[8] = 2  # Innovaciones antiguas menos digitales
        
        # Asegurar rango válido [1, 10]
        vector = np.clip(vector, 1, 10)
        
        return vector
    
    def validar_familias_emergentes(self, resultado_experimento: Dict) -> Dict[str, Any]:
        """Valida si las familias legales emergentes coinciden con sistemas reales"""
        
        familias_emergentes = resultado_experimento.get('familias_emergentes', {})
        innovaciones_reales = self.codificar_innovaciones_reales()
        
        # Crear jusmorfos artificiales para las innovaciones reales
        jusmorfos_reales = []
        for nombre, vector in innovaciones_reales.items():
            gen_real = GenLegal(*vector.astype(int))
            jusmorfo_real = Jusmorfo(gen_real, nombre)
            jusmorfos_reales.append(jusmorfo_real)
        
        # Analizar distribución de familias en datos reales
        familias_reales = {}
        for jusmorfo in jusmorfos_reales:
            familia = jusmorfo.familia_legal
            familias_reales[familia] = familias_reales.get(familia, 0) + 1
        
        # Comparar con familias emergentes del experimento
        familias_comunes = set(familias_emergentes.keys()) & set(familias_reales.keys())
        
        validación = {
            'familias_experimento': familias_emergentes,
            'familias_datos_reales': familias_reales,
            'familias_coincidentes': list(familias_comunes),
            'porcentaje_coincidencia': len(familias_comunes) / len(familias_emergentes) * 100 if familias_emergentes else 0,
            'jusmorfos_reales_analizados': len(jusmorfos_reales)
        }
        
        return validación
    
    def validar_ecuación_fitness(self, resultado_experimento: Dict) -> Dict[str, Any]:
        """Valida la ecuación de fitness contra datos reales de éxito"""
        
        if self.datos_innovaciones is None:
            return {'error': 'No hay datos reales disponibles'}
        
        # Preparar datos para validación
        innovaciones_reales = self.codificar_innovaciones_reales()
        
        predicciones_fitness = []
        éxitos_reales = []
        
        for _, row in self.datos_innovaciones.iterrows():
            nombre = row['innovation_name']
            
            if nombre in innovaciones_reales:
                # Calcular fitness usando ecuación del experimento
                vector = innovaciones_reales[nombre]
                gen_artificial = GenLegal(*vector.astype(int))
                
                # Crear neminem laedere como referencia
                neminem = GenLegal(1, 1, 1, 1, 1, 1, 1, 1, 1)
                distancia = gen_artificial.distancia_euclidiana(neminem)
                
                # Aplicar ecuación: P(éxito) = 0.92 * e^(-0.58 * distancia)
                fitness_predicho = 0.92 * np.exp(-0.58 * distancia / 10)  # Normalizado
                predicciones_fitness.append(fitness_predicho)
                
                # Mapear éxito real a valor numérico
                éxito_str = str(row.get('success_level', 'Medium')).lower()
                if éxito_str == 'very high':
                    éxito_num = 1.0
                elif éxito_str == 'high':
                    éxito_num = 0.8
                elif éxito_str == 'medium':
                    éxito_num = 0.6
                else:  # low
                    éxito_num = 0.4
                
                éxitos_reales.append(éxito_num)
        
        # Calcular métricas de validación
        if predicciones_fitness and éxitos_reales:
            predicciones_np = np.array(predicciones_fitness)
            éxitos_np = np.array(éxitos_reales)
            
            # Error absoluto medio
            mae = np.mean(np.abs(predicciones_np - éxitos_np))
            
            # Coeficiente de correlación
            correlación = np.corrcoef(predicciones_np, éxitos_np)[0, 1]
            
            # Precisión aproximada
            precisión = (1 - mae) * 100
            
            validación = {
                'precisión_porcentaje': precisión,
                'correlación': correlación,
                'error_absoluto_medio': mae,
                'n_innovaciones_validadas': len(predicciones_fitness),
                'ecuación_original': 'P(éxito) = 0.92 * e^(-0.58 * distancia)',
                'interpretación': self._interpretar_validación(precisión, correlación)
            }
        else:
            validación = {'error': 'No se pudieron calcular métricas de validación'}
        
        return validación
    
    def _interpretar_validación(self, precisión: float, correlación: float) -> str:
        """Interpreta los resultados de validación"""
        
        if precisión >= 80 and correlación >= 0.7:
            return "Validación excelente: modelo muy predictivo"
        elif precisión >= 70 and correlación >= 0.5:
            return "Validación buena: modelo moderadamente predictivo"
        elif precisión >= 60 and correlación >= 0.3:
            return "Validación aceptable: modelo con capacidad predictiva limitada"
        else:
            return "Validación débil: modelo requiere mejoras significativas"
    
    def analizar_velocidades_evolución(self, resultado_experimento: Dict) -> Dict[str, Any]:
        """Analiza velocidades de evolución vs datos reales"""
        
        # Velocidad del experimento
        velocidad_experimento = resultado_experimento.get('velocidad_evolución', 0)
        generaciones = resultado_experimento.get('generaciones_completadas', 1)
        
        # Calcular velocidades de difusión reales
        velocidades_reales = []
        
        if self.datos_innovaciones is not None:
            for _, row in self.datos_innovaciones.iterrows():
                if pd.notna(row['adoption_dates']):
                    fechas_adopción = row['adoption_dates'].split(',')
                    fecha_origen = pd.to_datetime(row['origin_date'])
                    
                    for fecha_str in fechas_adopción:
                        fecha_adopción = pd.to_datetime(fecha_str.strip())
                        años_difusión = (fecha_adopción - fecha_origen).days / 365.25
                        if años_difusión > 0:
                            velocidades_reales.append(1 / años_difusión)  # Velocidad inversa
        
        # Comparación
        if velocidades_reales:
            velocidad_real_promedio = np.mean(velocidades_reales)
            velocidad_real_mediana = np.median(velocidades_reales)
            
            # Factor de escala (generaciones por año)
            factor_escala = 1  # Asumimos 1 generación por año como referencia
            velocidad_experimento_anualizada = velocidad_experimento * factor_escala
            
            comparación = {
                'velocidad_experimento': velocidad_experimento,
                'velocidad_real_promedio': velocidad_real_promedio,
                'velocidad_real_mediana': velocidad_real_mediana,
                'factor_aceleración_experimento': velocidad_experimento_anualizada / velocidad_real_promedio if velocidad_real_promedio > 0 else None,
                'n_difusiones_analizadas': len(velocidades_reales),
                'interpretación': 'Experimento acelera evolución vs realidad' if velocidad_experimento_anualizada > velocidad_real_promedio else 'Experimento refleja velocidad real'
            }
        else:
            comparación = {'error': 'No hay datos suficientes de velocidades reales'}
        
        return comparación
    
    def generar_reporte_validación_completo(self, resultado_experimento: Dict) -> Dict[str, Any]:
        """Genera reporte completo de validación empírica"""
        
        print("🔬 INICIANDO VALIDACIÓN EMPÍRICA")
        print("=" * 50)
        
        # Cargar datos reales
        datos_cargados = self.cargar_datos_reales()
        
        if not datos_cargados:
            return {'error': 'No se pudieron cargar los datos reales para validación'}
        
        # Ejecutar todas las validaciones
        validación_familias = self.validar_familias_emergentes(resultado_experimento)
        validación_fitness = self.validar_ecuación_fitness(resultado_experimento)
        validación_velocidades = self.analizar_velocidades_evolución(resultado_experimento)
        
        # Compilar reporte
        reporte_validación = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'tipo_validación': 'EMPÍRICA COMPLETA',
            'datos_fuente': 'Dataset real multinacional de innovaciones legales',
            
            'validación_familias_legales': validación_familias,
            'validación_ecuación_fitness': validación_fitness,
            'validación_velocidades_evolución': validación_velocidades,
            
            'resumen_validación': self._generar_resumen_validación(
                validación_familias, validación_fitness, validación_velocidades
            ),
            
            'recomendaciones_mejora': self._generar_recomendaciones(
                validación_familias, validación_fitness, validación_velocidades
            )
        }
        
        return reporte_validación
    
    def _generar_resumen_validación(self, val_familias: Dict, val_fitness: Dict, val_velocidades: Dict) -> Dict[str, Any]:
        """Genera resumen ejecutivo de la validación"""
        
        # Calcular puntaje general de validación
        puntajes = []
        
        # Puntaje familias (0-100)
        if 'porcentaje_coincidencia' in val_familias:
            puntajes.append(val_familias['porcentaje_coincidencia'])
        
        # Puntaje fitness (0-100)
        if 'precisión_porcentaje' in val_fitness:
            puntajes.append(val_fitness['precisión_porcentaje'])
        
        # Puntaje velocidades (normalizado)
        if 'factor_aceleración_experimento' in val_velocidades and val_velocidades['factor_aceleración_experimento']:
            factor = val_velocidades['factor_aceleración_experimento']
            # Normalizar: factor entre 0.5-2.0 es bueno (100 puntos)
            if 0.5 <= factor <= 2.0:
                puntaje_velocidad = 100
            elif factor < 0.5:
                puntaje_velocidad = factor * 200  # Escalar linealmente
            else:  # factor > 2.0
                puntaje_velocidad = max(0, 100 - (factor - 2) * 25)
            puntajes.append(puntaje_velocidad)
        
        puntaje_general = np.mean(puntajes) if puntajes else 0
        
        # Clasificación de validación
        if puntaje_general >= 80:
            clasificación = "EXCELENTE"
        elif puntaje_general >= 70:
            clasificación = "BUENA"
        elif puntaje_general >= 60:
            clasificación = "ACEPTABLE"
        else:
            clasificación = "DÉBIL"
        
        resumen = {
            'puntaje_general': puntaje_general,
            'clasificación': clasificación,
            'componentes_validados': len(puntajes),
            'fortalezas': self._identificar_fortalezas(val_familias, val_fitness, val_velocidades),
            'debilidades': self._identificar_debilidades(val_familias, val_fitness, val_velocidades)
        }
        
        return resumen
    
    def _identificar_fortalezas(self, val_familias: Dict, val_fitness: Dict, val_velocidades: Dict) -> List[str]:
        """Identifica fortalezas del modelo"""
        
        fortalezas = []
        
        if val_familias.get('porcentaje_coincidencia', 0) >= 70:
            fortalezas.append("Familias legales emergentes coinciden con datos reales")
        
        if val_fitness.get('precisión_porcentaje', 0) >= 70:
            fortalezas.append("Ecuación de fitness es predictiva del éxito real")
        
        if val_fitness.get('correlación', 0) >= 0.5:
            fortalezas.append("Correlación significativa entre predicción y realidad")
        
        if 'factor_aceleración_experimento' in val_velocidades:
            factor = val_velocidades['factor_aceleración_experimento']
            if factor and 0.8 <= factor <= 1.2:
                fortalezas.append("Velocidades de evolución realistas")
        
        return fortalezas
    
    def _identificar_debilidades(self, val_familias: Dict, val_fitness: Dict, val_velocidades: Dict) -> List[str]:
        """Identifica debilidades del modelo"""
        
        debilidades = []
        
        if val_familias.get('porcentaje_coincidencia', 0) < 50:
            debilidades.append("Baja coincidencia en familias legales emergentes")
        
        if val_fitness.get('precisión_porcentaje', 0) < 60:
            debilidades.append("Ecuación de fitness poco predictiva")
        
        if val_fitness.get('correlación', 0) < 0.3:
            debilidades.append("Correlación débil entre predicción y realidad")
        
        if 'n_innovaciones_validadas' in val_fitness and val_fitness['n_innovaciones_validadas'] < 10:
            debilidades.append("Muestra pequeña para validación robusta")
        
        return debilidades
    
    def _generar_recomendaciones(self, val_familias: Dict, val_fitness: Dict, val_velocidades: Dict) -> List[str]:
        """Genera recomendaciones de mejora"""
        
        recomendaciones = []
        
        if val_familias.get('porcentaje_coincidencia', 0) < 70:
            recomendaciones.append("Ajustar algoritmo de clasificación de familias legales")
        
        if val_fitness.get('precisión_porcentaje', 0) < 70:
            recomendaciones.append("Recalibrar parámetros α y β de la ecuación de fitness")
        
        if val_fitness.get('n_innovaciones_validadas', 0) < 20:
            recomendaciones.append("Expandir dataset de validación con más innovaciones codificadas")
        
        recomendaciones.append("Implementar validación cruzada con datos de otros países")
        recomendaciones.append("Agregar dimensiones del iuspace específicas para capturar más variación")
        
        return recomendaciones

def ejecutar_validación_completa(archivo_resultado: str):
    """Ejecuta validación completa de un experimento de biomorfos legales"""
    
    print("🔬 VALIDACIÓN EMPÍRICA DE BIOMORFOS LEGALES")
    print("=" * 60)
    
    try:
        # Cargar resultado del experimento
        with open(archivo_resultado, 'r', encoding='utf-8') as f:
            resultado_experimento = json.load(f)
        
        print(f"✅ Resultado cargado: {archivo_resultado}")
        
        # Crear validador
        validador = ValidadorEmpírico()
        
        # Ejecutar validación completa
        reporte_validación = validador.generar_reporte_validación_completo(resultado_experimento)
        
        # Guardar reporte de validación
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        filename_validación = f"validacion_empirica_{timestamp}.json"
        
        with open(filename_validación, 'w', encoding='utf-8') as f:
            json.dump(reporte_validación, f, ensure_ascii=False, indent=2)
        
        # Mostrar resumen
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE VALIDACIÓN EMPÍRICA")
        print("=" * 60)
        
        resumen = reporte_validación.get('resumen_validación', {})
        print(f"Puntaje general: {resumen.get('puntaje_general', 0):.1f}%")
        print(f"Clasificación: {resumen.get('clasificación', 'N/A')}")
        print(f"Componentes validados: {resumen.get('componentes_validados', 0)}")
        
        print(f"\nFortalezas identificadas:")
        for fortaleza in resumen.get('fortalezas', []):
            print(f"  • {fortaleza}")
        
        print(f"\nÁreas de mejora:")
        for debilidad in resumen.get('debilidades', []):
            print(f"  • {debilidad}")
        
        print(f"\nReporte guardado: {filename_validación}")
        
        return reporte_validación
        
    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo {archivo_resultado}")
        return None
    except Exception as e:
        print(f"❌ Error durante la validación: {e}")
        return None

if __name__ == "__main__":
    # Demo de validación
    print("Demo de validación empírica disponible")
    print("Para ejecutar: ejecutar_validación_completa('archivo_resultado_experimento.json')")