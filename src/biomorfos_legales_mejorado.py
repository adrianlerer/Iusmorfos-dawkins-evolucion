#!/usr/bin/env python3
"""
BIOMORFOS LEGALES MEJORADOS - Versión corregida con evolución efectiva

Versión mejorada que resuelve el problema de estancamiento evolutivo
mediante una función de fitness balanceada que favorece la diversidad.

Author: AI Assistant (Genspark/Claude)
Date: 2025-09-21
Version: 2.0 - Biomorfos Legales Mejorados
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
import json
from datetime import datetime
import copy
import random
from collections import defaultdict
import pandas as pd

# Importar las clases base
from biomorfos_legales_dawkins import GenLegal, Jusmorfo

class SimuladorBiomorfosMejorado:
    """Versión mejorada del simulador con mejor función de fitness"""
    
    def __init__(self):
        self.población_actual: List[Jusmorfo] = []
        self.árbol_evolutivo: Dict[str, Dict] = {}
        self.generación_actual = 0
        self.neminem_laedere = self.crear_neminem_laedere()
        self.historia_evolución: List[Dict] = []
        self.familias_emergentes: Dict[str, List[Jusmorfo]] = defaultdict(list)
        
        # Parámetros mejorados
        self.tamaño_descendencia = 9
        self.rango_mutación = 1
        self.modo_selección = "automático"
        
        # Nuevos parámetros para evitar estancamiento
        self.factor_diversidad = 0.3  # Peso para favorecer diversidad
        self.factor_complejidad = 0.4  # Peso para favorecer complejidad
        self.factor_balance = 0.3     # Peso para balance entre dimensiones
        
    def crear_neminem_laedere(self) -> Jusmorfo:
        """Crea el sistema legal primordial"""
        gen_primordial = GenLegal(
            formalismo=1, centralización=1, codificación=1, individualismo=1, 
            punitividad=1, procesal_complejidad=1, economía_integración=1, 
            internacionalización=1, digitalización=1, generación=0, padre_id=None
        )
        
        return Jusmorfo(gen=gen_primordial, nombre="Neminem Laedere",
                       descripción="Principio legal primordial: 'No dañar a nadie'")
    
    def calcular_fitness_mejorado(self, jusmorfo: Jusmorfo) -> float:
        """Función de fitness mejorada que favorece evolución y diversidad"""
        
        vector = jusmorfo.gen.to_vector()
        
        # 1. Componente de complejidad (favorece sistemas más desarrollados)
        complejidad_normalizada = np.mean(vector) / 10.0
        fitness_complejidad = complejidad_normalizada
        
        # 2. Componente de diversidad (favorece diferencias con el ancestro)
        vector_ancestro = self.neminem_laedere.gen.to_vector()
        distancia_ancestro = np.linalg.norm(vector - vector_ancestro)
        fitness_diversidad = min(1.0, distancia_ancestro / 15.0)  # Normalizado
        
        # 3. Componente de balance (evita sistemas muy extremos)
        desviación_estándar = np.std(vector)
        fitness_balance = max(0.1, 1.0 - (desviación_estándar / 5.0))
        
        # 4. Bonificación por explorar nuevas dimensiones
        bonificación_exploración = 0.0
        for i, valor in enumerate(vector):
            if valor > vector_ancestro[i] + 1:  # Ha evolucionado esta dimensión
                bonificación_exploración += 0.05
        
        # Fitness combinado
        fitness_total = (
            fitness_complejidad * self.factor_complejidad +
            fitness_diversidad * self.factor_diversidad +
            fitness_balance * self.factor_balance +
            bonificación_exploración
        )
        
        return min(1.0, fitness_total)  # Cap a 1.0
    
    def reproducción_mejorada(self, padre: Jusmorfo) -> List[Jusmorfo]:
        """Reproducción con mutaciones más diversas"""
        descendientes = []
        
        for i in range(self.tamaño_descendencia):
            gen_hijo = copy.deepcopy(padre.gen)
            gen_hijo.generación = self.generación_actual + 1
            gen_hijo.padre_id = padre.gen.id_único
            gen_hijo.id_único = f"gen_g{gen_hijo.generación}_{i}_{datetime.now().microsecond + i}"
            
            # Estrategias de mutación variadas
            if i < 3:
                # Mutación conservadora: una dimensión ±1
                self._mutación_conservadora(gen_hijo)
            elif i < 6:
                # Mutación exploratoria: incrementar una dimensión significativamente
                self._mutación_exploratoria(gen_hijo)
            else:
                # Mutación equilibradora: ajustar hacia balance
                self._mutación_equilibradora(gen_hijo, padre)
            
            # Crear jusmorfo descendiente
            jusmorfo_hijo = Jusmorfo(gen=gen_hijo, nombre=f"Sistema_G{gen_hijo.generación}_{i+1}")
            descendientes.append(jusmorfo_hijo)
            
            # Registrar en árbol evolutivo
            self.árbol_evolutivo[gen_hijo.id_único] = {
                'jusmorfo': jusmorfo_hijo,
                'padre': padre.gen.id_único,
                'generación': gen_hijo.generación,
                'tipo_mutación': self._get_tipo_mutación(i)
            }
        
        return descendientes
    
    def _mutación_conservadora(self, gen: GenLegal):
        """Mutación conservadora: cambio pequeño en una dimensión"""
        dimensión = random.randint(0, 8)
        mutación = random.choice([-1, 1])
        
        vector = gen.to_vector()
        vector[dimensión] = np.clip(vector[dimensión] + mutación, 1, 10)
        self._actualizar_gen_desde_vector(gen, vector)
    
    def _mutación_exploratoria(self, gen: GenLegal):
        """Mutación exploratoria: incremento más significativo"""
        dimensión = random.randint(0, 8)
        incremento = random.randint(1, 3)  # Incremento de 1-3
        
        vector = gen.to_vector()
        vector[dimensión] = np.clip(vector[dimensión] + incremento, 1, 10)
        self._actualizar_gen_desde_vector(gen, vector)
    
    def _mutación_equilibradora(self, gen: GenLegal, padre: Jusmorfo):
        """Mutación equilibradora: busca balance entre dimensiones"""
        vector = gen.to_vector()
        
        # Identificar dimensión más baja y aumentarla
        dimensión_mínima = np.argmin(vector)
        vector[dimensión_mínima] = min(10, vector[dimensión_mínima] + 2)
        
        self._actualizar_gen_desde_vector(gen, vector)
    
    def _actualizar_gen_desde_vector(self, gen: GenLegal, vector: np.ndarray):
        """Actualiza un gen desde un vector"""
        gen.formalismo = int(vector[0])
        gen.centralización = int(vector[1])
        gen.codificación = int(vector[2])
        gen.individualismo = int(vector[3])
        gen.punitividad = int(vector[4])
        gen.procesal_complejidad = int(vector[5])
        gen.economía_integración = int(vector[6])
        gen.internacionalización = int(vector[7])
        gen.digitalización = int(vector[8])
    
    def _get_tipo_mutación(self, índice: int) -> str:
        """Retorna el tipo de mutación según el índice"""
        if índice < 3:
            return "conservadora"
        elif índice < 6:
            return "exploratoria"
        else:
            return "equilibradora"
    
    def selección_automática_mejorada(self, descendientes: List[Jusmorfo]) -> Jusmorfo:
        """Selección automática mejorada con fitness diverso"""
        
        for jusmorfo in descendientes:
            jusmorfo.gen.fitness = self.calcular_fitness_mejorado(jusmorfo)
        
        # Seleccionar el de mayor fitness
        mejor_descendiente = max(descendientes, key=lambda j: j.gen.fitness)
        
        print(f"\n🤖 SELECCIÓN MEJORADA - Generación {self.generación_actual + 1}")
        print(f"Seleccionado: {mejor_descendiente.nombre}")
        print(f"Fitness: {mejor_descendiente.gen.fitness:.3f}")
        print(f"Genes: {mejor_descendiente.gen.to_vector()}")
        print(f"Familia: {mejor_descendiente.familia_legal}")
        print(f"Complejidad: {mejor_descendiente.complejidad:.2f}")
        
        return mejor_descendiente
    
    def evolucionar_una_generación(self, padre: Jusmorfo) -> Jusmorfo:
        """Evoluciona una generación con algoritmo mejorado"""
        
        # Reproducción mejorada
        descendientes = self.reproducción_mejorada(padre)
        
        # Selección mejorada
        elegido = self.selección_automática_mejorada(descendientes)
        
        # Registrar en historia
        self.historia_evolución.append({
            'generación': self.generación_actual + 1,
            'padre': padre.nombre,
            'elegido': elegido.nombre,
            'familia': elegido.familia_legal,
            'complejidad': elegido.complejidad,
            'genes': elegido.gen.to_vector().tolist(),
            'fitness': elegido.gen.fitness,
            'fitness_componentes': self._analizar_fitness_componentes(elegido)
        })
        
        # Agregar a familias emergentes
        self.familias_emergentes[elegido.familia_legal].append(elegido)
        
        self.generación_actual += 1
        return elegido
    
    def _analizar_fitness_componentes(self, jusmorfo: Jusmorfo) -> Dict[str, float]:
        """Analiza los componentes del fitness para debugging"""
        
        vector = jusmorfo.gen.to_vector()
        vector_ancestro = self.neminem_laedere.gen.to_vector()
        
        complejidad = np.mean(vector) / 10.0
        diversidad = min(1.0, np.linalg.norm(vector - vector_ancestro) / 15.0)
        balance = max(0.1, 1.0 - (np.std(vector) / 5.0))
        
        return {
            'complejidad': complejidad,
            'diversidad': diversidad,
            'balance': balance
        }
    
    def ejecutar_experimento_mejorado(self, generaciones: int) -> Dict[str, Any]:
        """Ejecuta experimento mejorado"""
        
        print("🧬 BIOMORFOS LEGALES MEJORADOS - REPLICACIÓN DAWKINS v2.0")
        print("=" * 65)
        print(f"Comenzando con: {self.neminem_laedere.nombre}")
        print(f"Generaciones objetivo: {generaciones}")
        print("Función de fitness: Complejidad + Diversidad + Balance")
        
        # Comenzar evolución
        actual = self.neminem_laedere
        
        for gen in range(generaciones):
            actual = self.evolucionar_una_generación(actual)
            
            # Reporte cada 5 generaciones
            if (gen + 1) % 5 == 0:
                self.generar_reporte_intermedio()
        
        # Resultado final
        resultado = self.generar_reporte_final(actual)
        return resultado
    
    def generar_reporte_intermedio(self):
        """Reporte de progreso mejorado"""
        
        print(f"\n📊 REPORTE INTERMEDIO - Generación {self.generación_actual}")
        print("-" * 55)
        
        if self.historia_evolución:
            último = self.historia_evolución[-1]
            print(f"Sistema actual: {último['familia']} (fitness: {último['fitness']:.3f})")
            print(f"Complejidad: {último['complejidad']:.2f}")
            print(f"Genes actuales: {último['genes']}")
            
            # Componentes de fitness
            if 'fitness_componentes' in último:
                comp = último['fitness_componentes']
                print(f"Fitness - Complejidad: {comp['complejidad']:.3f}, "
                      f"Diversidad: {comp['diversidad']:.3f}, "
                      f"Balance: {comp['balance']:.3f}")
        
        # Familias emergentes
        print("Familias legales emergentes:")
        for familia, jusmorfos in self.familias_emergentes.items():
            print(f"  • {familia}: {len(jusmorfos)} apariciones")
    
    def generar_reporte_final(self, sistema_final: Jusmorfo) -> Dict[str, Any]:
        """Genera reporte final mejorado"""
        
        return {
            'experimento': 'Biomorfos Legales Mejorados - Replicación Dawkins v2.0',
            'timestamp': datetime.now().isoformat(),
            'generaciones_completadas': self.generación_actual,
            'modo_selección': 'automático_mejorado',
            
            'parámetros_fitness': {
                'factor_complejidad': self.factor_complejidad,
                'factor_diversidad': self.factor_diversidad,
                'factor_balance': self.factor_balance
            },
            
            'sistema_inicial': {
                'nombre': self.neminem_laedere.nombre,
                'genes': self.neminem_laedere.gen.to_vector().tolist(),
                'complejidad': self.neminem_laedere.complejidad
            },
            
            'sistema_final': {
                'nombre': sistema_final.nombre,
                'familia_legal': sistema_final.familia_legal,
                'genes': sistema_final.gen.to_vector().tolist(),
                'complejidad': sistema_final.complejidad,
                'características': sistema_final.características,
                'fitness': sistema_final.gen.fitness
            },
            
            'evolución_completa': {
                'distancia_total_recorrida': sistema_final.gen.distancia_euclidiana(self.neminem_laedere.gen),
                'incremento_complejidad': sistema_final.complejidad - self.neminem_laedere.complejidad,
                'familias_exploradas': len(self.familias_emergentes),
                'historia_generaciones': self.historia_evolución
            },
            
            'familias_emergentes': {
                familia: len(jusmorfos) for familia, jusmorfos in self.familias_emergentes.items()
            },
            
            'análisis_evolución': self.analizar_patrones_evolución(),
            'velocidad_evolución': self.calcular_velocidad_evolución_mejorada(),
            'predicción_complejidad_moderna': self.predecir_generaciones_modernas()
        }
    
    def analizar_patrones_evolución(self) -> Dict[str, Any]:
        """Analiza patrones en la evolución observada"""
        
        if len(self.historia_evolución) < 2:
            return {}
        
        # Analizar tendencias por dimensión
        dimensiones = ['formalismo', 'centralización', 'codificación', 'individualismo',
                      'punitividad', 'procesal_complejidad', 'economía_integración',
                      'internacionalización', 'digitalización']
        
        tendencias = {}
        for i, dim in enumerate(dimensiones):
            valores = [h['genes'][i] for h in self.historia_evolución]
            if len(valores) > 1:
                tendencia = valores[-1] - valores[0]  # Cambio total
                tendencias[dim] = {
                    'cambio_total': tendencia,
                    'valor_inicial': valores[0],
                    'valor_final': valores[-1],
                    'dirección': 'creciente' if tendencia > 0 else 'decreciente' if tendencia < 0 else 'estable'
                }
        
        # Identificar dimensiones más evolutivas
        cambios_absolutos = [(dim, abs(data['cambio_total'])) for dim, data in tendencias.items()]
        cambios_absolutos.sort(key=lambda x: x[1], reverse=True)
        
        return {
            'tendencias_por_dimensión': tendencias,
            'dimensiones_más_evolutivas': [dim for dim, _ in cambios_absolutos[:3]],
            'estabilidad_general': np.mean([abs(data['cambio_total']) for data in tendencias.values()])
        }
    
    def calcular_velocidad_evolución_mejorada(self) -> float:
        """Calcula velocidad de evolución considerando múltiples factores"""
        
        if len(self.historia_evolución) < 2:
            return 0.0
        
        # Velocidad de cambio de complejidad
        complejidades = [h['complejidad'] for h in self.historia_evolución]
        vel_complejidad = abs(complejidades[-1] - complejidades[0]) / len(complejidades)
        
        # Velocidad de diversificación de familias
        vel_familias = len(self.familias_emergentes) / len(self.historia_evolución)
        
        return (vel_complejidad + vel_familias) / 2
    
    def predecir_generaciones_modernas(self) -> int:
        """Predice generaciones necesarias para complejidad moderna"""
        
        complejidad_moderna = 7.5  # Objetivo más realista
        
        if not self.historia_evolución:
            return float('inf')
        
        complejidad_actual = self.historia_evolución[-1]['complejidad']
        velocidad = self.calcular_velocidad_evolución_mejorada()
        
        if velocidad <= 0:
            return float('inf')
        
        generaciones_restantes = (complejidad_moderna - complejidad_actual) / velocidad
        return max(0, int(generaciones_restantes))

def ejecutar_experimento_mejorado(generaciones: int = 30):
    """Ejecuta el experimento mejorado"""
    
    print("🚀 EJECUTANDO BIOMORFOS LEGALES MEJORADOS")
    print("=" * 60)
    
    simulador = SimuladorBiomorfosMejorado()
    resultado = simulador.ejecutar_experimento_mejorado(generaciones)
    
    # Guardar resultado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"biomorfos_mejorado_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(resultado, f, ensure_ascii=False, indent=2)
    
    # Mostrar resumen
    print("\n" + "=" * 60)
    print("🎉 EXPERIMENTO MEJORADO COMPLETADO")
    print("=" * 60)
    
    sistema_final = resultado['sistema_final']
    evolución = resultado['evolución_completa']
    
    print(f"Sistema final: {sistema_final['familia_legal']}")
    print(f"Complejidad: {resultado['sistema_inicial']['complejidad']:.2f} → {sistema_final['complejidad']:.2f}")
    print(f"Genes finales: {sistema_final['genes']}")
    print(f"Distancia evolutiva: {evolución['distancia_total_recorrida']:.2f}")
    print(f"Familias emergentes: {len(resultado['familias_emergentes'])}")
    
    for familia, count in resultado['familias_emergentes'].items():
        print(f"  • {familia}: {count} apariciones")
    
    print(f"\nResultado guardado: {filename}")
    
    return resultado, simulador

if __name__ == "__main__":
    resultado, simulador = ejecutar_experimento_mejorado(30)