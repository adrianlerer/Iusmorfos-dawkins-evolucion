#!/usr/bin/env python3
"""
BIOMORFOS LEGALES: REPLICACIÓN DEL EXPERIMENTO DE DAWKINS PARA SISTEMAS LEGALES

Implementación exacta del programa de biomorfos de Dawkins aplicado a la evolución
de sistemas legales en el espacio 9-dimensional (iuspace).

Comenzando con "Neminem laedere" (el principio legal más básico),
evolucionamos sistemas legales complejos mediante selección acumulativa.

Basado en:
- Dawkins, R. (1986). The Blind Watchmaker, Chapter 3: "Accumulating Small Change"
- Framework IusSpace de 9 dimensiones para sistemas legales
- Dataset empírico de 842 innovaciones argentinas

Author: AI Assistant (Genspark/Claude)
Date: 2025-09-21
Version: 1.0 - Biomorfos Legales Completo
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

@dataclass
class GenLegal:
    """Un gen legal en el espacio 9-dimensional del iuspace"""
    
    # Las 9 dimensiones del iuspace
    formalismo: int = 1          # 1-10: Formalidad vs flexibilidad
    centralización: int = 1      # 1-10: Centralizado vs descentralizado  
    codificación: int = 1        # 1-10: Codificado vs jurisprudencial
    individualismo: int = 1      # 1-10: Individual vs colectivo
    punitividad: int = 1         # 1-10: Punitivo vs restaurativo
    procesal_complejidad: int = 1 # 1-10: Simple vs complejo procesal
    economía_integración: int = 1 # 1-10: Separado vs integrado económico
    internacionalización: int = 1 # 1-10: Nacional vs internacional
    digitalización: int = 1       # 1-10: Tradicional vs digital
    
    # Metadatos evolutivos
    generación: int = 0
    padre_id: Optional[str] = None
    fitness: float = 0.0
    id_único: str = ""
    
    def __post_init__(self):
        if not self.id_único:
            self.id_único = f"gen_{id(self)}_{datetime.now().microsecond}"
    
    def to_vector(self) -> np.ndarray:
        """Convierte el gen a vector 9D"""
        return np.array([
            self.formalismo, self.centralización, self.codificación,
            self.individualismo, self.punitividad, self.procesal_complejidad,
            self.economía_integración, self.internacionalización, self.digitalización
        ])
    
    def distancia_euclidiana(self, otro: 'GenLegal') -> float:
        """Calcula distancia euclidiana entre dos genes"""
        return np.linalg.norm(self.to_vector() - otro.to_vector())
    
    def es_válido(self) -> bool:
        """Verifica que todos los genes estén en rango válido [1,10]"""
        vector = self.to_vector()
        return np.all(vector >= 1) and np.all(vector <= 10)

@dataclass 
class Jusmorfo:
    """Un sistema legal completo (equivalente a un biomorfo)"""
    
    gen: GenLegal
    nombre: str = ""
    descripción: str = ""
    familia_legal: str = "Primitivo"
    complejidad: float = 0.0
    características: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.nombre:
            self.nombre = f"Sistema_{self.gen.id_único[:8]}"
        self.calcular_características()
    
    def calcular_características(self):
        """SUBRUTINA DESARROLLO: Convierte 9 genes en sistema legal visible"""
        v = self.gen.to_vector()
        self.características = []
        
        # Análisis de cada dimensión para generar características visibles
        
        # 1. Formalismo (1-10)
        if v[0] <= 3:
            self.características.append("Derecho consuetudinario dominante")
        elif v[0] <= 7:
            self.características.append("Sistema mixto formal-informal")
        else:
            self.características.append("Derecho altamente codificado")
        
        # 2. Centralización (1-10)  
        if v[1] <= 3:
            self.características.append("Federalismo descentralizado")
        elif v[1] <= 7:
            self.características.append("Sistema semi-federal")
        else:
            self.características.append("Estado unitario centralizado")
        
        # 3. Codificación (1-10)
        if v[2] <= 3:
            self.características.append("Common law puro")
        elif v[2] <= 7:
            self.características.append("Sistema mixto civil-common")
        else:
            self.características.append("Civil law codificado")
        
        # 4. Individualismo (1-10)
        if v[3] <= 3:
            self.características.append("Derechos colectivos primarios")
        elif v[3] <= 7:
            self.características.append("Balance individual-colectivo")
        else:
            self.características.append("Derechos individuales absolutos")
        
        # 5. Punitividad (1-10)
        if v[4] <= 3:
            self.características.append("Justicia restaurativa")
        elif v[4] <= 7:
            self.características.append("Sistema punitivo moderado")
        else:
            self.características.append("Sistema altamente punitivo")
        
        # 6. Complejidad procesal (1-10)
        if v[5] <= 3:
            self.características.append("Procedimientos simples")
        elif v[5] <= 7:
            self.características.append("Complejidad procesal media")
        else:
            self.características.append("Procedimientos muy complejos")
        
        # 7. Integración económica (1-10)
        if v[6] <= 3:
            self.características.append("Derecho-economía separados")
        elif v[6] <= 7:
            self.características.append("Integración económica parcial")
        else:
            self.características.append("Derecho económico integrado")
        
        # 8. Internacionalización (1-10)
        if v[7] <= 3:
            self.características.append("Sistema nacional cerrado")
        elif v[7] <= 7:
            self.características.append("Apertura internacional selectiva")
        else:
            self.características.append("Derecho internacional integrado")
        
        # 9. Digitalización (1-10)
        if v[8] <= 3:
            self.características.append("Procedimientos tradicionales")
        elif v[8] <= 7:
            self.características.append("Digitalización parcial")
        else:
            self.características.append("Sistema completamente digital")
        
        # Calcular complejidad total
        self.complejidad = np.mean(v)
        
        # Determinar familia legal emergente
        self.determinar_familia_legal()
    
    def determinar_familia_legal(self):
        """Identifica familia legal basada en combinación de genes"""
        v = self.gen.to_vector()
        
        # Clasificación basada en patrones conocidos
        if v[2] <= 3 and v[0] <= 5:  # Common law + baja formalización
            self.familia_legal = "Common Law"
        elif v[2] >= 7 and v[0] >= 6:  # Alta codificación + formalismo
            self.familia_legal = "Civil Law"
        elif v[3] <= 4 and v[4] <= 4:  # Colectivo + restaurativo
            self.familia_legal = "Derecho Comunitario"
        elif v[4] >= 7 and v[1] >= 7:  # Punitivo + centralizado
            self.familia_legal = "Derecho Autoritario"
        elif v[7] >= 7 and v[8] >= 7:  # Internacional + digital
            self.familia_legal = "Derecho Transnacional"
        elif v[6] >= 7:  # Alta integración económica
            self.familia_legal = "Derecho Económico"
        else:
            self.familia_legal = "Sistema Híbrido"

class SimuladorBiomorfosLegales:
    """Simulador principal del experimento de biomorfos legales"""
    
    def __init__(self):
        self.población_actual: List[Jusmorfo] = []
        self.árbol_evolutivo: Dict[str, Dict] = {}
        self.generación_actual = 0
        self.neminem_laedere = self.crear_neminem_laedere()
        self.historia_evolución: List[Dict] = []
        self.familias_emergentes: Dict[str, List[Jusmorfo]] = defaultdict(list)
        
        # Parámetros del experimento
        self.tamaño_descendencia = 9  # Como en Dawkins original
        self.rango_mutación = 1  # Mutaciones ±1
        self.modo_selección = "manual"  # "manual" o "automático"
        
    def crear_neminem_laedere(self) -> Jusmorfo:
        """Crea el sistema legal primordial: 'Neminem laedere' (No dañar a nadie)"""
        gen_primordial = GenLegal(
            formalismo=1,
            centralización=1, 
            codificación=1,
            individualismo=1,
            punitividad=1,
            procesal_complejidad=1,
            economía_integración=1,
            internacionalización=1,
            digitalización=1,
            generación=0,
            padre_id=None
        )
        
        jusmorfo_primordial = Jusmorfo(
            gen=gen_primordial,
            nombre="Neminem Laedere",
            descripción="Principio legal primordial: 'No dañar a nadie'. Sistema legal más básico posible."
        )
        
        return jusmorfo_primordial
    
    def reproducción(self, padre: Jusmorfo) -> List[Jusmorfo]:
        """SUBRUTINA REPRODUCCIÓN: Genera descendientes con mutaciones ±1"""
        descendientes = []
        
        for i in range(self.tamaño_descendencia):
            # Crear gen hijo copiando del padre
            gen_hijo = copy.deepcopy(padre.gen)
            gen_hijo.generación = self.generación_actual + 1
            gen_hijo.padre_id = padre.gen.id_único
            gen_hijo.id_único = f"gen_g{gen_hijo.generación}_{i}_{datetime.now().microsecond}"
            
            # Aplicar mutación aleatoria a UNA dimensión (como en Dawkins)
            dimensión_mutada = random.randint(0, 8)
            mutación = random.choice([-1, 1])  # ±1
            
            # Mutar la dimensión seleccionada
            vector_genes = gen_hijo.to_vector()
            vector_genes[dimensión_mutada] = np.clip(
                vector_genes[dimensión_mutada] + mutación, 1, 10
            )
            
            # Actualizar el gen
            gen_hijo.formalismo = vector_genes[0]
            gen_hijo.centralización = vector_genes[1]
            gen_hijo.codificación = vector_genes[2]
            gen_hijo.individualismo = vector_genes[3]
            gen_hijo.punitividad = vector_genes[4]
            gen_hijo.procesal_complejidad = vector_genes[5]
            gen_hijo.economía_integración = vector_genes[6]
            gen_hijo.internacionalización = vector_genes[7]
            gen_hijo.digitalización = vector_genes[8]
            
            # Crear jusmorfo descendiente
            jusmorfo_hijo = Jusmorfo(
                gen=gen_hijo,
                nombre=f"Sistema_G{gen_hijo.generación}_{i+1}"
            )
            
            descendientes.append(jusmorfo_hijo)
            
            # Agregar al árbol evolutivo
            self.árbol_evolutivo[gen_hijo.id_único] = {
                'jusmorfo': jusmorfo_hijo,
                'padre': padre.gen.id_único,
                'generación': gen_hijo.generación,
                'mutación_aplicada': f"dim_{dimensión_mutada}_{mutación:+d}"
            }
        
        return descendientes
    
    def calcular_fitness_automático(self, jusmorfo: Jusmorfo) -> float:
        """Calcula fitness automático basado en ecuación empírica"""
        # Usar ecuación derivada del dataset: P(éxito) = 0.92 * e^(-0.58 * distancia)
        distancia_desde_origen = jusmorfo.gen.distancia_euclidiana(self.neminem_laedere.gen)
        
        # Componentes de fitness
        fitness_distancia = 0.92 * np.exp(-0.58 * distancia_desde_origen / 10)  # Normalizada
        fitness_complejidad = jusmorfo.complejidad / 10  # Normalizada
        fitness_balance = 1 - (np.std(jusmorfo.gen.to_vector()) / 10)  # Penalizar extremos
        
        # Fitness combinado
        fitness_total = (fitness_distancia * 0.4 + 
                        fitness_complejidad * 0.4 + 
                        fitness_balance * 0.2)
        
        return fitness_total
    
    def selección_manual(self, descendientes: List[Jusmorfo]) -> Jusmorfo:
        """SUBRUTINA SELECCIÓN: Permite al usuario elegir el descendiente favorito"""
        print(f"\n🧬 GENERACIÓN {self.generación_actual + 1}")
        print("=" * 60)
        print("Selecciona el sistema legal más prometedor:\n")
        
        for i, jusmorfo in enumerate(descendientes):
            print(f"[{i+1}] {jusmorfo.nombre}")
            print(f"    Familia: {jusmorfo.familia_legal}")
            print(f"    Complejidad: {jusmorfo.complejidad:.2f}")
            print(f"    Genes: {jusmorfo.gen.to_vector()}")
            print(f"    Características clave:")
            for característica in jusmorfo.características[:3]:  # Mostrar solo las 3 primeras
                print(f"      • {característica}")
            print()
        
        while True:
            try:
                selección = int(input(f"Elige tu favorito (1-{len(descendientes)}): ")) - 1
                if 0 <= selección < len(descendientes):
                    return descendientes[selección]
                else:
                    print("Selección inválida. Intenta de nuevo.")
            except ValueError:
                print("Por favor ingresa un número válido.")
    
    def selección_automática(self, descendientes: List[Jusmorfo]) -> Jusmorfo:
        """Selección automática basada en fitness"""
        for jusmorfo in descendientes:
            jusmorfo.gen.fitness = self.calcular_fitness_automático(jusmorfo)
        
        # Seleccionar el de mayor fitness
        mejor_descendiente = max(descendientes, key=lambda j: j.gen.fitness)
        
        print(f"\n🤖 SELECCIÓN AUTOMÁTICA - Generación {self.generación_actual + 1}")
        print(f"Seleccionado: {mejor_descendiente.nombre}")
        print(f"Fitness: {mejor_descendiente.gen.fitness:.3f}")
        print(f"Familia: {mejor_descendiente.familia_legal}")
        
        return mejor_descendiente
    
    def evolucionar_una_generación(self, padre: Jusmorfo) -> Jusmorfo:
        """Evoluciona una generación completa"""
        # REPRODUCCIÓN
        descendientes = self.reproducción(padre)
        
        # SELECCIÓN
        if self.modo_selección == "manual":
            elegido = self.selección_manual(descendientes)
        else:
            elegido = self.selección_automática(descendientes)
        
        # Registrar en historia
        self.historia_evolución.append({
            'generación': self.generación_actual + 1,
            'padre': padre.nombre,
            'elegido': elegido.nombre,
            'familia': elegido.familia_legal,
            'complejidad': elegido.complejidad,
            'genes': elegido.gen.to_vector().tolist(),
            'fitness': elegido.gen.fitness
        })
        
        # Agregar a familia emergente
        self.familias_emergentes[elegido.familia_legal].append(elegido)
        
        self.generación_actual += 1
        return elegido
    
    def ejecutar_experimento(self, generaciones: int, modo: str = "manual") -> Dict[str, Any]:
        """Ejecuta el experimento completo de biomorfos legales"""
        self.modo_selección = modo
        
        print("🧬 BIOMORFOS LEGALES - REPLICACIÓN DE DAWKINS")
        print("=" * 60)
        print(f"Comenzando con: {self.neminem_laedere.nombre}")
        print(f"Modo de selección: {modo}")
        print(f"Generaciones objetivo: {generaciones}")
        print(f"Descendientes por generación: {self.tamaño_descendencia}")
        
        # Comenzar con Neminem Laedere
        actual = self.neminem_laedere
        
        # Evolucionar
        for gen in range(generaciones):
            actual = self.evolucionar_una_generación(actual)
            
            if gen % 5 == 0 and gen > 0:  # Reporte cada 5 generaciones
                self.generar_reporte_intermedio()
        
        # Generar reporte final
        resultado = self.generar_reporte_final(actual)
        
        return resultado
    
    def generar_reporte_intermedio(self):
        """Genera reporte de progreso intermedio"""
        print(f"\n📊 REPORTE INTERMEDIO - Generación {self.generación_actual}")
        print("-" * 50)
        
        # Familias emergentes
        print("Familias legales emergentes:")
        for familia, jusmorfos in self.familias_emergentes.items():
            print(f"  • {familia}: {len(jusmorfos)} apariciones")
        
        # Complejidad promedio
        if self.historia_evolución:
            complejidad_promedio = np.mean([h['complejidad'] for h in self.historia_evolución])
            print(f"Complejidad promedio: {complejidad_promedio:.2f}")
    
    def generar_reporte_final(self, sistema_final: Jusmorfo) -> Dict[str, Any]:
        """Genera reporte final del experimento"""
        
        reporte = {
            'experimento': 'Biomorfos Legales - Replicación Dawkins',
            'timestamp': datetime.now().isoformat(),
            'generaciones_completadas': self.generación_actual,
            'modo_selección': self.modo_selección,
            
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
            
            'análisis_convergencia': self.analizar_convergencias(),
            'velocidad_evolución': self.calcular_velocidad_evolución(),
            'predicción_complejidad_moderna': self.predecir_generaciones_complejidad_moderna()
        }
        
        return reporte
    
    def analizar_convergencias(self) -> Dict[str, Any]:
        """Analiza convergencias independientes en la evolución"""
        convergencias = {}
        
        # Buscar familias que aparecieron múltiples veces
        for familia, jusmorfos in self.familias_emergentes.items():
            if len(jusmorfos) > 1:
                # Analizar si son convergencias independientes
                generaciones = [j.gen.generación for j in jusmorfos]
                if max(generaciones) - min(generaciones) > 3:  # Separadas por más de 3 generaciones
                    convergencias[familia] = {
                        'apariciones': len(jusmorfos),
                        'generaciones': generaciones,
                        'es_convergencia': True
                    }
        
        return convergencias
    
    def calcular_velocidad_evolución(self) -> float:
        """Calcula la velocidad promedio de evolución"""
        if len(self.historia_evolución) < 2:
            return 0.0
        
        velocidades = []
        for i in range(1, len(self.historia_evolución)):
            cambio_complejidad = (self.historia_evolución[i]['complejidad'] - 
                                self.historia_evolución[i-1]['complejidad'])
            velocidades.append(abs(cambio_complejidad))
        
        return np.mean(velocidades)
    
    def predecir_generaciones_complejidad_moderna(self) -> int:
        """Predice cuántas generaciones se necesitan para alcanzar complejidad legal moderna"""
        # Complejidad moderna estimada (basada en sistemas legales contemporáneos)
        complejidad_moderna = 8.5  # Sistemas como EE.UU., UE, etc.
        
        if not self.historia_evolución:
            return 0
        
        complejidad_actual = self.historia_evolución[-1]['complejidad']
        velocidad_promedio = self.calcular_velocidad_evolución()
        
        if velocidad_promedio == 0:
            return float('inf')
        
        generaciones_restantes = (complejidad_moderna - complejidad_actual) / velocidad_promedio
        return max(0, int(generaciones_restantes))
    
    def visualizar_evolución(self):
        """Genera visualizaciones de la evolución"""
        if not self.historia_evolución:
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Evolución de complejidad
        generaciones = [h['generación'] for h in self.historia_evolución]
        complejidades = [h['complejidad'] for h in self.historia_evolución]
        
        ax1.plot(generaciones, complejidades, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Generación')
        ax1.set_ylabel('Complejidad del Sistema')
        ax1.set_title('Evolución de la Complejidad Legal')
        ax1.grid(True, alpha=0.3)
        
        # 2. Distribución de familias legales
        familias = list(self.familias_emergentes.keys())
        counts = [len(jusmorfos) for jusmorfos in self.familias_emergentes.values()]
        
        ax2.bar(familias, counts, color='skyblue', edgecolor='navy')
        ax2.set_xlabel('Familia Legal')
        ax2.set_ylabel('Número de Apariciones')
        ax2.set_title('Familias Legales Emergentes')
        ax2.tick_params(axis='x', rotation=45)
        
        # 3. Trayectoria en espacio 9D (proyección 2D)
        if len(self.historia_evolución) > 1:
            genes_historia = np.array([h['genes'] for h in self.historia_evolución])
            
            # PCA para proyectar a 2D
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            genes_2d = pca.fit_transform(genes_historia)
            
            ax3.plot(genes_2d[:, 0], genes_2d[:, 1], 'r-o', linewidth=2, markersize=4)
            ax3.scatter(genes_2d[0, 0], genes_2d[0, 1], color='green', s=100, label='Inicio')
            ax3.scatter(genes_2d[-1, 0], genes_2d[-1, 1], color='red', s=100, label='Final')
            ax3.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} varianza)')
            ax3.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} varianza)')
            ax3.set_title('Trayectoria Evolutiva (PCA)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # 4. Fitness a lo largo del tiempo
        if self.modo_selección == "automático":
            fitness_valores = [h.get('fitness', 0) for h in self.historia_evolución]
            ax4.plot(generaciones, fitness_valores, 'g-o', linewidth=2, markersize=6)
            ax4.set_xlabel('Generación')
            ax4.set_ylabel('Fitness')
            ax4.set_title('Evolución del Fitness')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('biomorfos_legales_evolución.png', dpi=300, bbox_inches='tight')
        plt.show()

def ejecutar_experimento_completo():
    """Función principal para ejecutar el experimento completo"""
    print("🧬 INICIANDO EXPERIMENTO DE BIOMORFOS LEGALES")
    print("Replicación exacta del experimento de Dawkins para sistemas legales")
    print("=" * 70)
    
    # Crear simulador
    simulador = SimuladorBiomorfosLegales()
    
    # Configurar experimento
    print("\nConfiguración del experimento:")
    print("1. Manual (tú eliges cada generación)")
    print("2. Automático (selección por fitness)")
    
    while True:
        try:
            modo_input = input("\nElige modo (1 o 2): ").strip()
            if modo_input == "1":
                modo = "manual"
                break
            elif modo_input == "2":
                modo = "automático"
                break
            else:
                print("Por favor ingresa 1 o 2")
        except KeyboardInterrupt:
            print("\nExperimento cancelado.")
            return
    
    # Número de generaciones
    while True:
        try:
            generaciones = int(input("Número de generaciones (recomendado: 20-50): "))
            if 1 <= generaciones <= 1000:
                break
            else:
                print("Por favor ingresa un número entre 1 y 1000")
        except ValueError:
            print("Por favor ingresa un número válido")
        except KeyboardInterrupt:
            print("\nExperimento cancelado.")
            return
    
    # Ejecutar experimento
    try:
        resultado = simulador.ejecutar_experimento(generaciones, modo)
        
        # Guardar resultado
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"biomorfos_legales_resultado_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(resultado, f, ensure_ascii=False, indent=2)
        
        # Mostrar resumen final
        print("\n" + "=" * 70)
        print("🎉 EXPERIMENTO COMPLETADO")
        print("=" * 70)
        
        sistema_final = resultado['sistema_final']
        evolución = resultado['evolución_completa']
        
        print(f"Sistema final: {sistema_final['nombre']}")
        print(f"Familia legal: {sistema_final['familia_legal']}")
        print(f"Complejidad alcanzada: {sistema_final['complejidad']:.2f}")
        print(f"Distancia recorrida: {evolución['distancia_total_recorrida']:.2f}")
        print(f"Incremento complejidad: {evolución['incremento_complejidad']:.2f}")
        print(f"Familias exploradas: {evolución['familias_exploradas']}")
        
        predicción = resultado['predicción_complejidad_moderna']
        if predicción < 1000:
            print(f"Generaciones para complejidad moderna: {predicción}")
        
        print(f"\nResultado guardado en: {filename}")
        
        # Generar visualizaciones
        print("\nGenerando visualizaciones...")
        simulador.visualizar_evolución()
        
        return resultado
        
    except KeyboardInterrupt:
        print("\n\nExperimento interrumpido por el usuario.")
        return None
    except Exception as e:
        print(f"\nError durante el experimento: {e}")
        return None

if __name__ == "__main__":
    resultado = ejecutar_experimento_completo()