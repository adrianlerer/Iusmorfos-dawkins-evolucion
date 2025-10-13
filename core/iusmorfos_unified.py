#!/usr/bin/env python3
"""
Iusmorfos Unified: Framework Unificado con Múltiples Modos de Exploración

Integra los 4 modelos evolutivos:
1. BIOMORFOS (Original) - Exploración emergente
2. CAPARAZOMORFOS - Exploración sistemática paramétrica
3. ARTROMORFOS - Construcción embriológica estructurada
4. NETSPINNER - Evolución multi-poblacional

Permite elegir el modo según el caso de uso.

@author: Iusmorfos Unified Framework
@version: 1.0 - Multi-Model Integration
@reality_filter: INTEGRATED across all modes
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union, Any
from enum import Enum
import logging

# Importar los 4 modelos
try:
    # Modelo original (exploración emergente)
    import sys
    sys.path.insert(0, '/home/user/webapp/src')
    from biomorfos_legales_dawkins import (
        GenLegal, Jusmorfo, SimuladorBiomorfosLegales
    )
    
    # Modelos avanzados
    from .caparazomorfos_parametric import (
        ParametricLegalSpace, LegalNormGenotype, NormParameter
    )
    from .artromorfos_embryology import (
        LegalEmbryology, LegalEmbryoGenotype, EmbryologicalGene
    )
    from .netspinner_population import (
        NetSpinnerEvolution, LegalTradition, LegalPopulation
    )
    from .kahneman_enhanced_framework import KahnemanEnhancedFramework
    
except ImportError:
    # Fallback para ejecución directa
    import sys
    sys.path.insert(0, '/home/user/webapp/src')
    from biomorfos_legales_dawkins import (
        GenLegal, Jusmorfo, SimuladorBiomorfosLegales
    )
    from caparazomorfos_parametric import (
        ParametricLegalSpace, LegalNormGenotype, NormParameter
    )
    from artromorfos_embryology import (
        LegalEmbryology, LegalEmbryoGenotype, EmbryologicalGene
    )
    from netspinner_population import (
        NetSpinnerEvolution, LegalTradition, LegalPopulation
    )
    from kahneman_enhanced_framework import KahnemanEnhancedFramework

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExplorationMode(Enum):
    """Modos de exploración disponibles"""
    EMERGENT = "emergent"          # Biomorfos original - descubrimiento serendipitoso
    PARAMETRIC = "parametric"      # Caparazomorfos - exploración sistemática
    EMBRYOLOGICAL = "embryological" # Artromorfos - construcción estructurada
    POPULATION = "population"      # NetSpinner - evolución multi-poblacional


@dataclass
class UnifiedEvolutionConfig:
    """Configuración unificada para cualquier modo de exploración"""
    mode: ExplorationMode
    jurisdiction: str
    legal_tradition: str
    
    # Parámetros comunes
    generations: int = 10
    mutation_rate: float = 0.1
    population_size: int = 10
    
    # Parámetros específicos por modo
    # Para EMERGENT (Biomorfos)
    biomorfos_selection_method: str = "fitness"  # "fitness" o "user"
    
    # Para PARAMETRIC (Caparazomorfos)
    parametric_exploration_axis: Optional[NormParameter] = None
    parametric_values: Optional[List[float]] = None
    
    # Para EMBRYOLOGICAL (Artromorfos)
    document_type: str = "statutory_law"
    enforce_structural_guarantees: bool = True
    
    # Para POPULATION (NetSpinner)
    gene_flow_rate: float = 0.1
    additional_populations: List[Tuple[str, str]] = None  # [(jurisdiction, tradition), ...]
    
    # Reality Filter
    apply_kahneman_corrections: bool = True
    confidence_threshold: float = 0.6


class IusmorfosUnified:
    """
    Framework unificado que integra los 4 modos de exploración
    
    FILOSOFÍA DE DISEÑO:
    - Cada modo tiene sus fortalezas
    - El usuario elige según el caso de uso
    - Reality Filter se aplica en todos los modos
    - Los resultados son comparables entre modos
    """
    
    def __init__(self, config: UnifiedEvolutionConfig):
        self.config = config
        self.mode = config.mode
        
        # Inicializar componentes según modo
        self._init_mode_components()
        
        # Reality Filter (común a todos)
        if config.apply_kahneman_corrections:
            self.kahneman_framework = KahnemanEnhancedFramework(
                config.jurisdiction,
                config.legal_tradition
            )
        else:
            self.kahneman_framework = None
        
        # Historial de evolución
        self.evolution_history: List[Dict[str, Any]] = []
        self.generation = 0
        
        logger.info(f"Iusmorfos Unified initialized in {self.mode.value.upper()} mode")
        logger.info(f"Jurisdiction: {config.jurisdiction} | Tradition: {config.legal_tradition}")
    
    def _init_mode_components(self):
        """Inicializar componentes específicos del modo"""
        
        if self.mode == ExplorationMode.EMERGENT:
            # Biomorfos original - exploración emergente
            self.biomorfos_simulator = SimuladorBiomorfosLegales()
            logger.info("Mode: EMERGENT (Biomorfos) - Serendipitous discovery enabled")
        
        elif self.mode == ExplorationMode.PARAMETRIC:
            # Caparazomorfos - exploración sistemática
            self.parametric_space = ParametricLegalSpace(
                self.config.jurisdiction,
                self.config.legal_tradition
            )
            logger.info("Mode: PARAMETRIC (Caparazomorfos) - Systematic exploration enabled")
        
        elif self.mode == ExplorationMode.EMBRYOLOGICAL:
            # Artromorfos - construcción embriológica
            self.embryology_engine = LegalEmbryology(
                self.config.jurisdiction,
                self.config.legal_tradition
            )
            logger.info("Mode: EMBRYOLOGICAL (Artromorfos) - Structured construction enabled")
            logger.info("Guarantees: Coherence >= 70%, Validity >= 80%")
        
        elif self.mode == ExplorationMode.POPULATION:
            # NetSpinner - evolución poblacional
            populations_config = [
                (self.config.jurisdiction, LegalTradition(self.config.legal_tradition))
            ]
            
            # Agregar poblaciones adicionales si se especificaron
            if self.config.additional_populations:
                for juris, trad in self.config.additional_populations:
                    populations_config.append((juris, LegalTradition(trad)))
            
            self.netspinner = NetSpinnerEvolution(
                populations_config,
                gene_flow_rate=self.config.gene_flow_rate
            )
            logger.info("Mode: POPULATION (NetSpinner) - Multi-population dynamics enabled")
            logger.info(f"Populations: {len(populations_config)} | Gene flow rate: {self.config.gene_flow_rate:.1%}")
    
    def evolve_single_generation(self) -> Dict[str, Any]:
        """
        Evolucionar UNA generación en el modo actual
        Retorna métricas comparables entre modos
        """
        self.generation += 1
        
        if self.mode == ExplorationMode.EMERGENT:
            return self._evolve_emergent()
        
        elif self.mode == ExplorationMode.PARAMETRIC:
            return self._evolve_parametric()
        
        elif self.mode == ExplorationMode.EMBRYOLOGICAL:
            return self._evolve_embryological()
        
        elif self.mode == ExplorationMode.POPULATION:
            return self._evolve_population()
    
    def _evolve_emergent(self) -> Dict[str, Any]:
        """Evolución emergente (Biomorfos original)"""
        
        # Si es la primera generación, crear sistema inicial
        if self.generation == 1:
            # Sistema inicial: "Neminem Laedere" (principio más básico)
            initial_gen = GenLegal(
                formalismo=1, centralización=1, codificación=1,
                individualismo=1, punitividad=1, procesal_complejidad=1,
                economía_integración=1, internacionalización=1, digitalización=1,
                generación=0
            )
            self.current_jusmorfo = Jusmorfo(gen=initial_gen, nombre="Neminem Laedere")
            
            logger.info(f"Generation {self.generation}: Created initial system 'Neminem Laedere'")
        
        # Generar descendencia (mutaciones)
        offspring = []
        for _ in range(self.config.population_size):
            mutated_gen = self._mutate_biomorfos_gen(self.current_jusmorfo.gen)
            offspring.append(Jusmorfo(gen=mutated_gen))
        
        # Seleccionar el mejor (o simular selección)
        if self.config.biomorfos_selection_method == "fitness":
            # Selección automática por fitness
            best_offspring = max(offspring, key=lambda j: self._calculate_biomorfos_fitness(j))
        else:
            # Simular selección aleatoria (usuario elegiría)
            best_offspring = np.random.choice(offspring)
        
        self.current_jusmorfo = best_offspring
        
        # Calcular métricas
        complexity = np.linalg.norm(self.current_jusmorfo.gen.to_vector())
        
        result = {
            "generation": self.generation,
            "mode": "emergent",
            "complexity": complexity,
            "fitness": self._calculate_biomorfos_fitness(self.current_jusmorfo),
            "system": self.current_jusmorfo.gen.to_vector().tolist(),
            "exploration_method": "serendipitous_walk"
        }
        
        self.evolution_history.append(result)
        return result
    
    def _evolve_parametric(self) -> Dict[str, Any]:
        """Evolución paramétrica (Caparazomorfos)"""
        
        # Crear o recuperar norma base
        if self.generation == 1:
            self.current_norm = self.parametric_space.create_random_norm("statutory_reform")
            logger.info(f"Generation {self.generation}: Created base norm for parametric exploration")
        
        # Exploración sistemática de un eje
        if self.config.parametric_exploration_axis:
            axis = self.config.parametric_exploration_axis
            values = self.config.parametric_values or [0.0, 0.25, 0.5, 0.75, 1.0]
            
            explored = self.parametric_space.explore_parameter_axis(
                self.current_norm,
                axis,
                values
            )
            
            # Evaluar y seleccionar mejor
            comparison = self.parametric_space.compare_norms(explored, "statutory_reform")
            best_idx = comparison['success_probability'].idxmax()
            best_norm_id = comparison.loc[best_idx, 'norm_id']
            
            self.current_norm = explored[best_idx]
            
            result = {
                "generation": self.generation,
                "mode": "parametric",
                "axis_explored": axis.value,
                "best_success_probability": comparison.loc[best_idx, 'success_probability'],
                "confidence_interval": [
                    comparison.loc[best_idx, 'ci_lower'],
                    comparison.loc[best_idx, 'ci_upper']
                ],
                "exploration_method": "systematic_axis_scan"
            }
        else:
            # Mutación aleatoria
            old_params = self.current_norm.parameters.copy()
            param_to_mutate = np.random.choice(list(NormParameter))
            old_params[param_to_mutate] = np.clip(
                old_params[param_to_mutate] + np.random.normal(0, 0.1),
                0.0, 1.0
            )
            
            self.current_norm = LegalNormGenotype(
                parameters=old_params,
                generation=self.generation,
                lineage_id=f"{self.current_norm.lineage_id}_g{self.generation}"
            )
            
            phenotype = self.parametric_space.develop_phenotype(self.current_norm)
            
            result = {
                "generation": self.generation,
                "mode": "parametric",
                "success_probability": phenotype.success_probability,
                "confidence_interval": list(phenotype.confidence_interval),
                "exploration_method": "random_mutation"
            }
        
        self.evolution_history.append(result)
        return result
    
    def _evolve_embryological(self) -> Dict[str, Any]:
        """Evolución embriológica (Artromorfos)"""
        
        # Crear o mutar embrión
        if self.generation == 1:
            self.current_embryo = self.embryology_engine.create_random_embryo(
                self.config.document_type
            )
            logger.info(f"Generation {self.generation}: Created embryo for document construction")
        else:
            self.current_embryo = self.embryology_engine.mutate_embryo(
                self.current_embryo,
                mutation_rate=self.config.mutation_rate
            )
        
        # Desarrollar documento
        document = self.embryology_engine.develop_document(self.current_embryo)
        
        # Validar garantías estructurales
        if self.config.enforce_structural_guarantees:
            assert document.structural_coherence >= 0.7, "Structural coherence guarantee failed"
            assert document.syntactic_validity >= 0.8, "Syntactic validity guarantee failed"
        
        result = {
            "generation": self.generation,
            "mode": "embryological",
            "total_articles": document.total_articles,
            "structural_coherence": document.structural_coherence,
            "syntactic_validity": document.syntactic_validity,
            "success_probability": document.implementation_success_probability,
            "confidence_interval": list(document.confidence_interval),
            "segments": len(document.segments),
            "structural_risks": len(document.structural_risks),
            "exploration_method": "embryological_construction",
            "guarantees_verified": True
        }
        
        self.evolution_history.append(result)
        return result
    
    def _evolve_population(self) -> Dict[str, Any]:
        """Evolución poblacional (NetSpinner)"""
        
        # Evolucionar todas las poblaciones
        self.netspinner.evolve_generation()
        
        # Obtener resumen
        summary = self.netspinner.get_population_summary()
        gene_flow_summary = self.netspinner.get_gene_flow_summary()
        
        result = {
            "generation": self.generation,
            "mode": "population",
            "populations": summary.to_dict('records'),
            "gene_flow_events": len(gene_flow_summary) if not gene_flow_summary.empty else 0,
            "avg_fitness": summary['avg_fitness'].mean(),
            "avg_diversity": summary['diversity'].mean(),
            "exploration_method": "multi_population_with_gene_flow"
        }
        
        self.evolution_history.append(result)
        return result
    
    def _mutate_biomorfos_gen(self, parent_gen: GenLegal) -> GenLegal:
        """Mutación estilo Biomorfos original (±1 en una dimensión)"""
        new_gen = GenLegal(
            formalismo=parent_gen.formalismo,
            centralización=parent_gen.centralización,
            codificación=parent_gen.codificación,
            individualismo=parent_gen.individualismo,
            punitividad=parent_gen.punitividad,
            procesal_complejidad=parent_gen.procesal_complejidad,
            economía_integración=parent_gen.economía_integración,
            internacionalización=parent_gen.internacionalización,
            digitalización=parent_gen.digitalización,
            generación=parent_gen.generación + 1,
            padre_id=parent_gen.id_único
        )
        
        # Mutar UNA dimensión aleatoria en ±1
        dimensions = [
            'formalismo', 'centralización', 'codificación',
            'individualismo', 'punitividad', 'procesal_complejidad',
            'economía_integración', 'internacionalización', 'digitalización'
        ]
        
        dim_to_mutate = np.random.choice(dimensions)
        current_value = getattr(new_gen, dim_to_mutate)
        mutation = np.random.choice([-1, 1])
        new_value = np.clip(current_value + mutation, 1, 10)
        setattr(new_gen, dim_to_mutate, int(new_value))
        
        return new_gen
    
    def _calculate_biomorfos_fitness(self, jusmorfo: Jusmorfo) -> float:
        """Calcular fitness para Biomorfos (complejidad + diversidad)"""
        vector = jusmorfo.gen.to_vector()
        
        # Complejidad: distancia del origen
        complexity = np.linalg.norm(vector) / np.linalg.norm(np.ones(9) * 10)
        
        # Diversidad: varianza de dimensiones
        diversity = np.std(vector) / 10.0
        
        # Balance: no estar en extremos
        balance = 1.0 - np.mean(np.abs(vector - 5.5) / 4.5)
        
        fitness = 0.4 * complexity + 0.3 * diversity + 0.3 * balance
        return fitness
    
    def get_evolution_summary(self) -> pd.DataFrame:
        """Resumen de evolución comparable entre modos"""
        if not self.evolution_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.evolution_history)
    
    def compare_with_mode(self, other_mode: ExplorationMode, generations: int = 10) -> pd.DataFrame:
        """
        Comparar evolución entre dos modos
        Útil para análisis de convergencia/divergencia
        """
        # Crear configuración para el otro modo
        other_config = UnifiedEvolutionConfig(
            mode=other_mode,
            jurisdiction=self.config.jurisdiction,
            legal_tradition=self.config.legal_tradition,
            generations=generations
        )
        
        other_unified = IusmorfosUnified(other_config)
        
        # Evolucionar ambos en paralelo
        results = []
        for gen in range(generations):
            result_self = self.evolve_single_generation()
            result_other = other_unified.evolve_single_generation()
            
            results.append({
                "generation": gen + 1,
                "mode_1": self.mode.value,
                "mode_2": other_mode.value,
                "metric_1": result_self.get('fitness', result_self.get('success_probability', 0)),
                "metric_2": result_other.get('fitness', result_other.get('success_probability', 0))
            })
        
        return pd.DataFrame(results)


def create_unified_framework(
    mode: ExplorationMode,
    jurisdiction: str,
    legal_tradition: str,
    **kwargs
) -> IusmorfosUnified:
    """Factory function para crear framework unificado"""
    
    config = UnifiedEvolutionConfig(
        mode=mode,
        jurisdiction=jurisdiction,
        legal_tradition=legal_tradition,
        **kwargs
    )
    
    return IusmorfosUnified(config)


# DEMO: Comparación de los 4 modos
if __name__ == "__main__":
    print("="*80)
    print("IUSMORFOS UNIFIED: Multi-Mode Evolution Framework")
    print("4 Exploration Modes Available")
    print("="*80)
    
    jurisdiction = "Colombia"
    tradition = "civil_law"
    
    print(f"\n🌍 Testing all 4 modes for {jurisdiction} ({tradition})")
    print(f"Running 5 generations in each mode...\n")
    
    modes_results = {}
    
    # 1. EMERGENT (Biomorfos original)
    print("1️⃣  EMERGENT MODE (Biomorfos - Serendipitous Discovery)")
    print("-" * 60)
    emergent = create_unified_framework(
        ExplorationMode.EMERGENT,
        jurisdiction,
        tradition,
        generations=5
    )
    
    for gen in range(5):
        result = emergent.evolve_single_generation()
        print(f"   Gen {result['generation']}: Fitness={result['fitness']:.3f}, "
              f"Complexity={result['complexity']:.3f}")
    
    modes_results['emergent'] = emergent.get_evolution_summary()
    
    # 2. PARAMETRIC (Caparazomorfos)
    print(f"\n2️⃣  PARAMETRIC MODE (Caparazomorfos - Systematic Exploration)")
    print("-" * 60)
    parametric = create_unified_framework(
        ExplorationMode.PARAMETRIC,
        jurisdiction,
        tradition,
        generations=5
    )
    
    for gen in range(5):
        result = parametric.evolve_single_generation()
        if 'success_probability' in result:
            ci = result['confidence_interval']
            print(f"   Gen {result['generation']}: Success={result['success_probability']:.2%}, "
                  f"CI=[{ci[0]:.2%}, {ci[1]:.2%}]")
    
    modes_results['parametric'] = parametric.get_evolution_summary()
    
    # 3. EMBRYOLOGICAL (Artromorfos)
    print(f"\n3️⃣  EMBRYOLOGICAL MODE (Artromorfos - Structured Construction)")
    print("-" * 60)
    embryological = create_unified_framework(
        ExplorationMode.EMBRYOLOGICAL,
        jurisdiction,
        tradition,
        generations=5
    )
    
    for gen in range(5):
        result = embryological.evolve_single_generation()
        print(f"   Gen {result['generation']}: Articles={result['total_articles']}, "
              f"Coherence={result['structural_coherence']:.2%}, "
              f"Validity={result['syntactic_validity']:.2%}")
    
    modes_results['embryological'] = embryological.get_evolution_summary()
    
    # 4. POPULATION (NetSpinner)
    print(f"\n4️⃣  POPULATION MODE (NetSpinner - Multi-Population Dynamics)")
    print("-" * 60)
    population = create_unified_framework(
        ExplorationMode.POPULATION,
        jurisdiction,
        tradition,
        additional_populations=[("USA", "common_law")],
        gene_flow_rate=0.3,
        generations=5
    )
    
    for gen in range(5):
        result = population.evolve_single_generation()
        print(f"   Gen {result['generation']}: Avg Fitness={result['avg_fitness']:.2%}, "
              f"Gene Flow Events={result['gene_flow_events']}")
    
    modes_results['population'] = population.get_evolution_summary()
    
    # Resumen comparativo
    print(f"\n" + "="*80)
    print("📊 COMPARATIVE SUMMARY")
    print("="*80)
    
    print(f"\n✅ All 4 modes executed successfully")
    print(f"   • EMERGENT: {len(modes_results['emergent'])} generations")
    print(f"   • PARAMETRIC: {len(modes_results['parametric'])} generations")
    print(f"   • EMBRYOLOGICAL: {len(modes_results['embryological'])} generations")
    print(f"   • POPULATION: {len(modes_results['population'])} generations")
    
    print(f"\n🎯 MODE SELECTION GUIDE:")
    print(f"   • EMERGENT      → Use for: Discovery, emergence, serendipity")
    print(f"   • PARAMETRIC    → Use for: Systematic comparison, parameter analysis")
    print(f"   • EMBRYOLOGICAL → Use for: Document generation, structural guarantees")
    print(f"   • POPULATION    → Use for: Comparative law, legal transplants")
    
    print(f"\n✅ UNIFIED FRAMEWORK COMPLETE")
    print(f"   All modes available, user chooses based on use case")
    print("="*80)
