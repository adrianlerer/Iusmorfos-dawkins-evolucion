#!/usr/bin/env python3
"""
CompetitiveArena Module - Iusmorfos Framework v4.0
===================================================

Implementa la modelización de dinámicas competitivas entre formas institucionales
basada en la metodología de biomorphs de Dawkins aplicada al sistema legal.

Author: Adrian Lerer & Claude (AI Assistant)
Version: 4.0
Date: September 2024

Key Concepts:
- Competitive exclusion modeling between institutional forms (iusmorfos)
- Power-law distributions (γ=2.3) in legal citation networks
- Fitness landscapes in 9-dimensional IusSpace
- Selection pressure modeling
- Extinction and speciation events
- Coevolutionary dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import scipy.stats as stats
from scipy.optimize import minimize
from collections import defaultdict
import networkx as nx
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IusmorfoSpecies:
    """
    Represents a species of institutional form in the competitive arena.
    """
    
    def __init__(self, 
                 species_id: str,
                 genotype: Dict[str, float],
                 phenotype: Dict[str, float],
                 fitness: float = 0.0,
                 population_size: int = 1,
                 genealogy: List[str] = None):
        """
        Initialize a new iusmorfo species.
        
        Args:
            species_id: Unique identifier for the species
            genotype: Constitutional parameters (9D vector)
            phenotype: Observed implementation characteristics
            fitness: Current fitness in competitive environment
            population_size: Number of instances in current environment
            genealogy: List of ancestor species IDs
        """
        self.species_id = species_id
        self.genotype = genotype
        self.phenotype = phenotype
        self.fitness = fitness
        self.population_size = population_size
        self.genealogy = genealogy or []
        self.birth_time = 0
        self.extinction_time = None
        self.mutation_rate = 0.05
        self.selection_pressure = {}
        
    def mutate(self, mutation_strength: float = 0.1) -> 'IusmorfoSpecies':
        """
        Create a mutated version of this species.
        
        Args:
            mutation_strength: Standard deviation of mutations
            
        Returns:
            New mutated IusmorfoSpecies
        """
        new_genotype = {}
        for key, value in self.genotype.items():
            # Apply Gaussian mutation
            mutation = np.random.normal(0, mutation_strength)
            new_value = np.clip(value + mutation, -1.0, 1.0)
            new_genotype[key] = new_value
            
        # Update genealogy
        new_genealogy = self.genealogy + [self.species_id]
        
        # Generate new species ID
        mutation_id = f"{self.species_id}_m{len(new_genealogy)}"
        
        return IusmorfoSpecies(
            species_id=mutation_id,
            genotype=new_genotype,
            phenotype=self.phenotype.copy(),  # Will be updated by environment
            fitness=0.0,  # Will be recalculated
            population_size=1,
            genealogy=new_genealogy
        )
        
    def crossover(self, other: 'IusmorfoSpecies', crossover_rate: float = 0.5) -> 'IusmorfoSpecies':
        """
        Create offspring through crossover with another species.
        
        Args:
            other: Partner species for crossover
            crossover_rate: Probability of taking gene from other parent
            
        Returns:
            New hybrid IusmorfoSpecies
        """
        new_genotype = {}
        
        for key in self.genotype.keys():
            if np.random.random() < crossover_rate:
                new_genotype[key] = other.genotype.get(key, self.genotype[key])
            else:
                new_genotype[key] = self.genotype[key]
                
        # Combine genealogies
        new_genealogy = list(set(self.genealogy + other.genealogy + [self.species_id, other.species_id]))
        
        # Generate hybrid ID
        hybrid_id = f"{self.species_id}_x_{other.species_id}"
        
        return IusmorfoSpecies(
            species_id=hybrid_id,
            genotype=new_genotype,
            phenotype={},  # Will be calculated by environment
            fitness=0.0,
            population_size=1,
            genealogy=new_genealogy
        )

class CompetitiveArena:
    """
    Main class for modeling competitive dynamics between institutional forms.
    Implements evolutionary dynamics in 9-dimensional IusSpace.
    """
    
    def __init__(self, 
                 dimensions: int = 9,
                 carrying_capacity: int = 100,
                 mutation_rate: float = 0.05,
                 selection_strength: float = 1.0):
        """
        Initialize the competitive arena.
        
        Args:
            dimensions: Number of dimensions in IusSpace (default 9)
            carrying_capacity: Maximum total population size
            mutation_rate: Probability of mutations per generation
            selection_strength: Intensity of selection pressure
        """
        self.dimensions = dimensions
        self.carrying_capacity = carrying_capacity
        self.mutation_rate = mutation_rate
        self.selection_strength = selection_strength
        
        # Population management
        self.species_population: Dict[str, IusmorfoSpecies] = {}
        self.generation = 0
        self.extinction_log = []
        self.speciation_log = []
        
        # Fitness landscape
        self.fitness_landscape = self._initialize_fitness_landscape()
        
        # Competition matrix (pairwise interaction strengths)
        self.competition_matrix = defaultdict(lambda: defaultdict(float))
        
        # Environmental parameters
        self.environmental_pressure = {}
        self.resource_distribution = np.ones(dimensions) / dimensions
        
        # Citation network (power-law distribution γ=2.3)
        self.citation_network = nx.DiGraph()
        self.gamma = 2.3  # Power-law exponent
        
        # Tracking and metrics
        self.diversity_history = []
        self.fitness_history = []
        self.population_history = []
        
        logger.info(f"Initialized CompetitiveArena with {dimensions}D space, K={carrying_capacity}")
        
    def _initialize_fitness_landscape(self) -> Dict[str, Any]:
        """
        Initialize the fitness landscape in 9D IusSpace.
        
        Returns:
            Dictionary defining fitness landscape parameters
        """
        landscape = {
            'peaks': [],  # Local fitness maxima
            'valleys': [],  # Local fitness minima
            'gradients': np.random.randn(self.dimensions, self.dimensions),
            'noise_level': 0.1,
            'temporal_drift': 0.01  # How much landscape changes per generation
        }
        
        # Add some random fitness peaks and valleys
        for _ in range(3):  # 3 fitness peaks
            peak_location = np.random.uniform(-1, 1, self.dimensions)
            peak_height = np.random.uniform(0.7, 1.0)
            peak_width = np.random.uniform(0.2, 0.5)
            landscape['peaks'].append({
                'location': peak_location,
                'height': peak_height,
                'width': peak_width
            })
            
        for _ in range(2):  # 2 fitness valleys  
            valley_location = np.random.uniform(-1, 1, self.dimensions)
            valley_depth = np.random.uniform(0.1, 0.3)
            valley_width = np.random.uniform(0.3, 0.6)
            landscape['valleys'].append({
                'location': valley_location,
                'depth': valley_depth,
                'width': valley_width
            })
            
        return landscape
        
    def add_species(self, species: IusmorfoSpecies) -> bool:
        """
        Add a new species to the arena.
        
        Args:
            species: IusmorfoSpecies to add
            
        Returns:
            True if successfully added, False if rejected
        """
        if species.species_id in self.species_population:
            logger.warning(f"Species {species.species_id} already exists")
            return False
            
        # Calculate initial fitness
        species.fitness = self.calculate_fitness(species)
        species.birth_time = self.generation
        
        self.species_population[species.species_id] = species
        
        # Add to citation network
        self.citation_network.add_node(species.species_id, **species.genotype)
        
        logger.info(f"Added species {species.species_id} with fitness {species.fitness:.3f}")
        return True
        
    def calculate_fitness(self, species: IusmorfoSpecies) -> float:
        """
        Calculate fitness of a species in current environment.
        
        Args:
            species: IusmorfoSpecies to evaluate
            
        Returns:
            Fitness value (0.0 to 1.0)
        """
        genotype_vector = np.array(list(species.genotype.values()))
        
        # Base fitness from landscape
        base_fitness = 0.5  # Neutral baseline
        
        # Add contributions from fitness peaks
        for peak in self.fitness_landscape['peaks']:
            distance = np.linalg.norm(genotype_vector - peak['location'])
            contribution = peak['height'] * np.exp(-distance**2 / peak['width']**2)
            base_fitness += contribution
            
        # Subtract contributions from fitness valleys
        for valley in self.fitness_landscape['valleys']:
            distance = np.linalg.norm(genotype_vector - valley['location'])
            penalty = valley['depth'] * np.exp(-distance**2 / valley['width']**2)
            base_fitness -= penalty
            
        # Competition effects
        competition_penalty = 0.0
        for other_id, other_species in self.species_population.items():
            if other_id != species.species_id:
                competition_strength = self.get_competition_strength(species, other_species)
                competition_penalty += competition_strength * other_species.population_size
                
        # Environmental pressure
        env_fitness = 1.0
        for pressure_type, pressure_value in self.environmental_pressure.items():
            if pressure_type in species.genotype:
                env_fitness *= 1.0 - abs(species.genotype[pressure_type] - pressure_value) * 0.2
                
        # Combine all fitness components
        total_fitness = base_fitness * env_fitness * (1.0 - competition_penalty * 0.1)
        
        # Add noise
        noise = np.random.normal(0, self.fitness_landscape['noise_level'])
        total_fitness += noise
        
        # Ensure fitness is in valid range
        return np.clip(total_fitness, 0.0, 1.0)
        
    def get_competition_strength(self, 
                               species1: IusmorfoSpecies, 
                               species2: IusmorfoSpecies) -> float:
        """
        Calculate competition strength between two species.
        
        Args:
            species1, species2: Species to compare
            
        Returns:
            Competition strength (0.0 to 1.0)
        """
        # Calculate genotype similarity
        genotype1 = np.array(list(species1.genotype.values()))
        genotype2 = np.array(list(species2.genotype.values()))
        
        # Euclidean distance in genotype space
        distance = np.linalg.norm(genotype1 - genotype2)
        max_distance = np.sqrt(self.dimensions * 4)  # Maximum possible distance in [-1,1]^D space
        
        # Convert distance to competition strength (closer = more competition)
        similarity = 1.0 - (distance / max_distance)
        competition = similarity ** 2  # Squared to make competition more local
        
        return competition
        
    def evolve_generation(self) -> Dict[str, Any]:
        """
        Evolve the population by one generation.
        
        Returns:
            Dictionary with evolution statistics
        """
        self.generation += 1
        logger.info(f"Evolving generation {self.generation}")
        
        # Update fitness for all species
        for species in self.species_population.values():
            species.fitness = self.calculate_fitness(species)
            
        # Selection phase
        survivors = self._selection()
        
        # Reproduction phase
        offspring = self._reproduction(survivors)
        
        # Mutation phase
        mutants = self._mutation(offspring)
        
        # Population regulation
        self._population_regulation()
        
        # Update landscape (temporal drift)
        self._update_landscape()
        
        # Update citation network
        self._update_citation_network()
        
        # Record statistics
        stats = self._calculate_generation_stats()
        self.diversity_history.append(stats['diversity'])
        self.fitness_history.append(stats['mean_fitness'])
        self.population_history.append(stats['total_population'])
        
        return stats
        
    def _selection(self) -> List[IusmorfoSpecies]:
        """
        Perform selection based on fitness.
        
        Returns:
            List of surviving species
        """
        survivors = []
        
        for species in self.species_population.values():
            # Fitness-proportional selection
            survival_probability = (species.fitness ** self.selection_strength)
            
            # Environmental carrying capacity effects
            total_population = sum(s.population_size for s in self.species_population.values())
            if total_population > self.carrying_capacity:
                carrying_capacity_effect = self.carrying_capacity / total_population
                survival_probability *= carrying_capacity_effect
                
            # Stochastic survival
            if np.random.random() < survival_probability:
                survivors.append(species)
            else:
                # Record extinction
                species.extinction_time = self.generation
                self.extinction_log.append({
                    'species_id': species.species_id,
                    'extinction_time': self.generation,
                    'fitness_at_extinction': species.fitness,
                    'genealogy': species.genealogy
                })
                logger.info(f"Species {species.species_id} went extinct (fitness: {species.fitness:.3f})")
                
        return survivors
        
    def _reproduction(self, survivors: List[IusmorfoSpecies]) -> List[IusmorfoSpecies]:
        """
        Generate offspring through reproduction.
        
        Args:
            survivors: List of surviving species
            
        Returns:
            List of new offspring
        """
        offspring = []
        
        for species in survivors:
            # Number of offspring proportional to fitness
            num_offspring = int(np.round(species.fitness * 2))  # Up to 2 offspring per species
            
            for _ in range(num_offspring):
                # Asexual reproduction (most common)
                if np.random.random() < 0.8:  # 80% asexual reproduction
                    child = IusmorfoSpecies(
                        species_id=f"{species.species_id}_offspring_{self.generation}",
                        genotype=species.genotype.copy(),
                        phenotype=species.phenotype.copy(),
                        fitness=0.0,  # Will be recalculated
                        population_size=1,
                        genealogy=species.genealogy + [species.species_id]
                    )
                else:  # Sexual reproduction (crossover)
                    if len(survivors) > 1:
                        partner = np.random.choice([s for s in survivors if s.species_id != species.species_id])
                        child = species.crossover(partner)
                    else:
                        continue  # Skip if no partner available
                        
                offspring.append(child)
                
        return offspring
        
    def _mutation(self, offspring: List[IusmorfoSpecies]) -> List[IusmorfoSpecies]:
        """
        Apply mutations to offspring.
        
        Args:
            offspring: List of offspring to mutate
            
        Returns:
            List including mutants
        """
        mutants = []
        
        for child in offspring:
            if np.random.random() < self.mutation_rate:
                mutant = child.mutate()
                mutants.append(mutant)
                
                # Record speciation event
                self.speciation_log.append({
                    'parent_species': child.species_id,
                    'new_species': mutant.species_id,
                    'speciation_time': self.generation,
                    'mutation_type': 'point_mutation'
                })
                
        return offspring + mutants
        
    def _population_regulation(self):
        """
        Regulate population size to stay within carrying capacity.
        """
        total_population = sum(s.population_size for s in self.species_population.values())
        
        if total_population > self.carrying_capacity:
            # Randomly remove individuals proportional to inverse fitness
            species_list = list(self.species_population.values())
            weights = [1.0 / (s.fitness + 0.01) for s in species_list]  # Inverse fitness weighting
            
            individuals_to_remove = total_population - self.carrying_capacity
            
            for _ in range(individuals_to_remove):
                if species_list:
                    chosen_species = np.random.choice(species_list, p=np.array(weights)/sum(weights))
                    chosen_species.population_size -= 1
                    
                    if chosen_species.population_size <= 0:
                        # Remove extinct species
                        del self.species_population[chosen_species.species_id]
                        species_list.remove(chosen_species)
                        weights = [1.0 / (s.fitness + 0.01) for s in species_list]
                        
    def _update_landscape(self):
        """
        Update fitness landscape due to temporal drift.
        """
        drift_strength = self.fitness_landscape['temporal_drift']
        
        # Randomly shift peak and valley locations
        for peak in self.fitness_landscape['peaks']:
            drift = np.random.normal(0, drift_strength, self.dimensions)
            peak['location'] = np.clip(peak['location'] + drift, -1, 1)
            
        for valley in self.fitness_landscape['valleys']:
            drift = np.random.normal(0, drift_strength, self.dimensions)
            valley['location'] = np.clip(valley['location'] + drift, -1, 1)
            
    def _update_citation_network(self):
        """
        Update citation network with power-law distribution (γ=2.3).
        """
        # Add citations between related species
        species_ids = list(self.species_population.keys())
        
        if len(species_ids) < 2:
            return
            
        # Generate power-law distributed citations
        for species_id in species_ids:
            # Number of citations follows power-law
            num_citations = max(1, int(np.random.pareto(self.gamma - 1) + 1))
            
            # Choose targets weighted by fitness and genealogical distance
            weights = []
            targets = []
            
            for other_id in species_ids:
                if other_id != species_id:
                    other_species = self.species_population[other_id]
                    
                    # Weight by fitness
                    weight = other_species.fitness
                    
                    # Weight by genealogical relatedness
                    genealogical_distance = len(set(
                        self.species_population[species_id].genealogy
                    ).symmetric_difference(set(other_species.genealogy)))
                    
                    weight *= np.exp(-genealogical_distance * 0.1)
                    
                    weights.append(weight)
                    targets.append(other_id)
                    
            if weights and targets:
                # Normalize weights
                weights = np.array(weights)
                if weights.sum() > 0:
                    weights = weights / weights.sum()
                    
                    # Add citations
                    for _ in range(min(num_citations, len(targets))):
                        target = np.random.choice(targets, p=weights)
                        self.citation_network.add_edge(species_id, target, 
                                                     generation=self.generation,
                                                     weight=1.0)
                        
    def _calculate_generation_stats(self) -> Dict[str, Any]:
        """
        Calculate statistics for current generation.
        
        Returns:
            Dictionary with generation statistics
        """
        if not self.species_population:
            return {
                'generation': self.generation,
                'num_species': 0,
                'total_population': 0,
                'mean_fitness': 0.0,
                'fitness_variance': 0.0,
                'diversity': 0.0
            }
            
        species_list = list(self.species_population.values())
        fitnesses = [s.fitness for s in species_list]
        populations = [s.population_size for s in species_list]
        
        # Calculate genetic diversity (mean pairwise distance)
        diversity = 0.0
        if len(species_list) > 1:
            distances = []
            for i, species1 in enumerate(species_list):
                for j, species2 in enumerate(species_list[i+1:], i+1):
                    genotype1 = np.array(list(species1.genotype.values()))
                    genotype2 = np.array(list(species2.genotype.values()))
                    distance = np.linalg.norm(genotype1 - genotype2)
                    distances.append(distance)
            diversity = np.mean(distances) if distances else 0.0
            
        return {
            'generation': self.generation,
            'num_species': len(species_list),
            'total_population': sum(populations),
            'mean_fitness': np.mean(fitnesses),
            'fitness_variance': np.var(fitnesses),
            'diversity': diversity,
            'max_fitness': max(fitnesses),
            'min_fitness': min(fitnesses)
        }
        
    def get_citation_network_stats(self) -> Dict[str, Any]:
        """
        Analyze citation network statistics.
        
        Returns:
            Dictionary with network statistics
        """
        if self.citation_network.number_of_nodes() == 0:
            return {'error': 'Empty citation network'}
            
        # Basic network statistics
        stats = {
            'num_nodes': self.citation_network.number_of_nodes(),
            'num_edges': self.citation_network.number_of_edges(),
            'density': nx.density(self.citation_network)
        }
        
        # Degree distribution
        in_degrees = [d for n, d in self.citation_network.in_degree()]
        out_degrees = [d for n, d in self.citation_network.out_degree()]
        
        stats['mean_in_degree'] = np.mean(in_degrees) if in_degrees else 0
        stats['mean_out_degree'] = np.mean(out_degrees) if out_degrees else 0
        
        # Check if degree distribution follows power law
        if in_degrees:
            # Fit power law to in-degree distribution
            in_degrees_nonzero = [d for d in in_degrees if d > 0]
            if len(in_degrees_nonzero) > 10:  # Need sufficient data
                try:
                    # Simple power-law fitting
                    log_degrees = np.log(in_degrees_nonzero)
                    log_counts, bin_edges = np.histogram(log_degrees, bins=10)
                    log_counts_nonzero = log_counts[log_counts > 0]
                    
                    if len(log_counts_nonzero) > 2:
                        x = np.arange(len(log_counts_nonzero))
                        slope, intercept, r_value, p_value, std_err = stats.linregress(x, np.log(log_counts_nonzero))
                        stats['power_law_exponent'] = -slope
                        stats['power_law_r_squared'] = r_value**2
                    else:
                        stats['power_law_exponent'] = None
                        stats['power_law_r_squared'] = None
                except:
                    stats['power_law_exponent'] = None
                    stats['power_law_r_squared'] = None
            else:
                stats['power_law_exponent'] = None
                stats['power_law_r_squared'] = None
                
        return stats
        
    def export_population_state(self) -> Dict[str, Any]:
        """
        Export current population state for analysis.
        
        Returns:
            Complete population state dictionary
        """
        return {
            'generation': self.generation,
            'species_population': {
                species_id: {
                    'genotype': species.genotype,
                    'phenotype': species.phenotype,
                    'fitness': species.fitness,
                    'population_size': species.population_size,
                    'genealogy': species.genealogy,
                    'birth_time': species.birth_time,
                    'extinction_time': species.extinction_time
                }
                for species_id, species in self.species_population.items()
            },
            'extinction_log': self.extinction_log,
            'speciation_log': self.speciation_log,
            'diversity_history': self.diversity_history,
            'fitness_history': self.fitness_history,
            'population_history': self.population_history,
            'citation_network_stats': self.get_citation_network_stats()
        }
        
    def simulate_evolution(self, 
                          num_generations: int = 100,
                          initial_species: List[IusmorfoSpecies] = None) -> Dict[str, Any]:
        """
        Run complete evolutionary simulation.
        
        Args:
            num_generations: Number of generations to simulate
            initial_species: Initial species to seed population
            
        Returns:
            Complete simulation results
        """
        logger.info(f"Starting evolution simulation for {num_generations} generations")
        
        # Add initial species if provided
        if initial_species:
            for species in initial_species:
                self.add_species(species)
        
        # If no species provided, create random initial population
        if not self.species_population:
            logger.info("Creating random initial population")
            for i in range(10):  # Start with 10 random species
                genotype = {f'dim_{j}': np.random.uniform(-1, 1) for j in range(self.dimensions)}
                phenotype = {f'trait_{j}': np.random.uniform(0, 1) for j in range(self.dimensions)}
                
                species = IusmorfoSpecies(
                    species_id=f"initial_species_{i}",
                    genotype=genotype,
                    phenotype=phenotype,
                    population_size=np.random.randint(1, 5)
                )
                self.add_species(species)
        
        # Run evolution
        generation_stats = []
        
        for gen in range(num_generations):
            try:
                stats = self.evolve_generation()
                generation_stats.append(stats)
                
                # Log progress every 10 generations
                if gen % 10 == 0:
                    logger.info(f"Generation {gen}: {stats['num_species']} species, "
                              f"mean fitness: {stats['mean_fitness']:.3f}, "
                              f"diversity: {stats['diversity']:.3f}")
                              
                # Stop if population goes extinct
                if stats['num_species'] == 0:
                    logger.warning(f"Population extinct at generation {gen}")
                    break
                    
            except Exception as e:
                logger.error(f"Error in generation {gen}: {str(e)}")
                break
                
        # Final results
        final_results = {
            'simulation_params': {
                'num_generations': num_generations,
                'dimensions': self.dimensions,
                'carrying_capacity': self.carrying_capacity,
                'mutation_rate': self.mutation_rate,
                'selection_strength': self.selection_strength
            },
            'generation_stats': generation_stats,
            'final_population': self.export_population_state(),
            'summary': {
                'total_extinctions': len(self.extinction_log),
                'total_speciations': len(self.speciation_log),
                'final_diversity': self.diversity_history[-1] if self.diversity_history else 0,
                'final_mean_fitness': self.fitness_history[-1] if self.fitness_history else 0
            }
        }
        
        logger.info(f"Evolution simulation completed. Final diversity: {final_results['summary']['final_diversity']:.3f}")
        
        return final_results

def create_constitutional_species(country: str, 
                                 constitution_params: Dict[str, float],
                                 implementation_gap: float = 0.0) -> IusmorfoSpecies:
    """
    Create an IusmorfoSpecies representing a country's constitutional system.
    
    Args:
        country: Country name/identifier
        constitution_params: 9-dimensional constitutional parameters
        implementation_gap: SAPNC reality filter coefficient
        
    Returns:
        IusmorfoSpecies representing the constitutional system
    """
    # Ensure we have 9 dimensions
    standard_dimensions = [
        'federal_structure', 'judicial_independence', 'democratic_participation',
        'individual_rights', 'separation_powers', 'constitutional_stability',
        'rule_of_law', 'social_rights', 'checks_balances'
    ]
    
    genotype = {}
    for i, dim in enumerate(standard_dimensions):
        if dim in constitution_params:
            genotype[dim] = constitution_params[dim]
        else:
            genotype[dim] = 0.0  # Default neutral value
            
    # Phenotype reflects implementation reality (genotype modified by SAPNC filter)
    phenotype = {}
    for dim, value in genotype.items():
        # Apply implementation gap
        actual_value = value * (1.0 - implementation_gap)
        phenotype[f"{dim}_actual"] = actual_value
        
    return IusmorfoSpecies(
        species_id=f"constitutional_system_{country}",
        genotype=genotype,
        phenotype=phenotype,
        fitness=0.0,  # Will be calculated by arena
        population_size=1
    )

# Example usage and testing
if __name__ == "__main__":
    # Create competitive arena
    arena = CompetitiveArena(dimensions=9, carrying_capacity=50)
    
    # Create some example constitutional species
    colombia_params = {
        'federal_structure': 0.3,
        'judicial_independence': 0.7,
        'democratic_participation': 0.6,
        'individual_rights': 0.8,
        'separation_powers': 0.5,
        'constitutional_stability': -0.2,
        'rule_of_law': 0.4,
        'social_rights': 0.9,
        'checks_balances': 0.6
    }
    
    usa_params = {
        'federal_structure': 0.8,
        'judicial_independence': 0.9,
        'democratic_participation': 0.7,
        'individual_rights': 0.9,
        'separation_powers': 0.8,
        'constitutional_stability': 0.6,
        'rule_of_law': 0.8,
        'social_rights': 0.3,
        'checks_balances': 0.9
    }
    
    # Create species
    colombia_species = create_constitutional_species("Colombia", colombia_params, implementation_gap=0.4)
    usa_species = create_constitutional_species("USA", usa_params, implementation_gap=0.1)
    
    # Run simulation
    initial_species = [colombia_species, usa_species]
    results = arena.simulate_evolution(num_generations=50, initial_species=initial_species)
    
    print("\n=== Competitive Arena Simulation Results ===")
    print(f"Final number of species: {results['summary']['final_diversity']}")
    print(f"Total extinctions: {results['summary']['total_extinctions']}")
    print(f"Total speciations: {results['summary']['total_speciations']}")
    print(f"Final mean fitness: {results['summary']['final_mean_fitness']:.3f}")