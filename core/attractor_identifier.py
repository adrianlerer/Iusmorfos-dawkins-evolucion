#!/usr/bin/env python3
"""
AttractorIdentifier Module - Iusmorfos Framework v4.0
=====================================================

Implementa la identificación de cuencas de atracción (attractor basins) en el 
espacio político 9-dimensional para predecir la convergencia de sistemas
institucionales hacia estados estables.

Author: Adrian Lerer & Claude (AI Assistant)
Version: 4.0
Date: September 2024

Key Concepts:
- Attractor basin identification in 9D political space
- Dynamical systems analysis of institutional evolution
- Stability analysis of constitutional equilibria
- Phase space reconstruction and trajectory analysis
- Lyapunov exponents for stability characterization
- Bifurcation analysis for transition detection
- Strange attractors for chaotic institutional dynamics
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import scipy.optimize as optimize
from scipy.integrate import odeint
from scipy.linalg import eigvals
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttractorBasin:
    """
    Represents an attractor basin in 9D political space.
    """
    
    def __init__(self, 
                 basin_id: str,
                 attractor_point: np.ndarray,
                 basin_boundaries: Dict[str, Tuple[float, float]],
                 stability_type: str = "stable_node",
                 lyapunov_exponents: List[float] = None):
        """
        Initialize attractor basin.
        
        Args:
            basin_id: Unique identifier for the basin
            attractor_point: Central attractor point in 9D space
            basin_boundaries: Boundaries of basin for each dimension
            stability_type: Type of stability (stable_node, spiral, saddle, etc.)
            lyapunov_exponents: List of Lyapunov exponents for stability analysis
        """
        self.basin_id = basin_id
        self.attractor_point = attractor_point
        self.basin_boundaries = basin_boundaries
        self.stability_type = stability_type
        self.lyapunov_exponents = lyapunov_exponents or []
        
        # Derived properties
        self.basin_volume = self._calculate_basin_volume()
        self.convergence_rate = self._estimate_convergence_rate()
        self.trajectories = []  # Stored trajectory data
        self.institutional_examples = []  # Real-world examples in this basin
        
    def _calculate_basin_volume(self) -> float:
        """Calculate approximate volume of the basin."""
        volume = 1.0
        for dim, (min_val, max_val) in self.basin_boundaries.items():
            volume *= (max_val - min_val)
        return volume
        
    def _estimate_convergence_rate(self) -> float:
        """Estimate convergence rate from Lyapunov exponents."""
        if self.lyapunov_exponents:
            # Most negative exponent determines convergence rate
            return -min(self.lyapunov_exponents)
        return 0.1  # Default estimate
        
    def contains_point(self, point: np.ndarray) -> bool:
        """
        Check if a point is within this basin.
        
        Args:
            point: 9D point to check
            
        Returns:
            True if point is in basin
        """
        for i, (dim, (min_val, max_val)) in enumerate(self.basin_boundaries.items()):
            if i < len(point):
                if point[i] < min_val or point[i] > max_val:
                    return False
        return True
        
    def distance_to_attractor(self, point: np.ndarray) -> float:
        """
        Calculate distance from point to attractor.
        
        Args:
            point: 9D point
            
        Returns:
            Euclidean distance to attractor
        """
        return np.linalg.norm(point - self.attractor_point)
        
    def predict_trajectory(self, 
                          initial_point: np.ndarray, 
                          dynamics_function: Callable,
                          time_steps: int = 100) -> np.ndarray:
        """
        Predict trajectory from initial point to attractor.
        
        Args:
            initial_point: Starting point in 9D space
            dynamics_function: Function defining system dynamics
            time_steps: Number of time steps to simulate
            
        Returns:
            Array of trajectory points
        """
        t = np.linspace(0, 10, time_steps)  # 10 time units
        trajectory = odeint(dynamics_function, initial_point, t)
        
        # Store trajectory for analysis
        self.trajectories.append({
            'initial_point': initial_point,
            'trajectory': trajectory,
            'time': t,
            'final_distance': self.distance_to_attractor(trajectory[-1])
        })
        
        return trajectory

class AttractorIdentifier:
    """
    Main class for identifying and analyzing attractor basins in 9D political space.
    """
    
    def __init__(self, 
                 dimensions: int = 9,
                 resolution: int = 50,
                 convergence_threshold: float = 1e-6,
                 max_iterations: int = 1000):
        """
        Initialize attractor identifier.
        
        Args:
            dimensions: Number of dimensions in political space
            resolution: Grid resolution for basin identification
            convergence_threshold: Threshold for convergence detection
            max_iterations: Maximum iterations for trajectory computation
        """
        self.dimensions = dimensions
        self.resolution = resolution
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        
        # Identified attractors and basins
        self.attractors: Dict[str, AttractorBasin] = {}
        self.basin_map = None  # Grid showing basin membership
        
        # System dynamics parameters
        self.dynamics_parameters = {}
        self.institutional_forces = {}
        
        # Analysis results
        self.bifurcation_points = []
        self.stability_matrix = None
        
        logger.info(f"Initialized AttractorIdentifier for {dimensions}D space")
        
    def set_institutional_dynamics(self, 
                                 dynamics_params: Dict[str, Any],
                                 force_functions: Dict[str, Callable] = None):
        """
        Set parameters for institutional dynamics.
        
        Args:
            dynamics_params: Parameters governing system dynamics
            force_functions: Custom force functions for different institutional pressures
        """
        self.dynamics_parameters = dynamics_params
        self.institutional_forces = force_functions or {}
        
        # Default institutional forces
        if 'democratic_pressure' not in self.institutional_forces:
            self.institutional_forces['democratic_pressure'] = self._democratic_force
        if 'authoritarian_drift' not in self.institutional_forces:
            self.institutional_forces['authoritarian_drift'] = self._authoritarian_force
        if 'economic_pressure' not in self.institutional_forces:
            self.institutional_forces['economic_pressure'] = self._economic_force
        if 'cultural_inertia' not in self.institutional_forces:
            self.institutional_forces['cultural_inertia'] = self._cultural_force
            
        logger.info("Institutional dynamics configured")
        
    def _democratic_force(self, state: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Calculate democratic pressure force in 9D space.
        
        Args:
            state: Current system state (9D vector)
            params: Force parameters
            
        Returns:
            Force vector
        """
        # Democratic pressure pushes toward higher democratic participation,
        # judicial independence, and individual rights
        target_democratic_state = np.array([
            0.5,   # federal_structure (moderate)
            0.8,   # judicial_independence (high)
            0.9,   # democratic_participation (very high)
            0.9,   # individual_rights (very high)
            0.7,   # separation_powers (high)
            0.3,   # constitutional_stability (moderate, allows change)
            0.8,   # rule_of_law (high)
            0.6,   # social_rights (moderate-high)
            0.8    # checks_balances (high)
        ])
        
        strength = params.get('democratic_strength', 0.1)
        force = strength * (target_democratic_state - state)
        
        # Apply diminishing returns
        distance = np.linalg.norm(target_democratic_state - state)
        decay_factor = np.exp(-distance * 2)
        
        return force * decay_factor
        
    def _authoritarian_force(self, state: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Calculate authoritarian drift force.
        
        Args:
            state: Current system state
            params: Force parameters
            
        Returns:
            Force vector
        """
        # Authoritarian drift reduces democratic participation, judicial independence
        target_authoritarian_state = np.array([
            -0.3,  # federal_structure (centralized)
            -0.6,  # judicial_independence (low)
            -0.7,  # democratic_participation (low)
            -0.4,  # individual_rights (restricted)
            -0.8,  # separation_powers (weak)
            0.8,   # constitutional_stability (rigid)
            -0.5,  # rule_of_law (selective)
            0.2,   # social_rights (moderate, used for control)
            -0.9   # checks_balances (weak)
        ])
        
        strength = params.get('authoritarian_strength', 0.05)
        force = strength * (target_authoritarian_state - state)
        
        # Authoritarian drift is often nonlinear and accelerating
        distance = np.linalg.norm(state - target_authoritarian_state)
        acceleration_factor = 1 + distance**2
        
        return force * acceleration_factor
        
    def _economic_force(self, state: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Calculate economic pressure force.
        
        Args:
            state: Current system state
            params: Force parameters
            
        Returns:
            Force vector
        """
        # Economic pressures affect rule of law, individual rights (property), social rights
        economic_crisis = params.get('economic_crisis', 0.0)  # 0-1 scale
        inequality = params.get('inequality', 0.5)
        
        force = np.zeros(self.dimensions)
        
        # Crisis reduces rule of law, increases authoritarian tendencies
        if economic_crisis > 0.5:
            force[6] -= economic_crisis * 0.2  # rule_of_law
            force[3] -= economic_crisis * 0.15  # individual_rights
            force[8] -= economic_crisis * 0.1   # checks_balances
            
        # High inequality increases pressure for social rights
        if inequality > 0.6:
            force[7] += inequality * 0.3  # social_rights
            force[2] += inequality * 0.2  # democratic_participation (populist pressure)
            
        return force
        
    def _cultural_force(self, state: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Calculate cultural inertia force.
        
        Args:
            state: Current system state
            params: Force parameters
            
        Returns:
            Force vector
        """
        # Cultural inertia resists change, creates path dependence
        cultural_baseline = params.get('cultural_baseline', np.zeros(self.dimensions))
        inertia_strength = params.get('inertia_strength', 0.8)
        
        # Force toward cultural baseline with strong inertia
        force = inertia_strength * (cultural_baseline - state) * 0.05
        
        return force
        
    def institutional_dynamics(self, state: np.ndarray, t: float, **kwargs) -> np.ndarray:
        """
        Define institutional dynamics for ODE integration.
        
        Args:
            state: Current system state (9D vector)
            t: Time parameter
            **kwargs: Additional parameters
            
        Returns:
            Derivative vector (ds/dt)
        """
        dsdt = np.zeros(self.dimensions)
        
        # Apply all institutional forces
        for force_name, force_function in self.institutional_forces.items():
            force_params = self.dynamics_parameters.get(force_name, {})
            force = force_function(state, force_params)
            dsdt += force
            
        # Add noise for realistic dynamics
        noise_level = self.dynamics_parameters.get('noise_level', 0.01)
        noise = np.random.normal(0, noise_level, self.dimensions)
        dsdt += noise
        
        # Apply bounds to keep system in valid range [-1, 1]
        for i in range(self.dimensions):
            if state[i] > 0.9:
                dsdt[i] -= abs(dsdt[i]) * (state[i] - 0.9) * 10  # Strong restoring force
            elif state[i] < -0.9:
                dsdt[i] += abs(dsdt[i]) * (-0.9 - state[i]) * 10
                
        return dsdt
        
    def find_fixed_points(self, 
                         num_seeds: int = 100,
                         optimization_method: str = 'L-BFGS-B') -> List[np.ndarray]:
        """
        Find fixed points (attractors) by optimization.
        
        Args:
            num_seeds: Number of random starting points
            optimization_method: Optimization algorithm
            
        Returns:
            List of fixed points found
        """
        logger.info(f"Searching for fixed points with {num_seeds} random seeds")
        
        fixed_points = []
        
        def objective(x):
            """Objective function: minimize norm of derivative."""
            dx = self.institutional_dynamics(x, 0)
            return np.linalg.norm(dx)
            
        bounds = [(-1.0, 1.0)] * self.dimensions
        
        for i in range(num_seeds):
            # Random starting point
            x0 = np.random.uniform(-1, 1, self.dimensions)
            
            try:
                result = optimize.minimize(objective, x0, 
                                        method=optimization_method,
                                        bounds=bounds,
                                        options={'ftol': self.convergence_threshold})
                
                if result.success and result.fun < self.convergence_threshold:
                    fixed_point = result.x
                    
                    # Check if this is a new fixed point (not too close to existing ones)
                    is_new = True
                    for existing_fp in fixed_points:
                        if np.linalg.norm(fixed_point - existing_fp) < 0.1:
                            is_new = False
                            break
                            
                    if is_new:
                        fixed_points.append(fixed_point)
                        logger.debug(f"Found fixed point {len(fixed_points)}: {fixed_point}")
                        
            except Exception as e:
                logger.debug(f"Optimization failed for seed {i}: {str(e)}")
                continue
                
        logger.info(f"Found {len(fixed_points)} fixed points")
        return fixed_points
        
    def analyze_stability(self, fixed_point: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability of a fixed point using linearization.
        
        Args:
            fixed_point: Fixed point to analyze
            
        Returns:
            Dictionary with stability analysis results
        """
        # Compute Jacobian matrix numerically
        epsilon = 1e-8
        jacobian = np.zeros((self.dimensions, self.dimensions))
        
        f0 = self.institutional_dynamics(fixed_point, 0)
        
        for i in range(self.dimensions):
            x_plus = fixed_point.copy()
            x_plus[i] += epsilon
            f_plus = self.institutional_dynamics(x_plus, 0)
            
            jacobian[:, i] = (f_plus - f0) / epsilon
            
        # Eigenvalue analysis
        eigenvalues = eigvals(jacobian)
        real_parts = np.real(eigenvalues)
        imag_parts = np.imag(eigenvalues)
        
        # Classify stability
        max_real_part = np.max(real_parts)
        
        if max_real_part < -1e-6:
            stability_type = "stable_node"
        elif max_real_part > 1e-6:
            stability_type = "unstable_node" 
        else:
            stability_type = "marginal"
            
        # Check for oscillatory behavior
        if np.any(np.abs(imag_parts) > 1e-6):
            if max_real_part < -1e-6:
                stability_type = "stable_spiral"
            elif max_real_part > 1e-6:
                stability_type = "unstable_spiral"
            else:
                stability_type = "center"
                
        # Check for saddle point
        if np.any(real_parts > 1e-6) and np.any(real_parts < -1e-6):
            stability_type = "saddle"
            
        return {
            'stability_type': stability_type,
            'eigenvalues': eigenvalues,
            'real_parts': real_parts,
            'imag_parts': imag_parts,
            'jacobian': jacobian,
            'max_real_part': max_real_part
        }
        
    def compute_lyapunov_exponents(self, 
                                  initial_point: np.ndarray,
                                  integration_time: float = 100.0,
                                  dt: float = 0.01) -> List[float]:
        """
        Compute Lyapunov exponents for chaotic dynamics detection.
        
        Args:
            initial_point: Starting point for trajectory
            integration_time: Total integration time
            dt: Time step
            
        Returns:
            List of Lyapunov exponents
        """
        n = self.dimensions
        
        # Initialize trajectory and tangent vectors
        state = initial_point.copy()
        tangent_vectors = np.eye(n)
        
        # Storage for Lyapunov sums
        lyap_sums = np.zeros(n)
        
        steps = int(integration_time / dt)
        
        for step in range(steps):
            # Integrate main trajectory
            k1 = self.institutional_dynamics(state, step * dt)
            k2 = self.institutional_dynamics(state + 0.5 * dt * k1, (step + 0.5) * dt)
            k3 = self.institutional_dynamics(state + 0.5 * dt * k2, (step + 0.5) * dt)
            k4 = self.institutional_dynamics(state + dt * k3, (step + 1) * dt)
            
            state += dt * (k1 + 2*k2 + 2*k3 + k4) / 6
            
            # Integrate tangent vectors (linearized dynamics)
            for i in range(n):
                tv = tangent_vectors[:, i]
                
                # Compute Jacobian-vector product numerically
                epsilon = 1e-8
                f0 = self.institutional_dynamics(state, step * dt)
                
                jv = np.zeros(n)
                for j in range(n):
                    state_pert = state.copy()
                    state_pert[j] += epsilon
                    f_pert = self.institutional_dynamics(state_pert, step * dt)
                    jv += tv[j] * (f_pert - f0) / epsilon
                    
                tangent_vectors[:, i] += dt * jv
                
            # Gram-Schmidt orthogonalization every 10 steps
            if step % 10 == 0:
                # QR decomposition
                Q, R = np.linalg.qr(tangent_vectors)
                tangent_vectors = Q
                
                # Accumulate Lyapunov sums
                for i in range(n):
                    lyap_sums[i] += np.log(abs(R[i, i]))
                    
        # Calculate Lyapunov exponents
        lyapunov_exponents = lyap_sums / integration_time
        
        return lyapunov_exponents.tolist()
        
    def identify_basins(self, 
                       fixed_points: List[np.ndarray],
                       grid_resolution: int = None) -> Dict[str, AttractorBasin]:
        """
        Identify basins of attraction for each fixed point.
        
        Args:
            fixed_points: List of fixed points (attractors)
            grid_resolution: Resolution for basin identification grid
            
        Returns:
            Dictionary of identified basins
        """
        if grid_resolution is None:
            grid_resolution = self.resolution
            
        logger.info(f"Identifying basins for {len(fixed_points)} attractors")
        
        basins = {}
        
        # Create sampling grid in 9D space (computationally intensive!)
        # Use Monte Carlo sampling instead of full grid for efficiency
        num_samples = grid_resolution ** 2  # Reduce from grid_resolution^9
        
        sample_points = []
        basin_assignments = []
        
        for _ in range(num_samples):
            # Random sampling in 9D space
            sample_point = np.random.uniform(-1, 1, self.dimensions)
            sample_points.append(sample_point)
            
            # Find which attractor this point converges to
            closest_attractor_idx = self._find_basin_membership(sample_point, fixed_points)
            basin_assignments.append(closest_attractor_idx)
            
        # Create basins based on clustering results
        for i, fixed_point in enumerate(fixed_points):
            # Find all points that converge to this attractor
            basin_points = [sample_points[j] for j, assignment in enumerate(basin_assignments) if assignment == i]
            
            if basin_points:
                basin_points = np.array(basin_points)
                
                # Calculate basin boundaries (approximate)
                boundaries = {}
                dimension_names = [
                    'federal_structure', 'judicial_independence', 'democratic_participation',
                    'individual_rights', 'separation_powers', 'constitutional_stability',
                    'rule_of_law', 'social_rights', 'checks_balances'
                ]
                
                for dim_idx, dim_name in enumerate(dimension_names):
                    if dim_idx < self.dimensions:
                        dim_values = basin_points[:, dim_idx]
                        boundaries[dim_name] = (np.min(dim_values), np.max(dim_values))
                        
                # Analyze stability
                stability_analysis = self.analyze_stability(fixed_point)
                
                # Compute Lyapunov exponents
                lyapunov_exps = self.compute_lyapunov_exponents(fixed_point)
                
                # Create basin
                basin = AttractorBasin(
                    basin_id=f"basin_{i}",
                    attractor_point=fixed_point,
                    basin_boundaries=boundaries,
                    stability_type=stability_analysis['stability_type'],
                    lyapunov_exponents=lyapunov_exps
                )
                
                basins[basin.basin_id] = basin
                
                logger.info(f"Created {basin.basin_id}: {basin.stability_type}, "
                          f"volume={basin.basin_volume:.3f}")
                
        self.attractors = basins
        return basins
        
    def _find_basin_membership(self, 
                              initial_point: np.ndarray, 
                              fixed_points: List[np.ndarray]) -> int:
        """
        Find which basin a point belongs to by trajectory integration.
        
        Args:
            initial_point: Starting point
            fixed_points: List of known attractors
            
        Returns:
            Index of closest attractor
        """
        # Integrate trajectory for a reasonable time
        t = np.linspace(0, 10, 100)
        
        try:
            trajectory = odeint(self.institutional_dynamics, initial_point, t)
            final_point = trajectory[-1]
            
            # Find closest fixed point
            distances = [np.linalg.norm(final_point - fp) for fp in fixed_points]
            return np.argmin(distances)
            
        except:
            # If integration fails, use simple closest distance
            distances = [np.linalg.norm(initial_point - fp) for fp in fixed_points]
            return np.argmin(distances)
            
    def predict_institutional_trajectory(self, 
                                       initial_state: Dict[str, float],
                                       time_horizon: float = 20.0,
                                       scenario_params: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Predict trajectory of institutional evolution.
        
        Args:
            initial_state: Initial institutional state (9D)
            time_horizon: Time horizon for prediction
            scenario_params: Scenario-specific parameters
            
        Returns:
            Prediction results with trajectory and convergence analysis
        """
        # Convert initial state to vector
        dimension_names = [
            'federal_structure', 'judicial_independence', 'democratic_participation',
            'individual_rights', 'separation_powers', 'constitutional_stability',
            'rule_of_law', 'social_rights', 'checks_balances'
        ]
        
        initial_vector = np.array([initial_state.get(dim, 0.0) for dim in dimension_names])
        
        # Update dynamics parameters with scenario
        if scenario_params:
            for key, value in scenario_params.items():
                if key in self.dynamics_parameters:
                    self.dynamics_parameters[key].update(value)
                else:
                    self.dynamics_parameters[key] = value
                    
        # Integrate trajectory
        t = np.linspace(0, time_horizon, int(time_horizon * 10))  # 10 points per time unit
        
        try:
            trajectory = odeint(self.institutional_dynamics, initial_vector, t)
            
            # Find convergent basin
            final_state = trajectory[-1]
            convergent_basin = None
            
            for basin_id, basin in self.attractors.items():
                if basin.contains_point(final_state):
                    convergent_basin = basin_id
                    break
                    
            # Calculate trajectory statistics
            trajectory_stats = {
                'initial_state': initial_vector,
                'final_state': final_state,
                'trajectory': trajectory,
                'time': t,
                'convergent_basin': convergent_basin,
                'total_change': np.linalg.norm(final_state - initial_vector),
                'convergence_time': self._estimate_convergence_time(trajectory),
                'stability_reached': self._check_stability_reached(trajectory[-50:]) if len(trajectory) > 50 else False
            }
            
            # Dimensional analysis
            dimensional_changes = {}
            for i, dim_name in enumerate(dimension_names):
                if i < len(initial_vector):
                    dimensional_changes[dim_name] = {
                        'initial': initial_vector[i],
                        'final': final_state[i],
                        'change': final_state[i] - initial_vector[i],
                        'trajectory': trajectory[:, i]
                    }
                    
            return {
                'success': True,
                'trajectory_stats': trajectory_stats,
                'dimensional_analysis': dimensional_changes,
                'scenario_params': scenario_params
            }
            
        except Exception as e:
            logger.error(f"Trajectory prediction failed: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'initial_state': initial_vector
            }
            
    def _estimate_convergence_time(self, trajectory: np.ndarray) -> float:
        """
        Estimate when trajectory converges to attractor.
        
        Args:
            trajectory: Trajectory array
            
        Returns:
            Estimated convergence time
        """
        if len(trajectory) < 10:
            return float('inf')
            
        # Look for when rate of change drops below threshold
        convergence_threshold = 0.01
        
        for i in range(10, len(trajectory)):
            recent_change = np.linalg.norm(trajectory[i] - trajectory[i-10])
            if recent_change < convergence_threshold:
                return float(i) / 10.0  # Convert to time units
                
        return float('inf')  # Did not converge
        
    def _check_stability_reached(self, trajectory_segment: np.ndarray) -> bool:
        """
        Check if trajectory has reached stable state.
        
        Args:
            trajectory_segment: Recent trajectory segment
            
        Returns:
            True if stable
        """
        if len(trajectory_segment) < 5:
            return False
            
        # Check variance in recent trajectory
        variances = np.var(trajectory_segment, axis=0)
        max_variance = np.max(variances)
        
        return max_variance < 0.001  # Very low variance = stable
        
    def detect_bifurcations(self, 
                           parameter_name: str,
                           parameter_range: Tuple[float, float],
                           num_steps: int = 50) -> List[Dict[str, Any]]:
        """
        Detect bifurcation points by parameter variation.
        
        Args:
            parameter_name: Parameter to vary
            parameter_range: (min, max) range for parameter
            num_steps: Number of parameter values to test
            
        Returns:
            List of detected bifurcation points
        """
        logger.info(f"Detecting bifurcations for parameter {parameter_name}")
        
        bifurcations = []
        param_values = np.linspace(parameter_range[0], parameter_range[1], num_steps)
        
        previous_attractors = None
        
        for param_val in param_values:
            # Update parameter
            if parameter_name in self.dynamics_parameters:
                original_value = self.dynamics_parameters[parameter_name]
                self.dynamics_parameters[parameter_name] = param_val
            else:
                self.dynamics_parameters[parameter_name] = param_val
                original_value = None
                
            try:
                # Find attractors for this parameter value
                fixed_points = self.find_fixed_points(num_seeds=20)  # Reduced for efficiency
                
                current_attractors = len(fixed_points)
                
                # Check for bifurcation
                if previous_attractors is not None and current_attractors != previous_attractors:
                    bifurcation = {
                        'parameter_value': param_val,
                        'parameter_name': parameter_name,
                        'attractors_before': previous_attractors,
                        'attractors_after': current_attractors,
                        'bifurcation_type': self._classify_bifurcation(previous_attractors, current_attractors)
                    }
                    bifurcations.append(bifurcation)
                    
                    logger.info(f"Bifurcation detected at {parameter_name}={param_val}: "
                              f"{previous_attractors} -> {current_attractors} attractors")
                              
                previous_attractors = current_attractors
                
            except Exception as e:
                logger.warning(f"Failed to analyze parameter value {param_val}: {str(e)}")
                continue
            finally:
                # Restore original parameter value
                if original_value is not None:
                    self.dynamics_parameters[parameter_name] = original_value
                elif parameter_name in self.dynamics_parameters:
                    del self.dynamics_parameters[parameter_name]
                    
        self.bifurcation_points.extend(bifurcations)
        return bifurcations
        
    def _classify_bifurcation(self, num_before: int, num_after: int) -> str:
        """
        Classify type of bifurcation.
        
        Args:
            num_before, num_after: Number of attractors before and after
            
        Returns:
            Bifurcation type string
        """
        if num_before < num_after:
            if num_after == num_before + 1:
                return "saddle_node"
            else:
                return "pitchfork"
        elif num_before > num_after:
            return "transcritical" 
        else:
            return "unknown"
            
    def export_analysis_results(self) -> Dict[str, Any]:
        """
        Export complete attractor analysis results.
        
        Returns:
            Complete analysis results dictionary
        """
        return {
            'attractors': {
                basin_id: {
                    'attractor_point': basin.attractor_point.tolist(),
                    'basin_boundaries': basin.basin_boundaries,
                    'stability_type': basin.stability_type,
                    'lyapunov_exponents': basin.lyapunov_exponents,
                    'basin_volume': basin.basin_volume,
                    'convergence_rate': basin.convergence_rate,
                    'num_trajectories': len(basin.trajectories)
                }
                for basin_id, basin in self.attractors.items()
            },
            'bifurcation_points': self.bifurcation_points,
            'dynamics_parameters': self.dynamics_parameters,
            'analysis_metadata': {
                'dimensions': self.dimensions,
                'resolution': self.resolution,
                'convergence_threshold': self.convergence_threshold,
                'num_attractors': len(self.attractors)
            }
        }

def create_scenario_analysis(country: str,
                           current_state: Dict[str, float],
                           scenarios: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Create comprehensive scenario analysis for institutional evolution.
    
    Args:
        country: Country identifier
        current_state: Current institutional state
        scenarios: Dictionary of scenarios with parameters
        
    Returns:
        Scenario analysis results
    """
    identifier = AttractorIdentifier()
    
    # Set up realistic institutional dynamics
    dynamics_params = {
        'democratic_pressure': {'democratic_strength': 0.1},
        'authoritarian_drift': {'authoritarian_strength': 0.05},
        'economic_pressure': {'economic_crisis': 0.0, 'inequality': 0.5},
        'cultural_inertia': {
            'cultural_baseline': np.array(list(current_state.values())[:9]),
            'inertia_strength': 0.8
        },
        'noise_level': 0.01
    }
    
    identifier.set_institutional_dynamics(dynamics_params)
    
    # Find current attractors
    fixed_points = identifier.find_fixed_points(num_seeds=50)
    basins = identifier.identify_basins(fixed_points)
    
    # Analyze each scenario
    scenario_results = {}
    
    for scenario_name, scenario_params in scenarios.items():
        logger.info(f"Analyzing scenario: {scenario_name}")
        
        prediction = identifier.predict_institutional_trajectory(
            initial_state=current_state,
            time_horizon=20.0,
            scenario_params=scenario_params
        )
        
        scenario_results[scenario_name] = {
            'prediction': prediction,
            'scenario_parameters': scenario_params
        }
        
    return {
        'country': country,
        'current_state': current_state,
        'attractors': identifier.export_analysis_results(),
        'scenarios': scenario_results,
        'analysis_summary': {
            'num_scenarios': len(scenarios),
            'num_attractors': len(basins),
            'current_basin': None  # Would need to determine which basin current state is in
        }
    }

# Example usage and testing
if __name__ == "__main__":
    # Create attractor identifier
    identifier = AttractorIdentifier(dimensions=9)
    
    # Set up institutional dynamics
    dynamics_params = {
        'democratic_pressure': {'democratic_strength': 0.08},
        'authoritarian_drift': {'authoritarian_strength': 0.03},
        'economic_pressure': {'economic_crisis': 0.2, 'inequality': 0.7},
        'cultural_inertia': {
            'cultural_baseline': np.array([0.2, 0.5, 0.6, 0.7, 0.4, 0.3, 0.5, 0.8, 0.6]),
            'inertia_strength': 0.9
        },
        'noise_level': 0.005
    }
    
    identifier.set_institutional_dynamics(dynamics_params)
    
    # Find attractors
    fixed_points = identifier.find_fixed_points(num_seeds=30)
    basins = identifier.identify_basins(fixed_points)
    
    # Example prediction
    colombia_state = {
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
    
    # Scenario: Economic crisis
    crisis_scenario = {
        'economic_pressure': {'economic_crisis': 0.8, 'inequality': 0.9},
        'authoritarian_drift': {'authoritarian_strength': 0.1}
    }
    
    prediction = identifier.predict_institutional_trajectory(
        initial_state=colombia_state,
        time_horizon=15.0,
        scenario_params=crisis_scenario
    )
    
    print("\n=== Attractor Analysis Results ===")
    print(f"Found {len(basins)} attractor basins")
    for basin_id, basin in basins.items():
        print(f"{basin_id}: {basin.stability_type}, volume={basin.basin_volume:.3f}")
        
    print(f"\nTrajectory prediction successful: {prediction['success']}")
    if prediction['success']:
        stats = prediction['trajectory_stats']
        print(f"Total institutional change: {stats['total_change']:.3f}")
        print(f"Convergence time: {stats['convergence_time']:.1f}")
        print(f"Convergent basin: {stats['convergent_basin']}")