#!/usr/bin/env python3
"""
Robustness Testing Suite for Iusmorfos
=====================================

Comprehensive robustness and sensitivity analysis for the legal evolution model.
Tests model stability across parameter variations, bootstrap validation, and
cross-validation procedures.

Following FAIR principles and reproducibility best practices.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import logging
from typing import Dict, List, Any, Tuple, Optional, Callable
from datetime import datetime
import warnings
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Statistical and ML libraries
from scipy import stats
from sklearn.model_selection import KFold, cross_val_score, Bootstrap
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.preprocessing import StandardScaler

# Configuration
from config import get_config

warnings.filterwarnings('ignore')


class RobustnessAnalyzer:
    """
    Comprehensive robustness testing for Iusmorfos framework.
    
    Features:
    - Parameter sensitivity analysis
    - Bootstrap validation
    - Cross-validation with multiple models
    - Stability testing across random seeds
    - Model comparison and benchmarking
    """
    
    def __init__(self, n_bootstrap_samples: int = 1000, 
                 n_cv_folds: int = 5, 
                 n_sensitivity_points: int = 20):
        """Initialize robustness analyzer."""
        self.config = get_config()
        self.logger = logging.getLogger('iusmorfos.robustness')
        
        self.n_bootstrap_samples = n_bootstrap_samples
        self.n_cv_folds = n_cv_folds
        self.n_sensitivity_points = n_sensitivity_points
        
        self.results = {
            'parameter_sensitivity': {},
            'bootstrap_validation': {},
            'cross_validation': {},
            'stability_analysis': {},
            'model_comparison': {},
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'config_seed': self.config.config['reproducibility']['random_seed'],
                'n_bootstrap_samples': n_bootstrap_samples,
                'n_cv_folds': n_cv_folds
            }
        }
        
        self.logger.info("🔬 Robustness analyzer initialized")
    
    def run_parameter_sensitivity_analysis(self, 
                                         base_parameters: Dict[str, float],
                                         parameter_ranges: Dict[str, Tuple[float, float]],
                                         evaluation_function: Callable) -> Dict[str, Any]:
        """
        Analyze sensitivity to parameter variations.
        
        Args:
            base_parameters: Base parameter configuration
            parameter_ranges: Range of values to test for each parameter
            evaluation_function: Function to evaluate model with given parameters
        
        Returns:
            Dictionary containing sensitivity analysis results
        """
        self.logger.info("🎯 Running parameter sensitivity analysis")
        
        sensitivity_results = {}
        
        for param_name, (min_val, max_val) in parameter_ranges.items():
            if param_name not in base_parameters:
                self.logger.warning(f"Parameter {param_name} not found in base parameters")
                continue
            
            self.logger.info(f"  Analyzing sensitivity to {param_name}")
            
            # Create parameter sweep
            param_values = np.linspace(min_val, max_val, self.n_sensitivity_points)
            results = []
            
            for value in param_values:
                # Create modified parameters
                test_params = base_parameters.copy()
                test_params[param_name] = value
                
                try:
                    # Evaluate model with modified parameters
                    result = evaluation_function(test_params)
                    results.append({
                        'parameter_value': float(value),
                        'result': result,
                        'success': True
                    })
                except Exception as e:
                    self.logger.warning(f"Failed evaluation at {param_name}={value}: {e}")
                    results.append({
                        'parameter_value': float(value),
                        'result': None,
                        'success': False,
                        'error': str(e)
                    })
            
            # Analyze sensitivity
            successful_results = [r for r in results if r['success']]
            
            if len(successful_results) >= 3:
                param_vals = [r['parameter_value'] for r in successful_results]
                outcome_vals = [r['result'] for r in successful_results]
                
                # Calculate sensitivity metrics
                correlation, p_value = stats.pearsonr(param_vals, outcome_vals)
                
                # Calculate relative sensitivity (normalized by parameter range)
                param_range = max_val - min_val
                outcome_range = max(outcome_vals) - min(outcome_vals)
                relative_sensitivity = (outcome_range / np.mean(outcome_vals)) / (param_range / base_parameters[param_name])
                
                sensitivity_results[param_name] = {
                    'correlation': float(correlation),
                    'p_value': float(p_value),
                    'relative_sensitivity': float(relative_sensitivity),
                    'outcome_range': float(outcome_range),
                    'parameter_range': float(param_range),
                    'n_successful_evaluations': len(successful_results),
                    'sensitivity_level': self._categorize_sensitivity(abs(correlation)),
                    'results': results
                }
            else:
                sensitivity_results[param_name] = {
                    'error': 'Insufficient successful evaluations',
                    'n_successful_evaluations': len(successful_results),
                    'results': results
                }
        
        # Rank parameters by sensitivity
        valid_params = {k: v for k, v in sensitivity_results.items() if 'correlation' in v}
        if valid_params:
            sensitivity_ranking = sorted(
                valid_params.keys(), 
                key=lambda k: abs(valid_params[k]['correlation']), 
                reverse=True
            )
            
            sensitivity_results['summary'] = {
                'most_sensitive_parameter': sensitivity_ranking[0] if sensitivity_ranking else None,
                'sensitivity_ranking': sensitivity_ranking,
                'high_sensitivity_count': sum(1 for v in valid_params.values() 
                                            if v['sensitivity_level'] == 'high'),
                'total_parameters_tested': len(parameter_ranges)
            }
        
        self.results['parameter_sensitivity'] = sensitivity_results
        self.logger.info(f"✅ Parameter sensitivity analysis complete: {len(sensitivity_results)} parameters")
        
        return sensitivity_results
    
    def _categorize_sensitivity(self, correlation: float) -> str:
        """Categorize sensitivity level based on correlation strength."""
        if correlation >= 0.7:
            return 'high'
        elif correlation >= 0.3:
            return 'medium'
        else:
            return 'low'
    
    def run_bootstrap_validation(self, 
                                data: pd.DataFrame,
                                target_column: str,
                                feature_columns: List[str],
                                model_class=RandomForestRegressor) -> Dict[str, Any]:
        """
        Perform bootstrap validation of model performance.
        
        Args:
            data: Input dataset
            target_column: Name of target variable
            feature_columns: List of feature column names
            model_class: Model class to use for validation
        
        Returns:
            Bootstrap validation results
        """
        self.logger.info(f"🔄 Running bootstrap validation ({self.n_bootstrap_samples} samples)")
        
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Remove any rows with NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) == 0:
            return {'error': 'No valid data for bootstrap validation'}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Bootstrap sampling
        n_samples = len(X)
        bootstrap_scores = {
            'r2_scores': [],
            'rmse_scores': [],
            'mae_scores': [],
            'sample_sizes': [],
            'feature_importances': []
        }
        
        np.random.seed(self.config.config['reproducibility']['random_seed'])
        
        for i in range(self.n_bootstrap_samples):
            # Bootstrap sample
            sample_idx = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X_scaled[sample_idx]
            y_boot = y[sample_idx]
            
            # Out-of-bag samples for validation
            oob_idx = np.setdiff1d(np.arange(n_samples), np.unique(sample_idx))
            
            if len(oob_idx) == 0:
                continue
            
            X_oob = X_scaled[oob_idx]
            y_oob = y[oob_idx]
            
            # Train model
            try:
                if model_class == RandomForestRegressor:
                    model = model_class(n_estimators=50, random_state=42, n_jobs=1)
                else:
                    model = model_class()
                
                model.fit(X_boot, y_boot)
                
                # Predict on out-of-bag samples
                y_pred = model.predict(X_oob)
                
                # Calculate metrics
                r2 = r2_score(y_oob, y_pred)
                rmse = np.sqrt(mean_squared_error(y_oob, y_pred))
                mae = mean_absolute_error(y_oob, y_pred)
                
                bootstrap_scores['r2_scores'].append(r2)
                bootstrap_scores['rmse_scores'].append(rmse)
                bootstrap_scores['mae_scores'].append(mae)
                bootstrap_scores['sample_sizes'].append(len(sample_idx))
                
                # Feature importance (if available)
                if hasattr(model, 'feature_importances_'):
                    bootstrap_scores['feature_importances'].append(model.feature_importances_.tolist())
                
            except Exception as e:
                self.logger.warning(f"Bootstrap sample {i} failed: {e}")
                continue
        
        # Calculate statistics
        if len(bootstrap_scores['r2_scores']) == 0:
            return {'error': 'No successful bootstrap samples'}
        
        validation_results = {
            'n_successful_samples': len(bootstrap_scores['r2_scores']),
            'performance_statistics': {
                'r2_score': {
                    'mean': float(np.mean(bootstrap_scores['r2_scores'])),
                    'std': float(np.std(bootstrap_scores['r2_scores'])),
                    'ci_lower': float(np.percentile(bootstrap_scores['r2_scores'], 2.5)),
                    'ci_upper': float(np.percentile(bootstrap_scores['r2_scores'], 97.5)),
                    'median': float(np.median(bootstrap_scores['r2_scores']))
                },
                'rmse': {
                    'mean': float(np.mean(bootstrap_scores['rmse_scores'])),
                    'std': float(np.std(bootstrap_scores['rmse_scores'])),
                    'ci_lower': float(np.percentile(bootstrap_scores['rmse_scores'], 2.5)),
                    'ci_upper': float(np.percentile(bootstrap_scores['rmse_scores'], 97.5)),
                    'median': float(np.median(bootstrap_scores['rmse_scores']))
                },
                'mae': {
                    'mean': float(np.mean(bootstrap_scores['mae_scores'])),
                    'std': float(np.std(bootstrap_scores['mae_scores'])),
                    'ci_lower': float(np.percentile(bootstrap_scores['mae_scores'], 2.5)),
                    'ci_upper': float(np.percentile(bootstrap_scores['mae_scores'], 97.5)),
                    'median': float(np.median(bootstrap_scores['mae_scores']))
                }
            },
            'stability_assessment': {
                'r2_cv': float(np.std(bootstrap_scores['r2_scores']) / np.mean(bootstrap_scores['r2_scores'])),
                'rmse_cv': float(np.std(bootstrap_scores['rmse_scores']) / np.mean(bootstrap_scores['rmse_scores'])),
                'stable_performance': np.std(bootstrap_scores['r2_scores']) < 0.1
            }
        }
        
        # Feature importance analysis
        if bootstrap_scores['feature_importances']:
            importances_array = np.array(bootstrap_scores['feature_importances'])
            validation_results['feature_importance_statistics'] = {}
            
            for i, feature_name in enumerate(feature_columns):
                validation_results['feature_importance_statistics'][feature_name] = {
                    'mean': float(np.mean(importances_array[:, i])),
                    'std': float(np.std(importances_array[:, i])),
                    'ci_lower': float(np.percentile(importances_array[:, i], 2.5)),
                    'ci_upper': float(np.percentile(importances_array[:, i], 97.5))
                }
        
        self.results['bootstrap_validation'] = validation_results
        self.logger.info(f"✅ Bootstrap validation complete: {validation_results['n_successful_samples']} samples")
        
        return validation_results
    
    def run_cross_validation_analysis(self,
                                    data: pd.DataFrame,
                                    target_column: str,
                                    feature_columns: List[str],
                                    models: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Comprehensive cross-validation analysis with multiple models.
        
        Args:
            data: Input dataset
            target_column: Name of target variable
            feature_columns: List of feature column names
            models: Dictionary of models to compare
        
        Returns:
            Cross-validation results for all models
        """
        self.logger.info(f"🔀 Running cross-validation analysis ({self.n_cv_folds} folds)")
        
        if models is None:
            models = {
                'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'Linear Regression': LinearRegression(),
                'Ridge Regression': RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0])
            }
        
        # Prepare data
        X = data[feature_columns].values
        y = data[target_column].values
        
        # Remove any rows with NaN values
        valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
        X = X[valid_idx]
        y = y[valid_idx]
        
        if len(X) < self.n_cv_folds:
            return {'error': f'Insufficient data for {self.n_cv_folds}-fold CV'}
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        cv_results = {}
        
        # Cross-validation setup
        cv = KFold(n_splits=self.n_cv_folds, shuffle=True, 
                  random_state=self.config.config['reproducibility']['random_seed'])
        
        for model_name, model in models.items():
            self.logger.info(f"  Evaluating {model_name}")
            
            try:
                # Perform cross-validation
                scores = cross_val_score(model, X_scaled, y, cv=cv, 
                                       scoring='r2', n_jobs=1)
                
                # Additional metrics
                fold_results = {'r2_scores': [], 'rmse_scores': [], 'mae_scores': []}
                
                for train_idx, test_idx in cv.split(X_scaled):
                    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                    y_train, y_test = y[train_idx], y[test_idx]
                    
                    model_copy = type(model)(**model.get_params()) if hasattr(model, 'get_params') else model
                    model_copy.fit(X_train, y_train)
                    y_pred = model_copy.predict(X_test)
                    
                    fold_results['r2_scores'].append(r2_score(y_test, y_pred))
                    fold_results['rmse_scores'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                    fold_results['mae_scores'].append(mean_absolute_error(y_test, y_pred))
                
                cv_results[model_name] = {
                    'r2_score': {
                        'mean': float(np.mean(fold_results['r2_scores'])),
                        'std': float(np.std(fold_results['r2_scores'])),
                        'scores': [float(s) for s in fold_results['r2_scores']]
                    },
                    'rmse': {
                        'mean': float(np.mean(fold_results['rmse_scores'])),
                        'std': float(np.std(fold_results['rmse_scores'])),
                        'scores': [float(s) for s in fold_results['rmse_scores']]
                    },
                    'mae': {
                        'mean': float(np.mean(fold_results['mae_scores'])),
                        'std': float(np.std(fold_results['mae_scores'])),
                        'scores': [float(s) for s in fold_results['mae_scores']]
                    }
                }
                
            except Exception as e:
                self.logger.error(f"Cross-validation failed for {model_name}: {e}")
                cv_results[model_name] = {'error': str(e)}
        
        # Model ranking
        valid_models = {k: v for k, v in cv_results.items() if 'error' not in v}
        if valid_models:
            model_ranking = sorted(
                valid_models.keys(),
                key=lambda k: valid_models[k]['r2_score']['mean'],
                reverse=True
            )
            
            cv_results['summary'] = {
                'best_model': model_ranking[0] if model_ranking else None,
                'model_ranking': model_ranking,
                'n_models_tested': len(models),
                'n_successful_models': len(valid_models)
            }
        
        self.results['cross_validation'] = cv_results
        self.logger.info(f"✅ Cross-validation complete: {len(valid_models)} successful models")
        
        return cv_results
    
    def run_stability_analysis(self,
                             experiment_function: Callable,
                             n_runs: int = 10,
                             seed_range: Tuple[int, int] = (1, 100)) -> Dict[str, Any]:
        """
        Test model stability across different random seeds.
        
        Args:
            experiment_function: Function that runs the experiment
            n_runs: Number of different seeds to test
            seed_range: Range of seed values to sample from
        
        Returns:
            Stability analysis results
        """
        self.logger.info(f"🎲 Running stability analysis ({n_runs} different seeds)")
        
        # Generate random seeds
        base_seed = self.config.config['reproducibility']['random_seed']
        np.random.seed(base_seed)
        test_seeds = np.random.randint(seed_range[0], seed_range[1], n_runs)
        
        stability_results = {
            'experiment_results': [],
            'seeds_tested': test_seeds.tolist(),
            'n_runs': n_runs
        }
        
        for i, seed in enumerate(test_seeds):
            self.logger.info(f"  Run {i+1}/{n_runs} with seed {seed}")
            
            try:
                # Run experiment with specific seed
                result = experiment_function(seed)
                stability_results['experiment_results'].append({
                    'seed': int(seed),
                    'result': result,
                    'success': True
                })
                
            except Exception as e:
                self.logger.warning(f"Experiment failed with seed {seed}: {e}")
                stability_results['experiment_results'].append({
                    'seed': int(seed),
                    'result': None,
                    'success': False,
                    'error': str(e)
                })
        
        # Analyze stability
        successful_results = [r for r in stability_results['experiment_results'] if r['success']]
        
        if len(successful_results) >= 2:
            # Extract key metrics from results (assuming numeric results)
            try:
                result_values = [r['result'] for r in successful_results]
                
                # If results are dictionaries, extract specific metrics
                if isinstance(result_values[0], dict):
                    # Extract fitness scores or similar metrics
                    fitness_values = []
                    complexity_values = []
                    
                    for result in result_values:
                        if 'fitness_final' in result:
                            fitness_values.append(result['fitness_final'])
                        if 'complejidad_final' in result:
                            complexity_values.append(result['complejidad_final'])
                    
                    stability_metrics = {}
                    
                    if fitness_values:
                        stability_metrics['fitness_stability'] = {
                            'mean': float(np.mean(fitness_values)),
                            'std': float(np.std(fitness_values)),
                            'cv': float(np.std(fitness_values) / np.mean(fitness_values)),
                            'range': float(max(fitness_values) - min(fitness_values)),
                            'stable': np.std(fitness_values) / np.mean(fitness_values) < 0.1
                        }
                    
                    if complexity_values:
                        stability_metrics['complexity_stability'] = {
                            'mean': float(np.mean(complexity_values)),
                            'std': float(np.std(complexity_values)),
                            'cv': float(np.std(complexity_values) / np.mean(complexity_values)),
                            'range': float(max(complexity_values) - min(complexity_values)),
                            'stable': np.std(complexity_values) / np.mean(complexity_values) < 0.15
                        }
                    
                    stability_results['stability_metrics'] = stability_metrics
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze stability metrics: {e}")
        
        stability_results['summary'] = {
            'n_successful_runs': len(successful_results),
            'success_rate': len(successful_results) / n_runs,
            'reproducible': len(successful_results) / n_runs >= 0.8
        }
        
        self.results['stability_analysis'] = stability_results
        self.logger.info(f"✅ Stability analysis complete: {len(successful_results)}/{n_runs} successful")
        
        return stability_results
    
    def generate_robustness_report(self) -> Dict[str, Any]:
        """Generate comprehensive robustness analysis report."""
        self.logger.info("📋 Generating robustness report")
        
        report = {
            'executive_summary': {},
            'detailed_results': self.results.copy(),
            'recommendations': [],
            'overall_assessment': {}
        }
        
        # Executive summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'total_tests_conducted': len([k for k in self.results.keys() if k != 'metadata']),
            'tests_passed': 0,
            'critical_issues': [],
            'warnings': []
        }
        
        # Analyze each test category
        if 'parameter_sensitivity' in self.results and self.results['parameter_sensitivity']:
            ps_results = self.results['parameter_sensitivity']
            if 'summary' in ps_results:
                high_sensitivity = ps_results['summary'].get('high_sensitivity_count', 0)
                if high_sensitivity > 2:
                    summary['critical_issues'].append(
                        f"High sensitivity to {high_sensitivity} parameters - model may be unstable"
                    )
                elif high_sensitivity > 0:
                    summary['warnings'].append(
                        f"Moderate sensitivity to {high_sensitivity} parameters"
                    )
                else:
                    summary['tests_passed'] += 1
        
        if 'bootstrap_validation' in self.results and self.results['bootstrap_validation']:
            bv_results = self.results['bootstrap_validation']
            if 'stability_assessment' in bv_results:
                if bv_results['stability_assessment'].get('stable_performance', False):
                    summary['tests_passed'] += 1
                else:
                    summary['warnings'].append("Bootstrap validation shows performance instability")
        
        if 'stability_analysis' in self.results and self.results['stability_analysis']:
            sa_results = self.results['stability_analysis']
            if 'summary' in sa_results:
                if sa_results['summary'].get('reproducible', False):
                    summary['tests_passed'] += 1
                else:
                    summary['critical_issues'].append(
                        f"Low reproducibility: {sa_results['summary']['success_rate']:.1%} success rate"
                    )
        
        report['executive_summary'] = summary
        
        # Generate recommendations
        recommendations = []
        
        if summary['critical_issues']:
            recommendations.append("🚨 Address critical reproducibility issues before deployment")
        
        if 'parameter_sensitivity' in self.results:
            ps_results = self.results['parameter_sensitivity']
            if 'summary' in ps_results and ps_results['summary'].get('high_sensitivity_count', 0) > 0:
                most_sensitive = ps_results['summary'].get('most_sensitive_parameter')
                if most_sensitive:
                    recommendations.append(
                        f"💡 Focus parameter tuning on {most_sensitive} (highest sensitivity)"
                    )
        
        if len(summary['warnings']) > len(summary['critical_issues']):
            recommendations.append("⚠️ Consider additional validation with larger datasets")
        
        if summary['tests_passed'] >= 2:
            recommendations.append("✅ Model shows good robustness properties")
        
        recommendations.extend([
            "📊 Consider ensemble methods to improve stability",
            "🔍 Implement continuous monitoring in production",
            "📝 Document all hyperparameter choices and sensitivity ranges"
        ])
        
        report['recommendations'] = recommendations
        
        # Overall assessment
        total_possible_tests = 3  # Adjust based on implemented tests
        test_pass_rate = summary['tests_passed'] / total_possible_tests
        
        if test_pass_rate >= 0.8 and not summary['critical_issues']:
            assessment = "ROBUST - Model ready for deployment"
            confidence = "high"
        elif test_pass_rate >= 0.6 and len(summary['critical_issues']) <= 1:
            assessment = "MODERATE - Additional validation recommended" 
            confidence = "medium"
        else:
            assessment = "WEAK - Significant robustness concerns"
            confidence = "low"
        
        report['overall_assessment'] = {
            'assessment': assessment,
            'confidence_level': confidence,
            'test_pass_rate': test_pass_rate,
            'ready_for_production': test_pass_rate >= 0.8 and not summary['critical_issues']
        }
        
        # Save report
        report_path = self.config.get_path('results_dir') / f'robustness_report_{self.config.timestamp}.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"📄 Robustness report saved: {report_path}")
        self.logger.info(f"🎯 Overall assessment: {assessment}")
        
        return report
    
    def run_full_robustness_suite(self,
                                data: pd.DataFrame,
                                target_column: str,
                                feature_columns: List[str],
                                experiment_function: Optional[Callable] = None,
                                base_parameters: Optional[Dict[str, float]] = None,
                                parameter_ranges: Optional[Dict[str, Tuple[float, float]]] = None) -> Dict[str, Any]:
        """
        Run complete robustness testing suite.
        
        Args:
            data: Dataset for validation
            target_column: Target variable name
            feature_columns: Feature column names
            experiment_function: Function for stability testing
            base_parameters: Base parameters for sensitivity analysis
            parameter_ranges: Parameter ranges for sensitivity analysis
        
        Returns:
            Complete robustness analysis results
        """
        self.logger.info("🚀 Starting full robustness test suite")
        
        # 1. Bootstrap validation
        self.run_bootstrap_validation(data, target_column, feature_columns)
        
        # 2. Cross-validation analysis
        self.run_cross_validation_analysis(data, target_column, feature_columns)
        
        # 3. Parameter sensitivity (if parameters provided)
        if base_parameters and parameter_ranges:
            def dummy_evaluation(params):
                # Simple dummy evaluation - replace with actual model
                return sum(params.values()) / len(params)
            
            evaluation_fn = experiment_function if experiment_function else dummy_evaluation
            self.run_parameter_sensitivity_analysis(base_parameters, parameter_ranges, evaluation_fn)
        
        # 4. Stability analysis (if experiment function provided)
        if experiment_function:
            self.run_stability_analysis(experiment_function)
        
        # 5. Generate comprehensive report
        report = self.generate_robustness_report()
        
        self.logger.info("🏁 Full robustness test suite complete")
        
        return {
            'individual_results': self.results,
            'comprehensive_report': report
        }


def create_sample_robustness_test():
    """Create a sample robustness test for demonstration."""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    
    # Generate correlated features
    complexity = np.random.gamma(2, 2, n_samples) + 1  # 1-10 range
    adoption = np.random.beta(2, 2, n_samples)  # 0-1 range
    citations = np.random.pareto(1.3, n_samples) * 2 + 1  # Power law
    
    # Create fitness as function of features with some noise
    fitness = (0.3 * (complexity / 10) + 0.4 * adoption + 0.3 * np.log1p(citations) / 10) + np.random.normal(0, 0.1, n_samples)
    fitness = np.clip(fitness, 0, 1)  # Ensure 0-1 range
    
    # Create reform type (categorical)
    reform_types = np.random.choice(['constitutional', 'civil', 'criminal', 'administrative'], n_samples)
    
    data = pd.DataFrame({
        'complexity': complexity,
        'adoption': adoption, 
        'citations': citations,
        'fitness_score': fitness,
        'reform_type': reform_types,
        'year': np.random.randint(2000, 2024, n_samples)
    })
    
    return data


def main():
    """Main robustness testing function."""
    print("🧬 Iusmorfos Robustness Testing Suite")
    print("=" * 45)
    
    # Create analyzer
    analyzer = RobustnessAnalyzer(
        n_bootstrap_samples=100,  # Reduced for demo
        n_cv_folds=5,
        n_sensitivity_points=10
    )
    
    # Create sample data
    sample_data = create_sample_robustness_test()
    print(f"📊 Created sample dataset: {len(sample_data)} records")
    
    # Define test parameters
    feature_columns = ['complexity', 'adoption', 'citations']
    target_column = 'fitness_score'
    
    base_parameters = {
        'mutation_rate': 0.1,
        'selection_pressure': 0.8,
        'complexity_weight': 0.3
    }
    
    parameter_ranges = {
        'mutation_rate': (0.05, 0.3),
        'selection_pressure': (0.5, 1.0),
        'complexity_weight': (0.1, 0.5)
    }
    
    # Sample experiment function
    def sample_experiment(seed):
        np.random.seed(seed)
        # Simulate experiment result
        return {
            'fitness_final': np.random.normal(0.7, 0.1),
            'complejidad_final': np.random.normal(5.0, 1.0),
            'generaciones_completadas': 10
        }
    
    # Run full robustness suite
    results = analyzer.run_full_robustness_suite(
        data=sample_data,
        target_column=target_column,
        feature_columns=feature_columns,
        experiment_function=sample_experiment,
        base_parameters=base_parameters,
        parameter_ranges=parameter_ranges
    )
    
    # Print summary
    report = results['comprehensive_report']
    summary = report['executive_summary']
    
    print(f"\n📋 Robustness Analysis Summary:")
    print(f"Tests conducted: {summary['total_tests_conducted']}")
    print(f"Tests passed: {summary['tests_passed']}")
    print(f"Critical issues: {len(summary['critical_issues'])}")
    print(f"Warnings: {len(summary['warnings'])}")
    
    print(f"\n🎯 Overall Assessment:")
    print(f"Status: {report['overall_assessment']['assessment']}")
    print(f"Confidence: {report['overall_assessment']['confidence_level']}")
    print(f"Production ready: {report['overall_assessment']['ready_for_production']}")
    
    if summary['critical_issues']:
        print(f"\n🚨 Critical Issues:")
        for issue in summary['critical_issues']:
            print(f"  - {issue}")
    
    if report['recommendations']:
        print(f"\n💡 Top Recommendations:")
        for rec in report['recommendations'][:3]:
            print(f"  {rec}")


if __name__ == "__main__":
    main()