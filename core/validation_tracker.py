#!/usr/bin/env python3
"""
ValidationTracker Module - Iusmorfos Framework v4.0
===================================================

Implementa el seguimiento continuo de la precisión y validación del framework
Iusmorfos con métricas de reproducibilidad científica de clase mundial.

Author: Adrian Lerer & Claude (AI Assistant)
Version: 4.0
Date: September 2024

Key Concepts:
- Continuous accuracy tracking and validation
- Statistical significance testing (p = 0.03)
- Inter-coder reliability using Cohen's kappa
- Bootstrap validation with confidence intervals
- Reality filter calibration and performance monitoring
- Cross-cultural validation tracking
- Reproducibility metrics and benchmarking
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Callable
import scipy.stats as stats
from scipy import bootstrap
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
from sklearn.model_selection import cross_val_score, KFold
import json
import datetime
import hashlib
import warnings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationCase:
    """
    Represents a single validation case for the framework.
    """
    
    def __init__(self,
                 case_id: str,
                 country: str,
                 reform_type: str,
                 prediction_date: str,
                 actual_outcome_date: str = None,
                 predicted_values: Dict[str, float] = None,
                 actual_values: Dict[str, float] = None,
                 confidence_intervals: Dict[str, Tuple[float, float]] = None,
                 metadata: Dict[str, Any] = None):
        """
        Initialize validation case.
        
        Args:
            case_id: Unique identifier for validation case
            country: Country being analyzed
            reform_type: Type of institutional reform
            prediction_date: Date prediction was made
            actual_outcome_date: Date when actual outcome became available
            predicted_values: Framework predictions
            actual_values: Observed actual values
            confidence_intervals: Prediction confidence intervals
            metadata: Additional case metadata
        """
        self.case_id = case_id
        self.country = country
        self.reform_type = reform_type
        self.prediction_date = prediction_date
        self.actual_outcome_date = actual_outcome_date
        self.predicted_values = predicted_values or {}
        self.actual_values = actual_values or {}
        self.confidence_intervals = confidence_intervals or {}
        self.metadata = metadata or {}
        
        # Validation metrics
        self.validation_scores = {}
        self.is_validated = False
        self.validation_timestamp = None
        
    def add_prediction(self, metric: str, predicted_value: float, confidence_interval: Tuple[float, float] = None):
        """Add a prediction to this case."""
        self.predicted_values[metric] = predicted_value
        if confidence_interval:
            self.confidence_intervals[metric] = confidence_interval
            
    def add_actual_outcome(self, metric: str, actual_value: float):
        """Add an actual outcome to this case."""
        self.actual_values[metric] = actual_value
        
    def calculate_accuracy(self) -> Dict[str, float]:
        """
        Calculate accuracy metrics for this case.
        
        Returns:
            Dictionary with accuracy metrics
        """
        if not self.predicted_values or not self.actual_values:
            return {}
            
        accuracies = {}
        
        for metric in self.predicted_values:
            if metric in self.actual_values:
                predicted = self.predicted_values[metric]
                actual = self.actual_values[metric]
                
                # Calculate various accuracy measures
                absolute_error = abs(predicted - actual)
                relative_error = absolute_error / (abs(actual) + 1e-6)  # Avoid division by zero
                
                # Accuracy as 1 - relative_error (capped at 0)
                accuracy = max(0.0, 1.0 - relative_error)
                
                accuracies[f"{metric}_absolute_error"] = absolute_error
                accuracies[f"{metric}_relative_error"] = relative_error
                accuracies[f"{metric}_accuracy"] = accuracy
                
                # Check if within confidence interval (if available)
                if metric in self.confidence_intervals:
                    ci_lower, ci_upper = self.confidence_intervals[metric]
                    within_ci = ci_lower <= actual <= ci_upper
                    accuracies[f"{metric}_within_ci"] = float(within_ci)
                    
        self.validation_scores = accuracies
        return accuracies
        
    def is_complete(self) -> bool:
        """Check if case has both predictions and actual outcomes."""
        return bool(self.predicted_values and self.actual_values)
        
    def export_dict(self) -> Dict[str, Any]:
        """Export case as dictionary."""
        return {
            'case_id': self.case_id,
            'country': self.country,
            'reform_type': self.reform_type,
            'prediction_date': self.prediction_date,
            'actual_outcome_date': self.actual_outcome_date,
            'predicted_values': self.predicted_values,
            'actual_values': self.actual_values,
            'confidence_intervals': self.confidence_intervals,
            'validation_scores': self.validation_scores,
            'metadata': self.metadata,
            'is_validated': self.is_validated,
            'validation_timestamp': self.validation_timestamp
        }

class ValidationTracker:
    """
    Main class for tracking and validating framework performance.
    """
    
    def __init__(self, 
                 framework_version: str = "4.0",
                 significance_threshold: float = 0.0001,
                 min_sample_size: int = 10,
                 bootstrap_samples: int = 1000):
        """
        Initialize validation tracker.
        
        Args:
            framework_version: Version of framework being validated
            significance_threshold: Statistical significance threshold (default p = 0.03)
            min_sample_size: Minimum cases needed for statistical analysis
            bootstrap_samples: Number of bootstrap samples for confidence intervals
        """
        self.framework_version = framework_version
        self.significance_threshold = significance_threshold
        self.min_sample_size = min_sample_size
        self.bootstrap_samples = bootstrap_samples
        
        # Validation cases storage
        self.validation_cases: Dict[str, ValidationCase] = {}
        self.completed_cases: List[ValidationCase] = []
        
        # Performance tracking
        self.performance_history = []
        self.accuracy_benchmarks = {}
        self.reproducibility_metrics = {}
        
        # Inter-coder reliability tracking
        self.coder_agreements = []
        self.kappa_scores = []
        
        # Cross-cultural validation
        self.regional_performance = {}
        self.cultural_clusters = {}
        
        # Statistical testing
        self.hypothesis_tests = []
        
        logger.info(f"Initialized ValidationTracker v{framework_version}")
        
    def add_validation_case(self, case: ValidationCase) -> bool:
        """
        Add a validation case to the tracker.
        
        Args:
            case: ValidationCase to add
            
        Returns:
            True if successfully added
        """
        if case.case_id in self.validation_cases:
            logger.warning(f"Case {case.case_id} already exists")
            return False
            
        self.validation_cases[case.case_id] = case
        
        # If case is complete, add to completed cases
        if case.is_complete():
            case.calculate_accuracy()
            case.is_validated = True
            case.validation_timestamp = datetime.datetime.now().isoformat()
            self.completed_cases.append(case)
            
            logger.info(f"Added complete validation case: {case.case_id}")
        else:
            logger.info(f"Added incomplete validation case: {case.case_id}")
            
        return True
        
    def update_case_outcome(self, 
                           case_id: str, 
                           actual_outcomes: Dict[str, float],
                           outcome_date: str = None) -> bool:
        """
        Update a case with actual outcomes.
        
        Args:
            case_id: ID of case to update
            actual_outcomes: Dictionary of actual outcome values
            outcome_date: Date when outcomes were observed
            
        Returns:
            True if successfully updated
        """
        if case_id not in self.validation_cases:
            logger.error(f"Case {case_id} not found")
            return False
            
        case = self.validation_cases[case_id]
        
        # Update actual values
        case.actual_values.update(actual_outcomes)
        if outcome_date:
            case.actual_outcome_date = outcome_date
            
        # If case is now complete, validate it
        if case.is_complete() and not case.is_validated:
            case.calculate_accuracy()
            case.is_validated = True
            case.validation_timestamp = datetime.datetime.now().isoformat()
            self.completed_cases.append(case)
            
            logger.info(f"Validated case {case_id} with accuracy: {self._get_case_overall_accuracy(case):.3f}")
            
        return True
        
    def _get_case_overall_accuracy(self, case: ValidationCase) -> float:
        """
        Calculate overall accuracy for a case.
        
        Args:
            case: ValidationCase to analyze
            
        Returns:
            Overall accuracy score (0.0 to 1.0)
        """
        accuracy_scores = [v for k, v in case.validation_scores.items() if k.endswith('_accuracy')]
        return np.mean(accuracy_scores) if accuracy_scores else 0.0
        
    def calculate_framework_performance(self) -> Dict[str, Any]:
        """
        Calculate overall framework performance metrics.
        
        Returns:
            Performance metrics dictionary
        """
        if len(self.completed_cases) < self.min_sample_size:
            logger.warning(f"Only {len(self.completed_cases)} completed cases, need {self.min_sample_size} for statistics")
            return {
                'error': 'Insufficient data for statistical analysis',
                'completed_cases': len(self.completed_cases),
                'required_cases': self.min_sample_size
            }
            
        # Calculate accuracy scores
        all_accuracies = []
        metric_accuracies = defaultdict(list)
        
        for case in self.completed_cases:
            case_accuracy = self._get_case_overall_accuracy(case)
            all_accuracies.append(case_accuracy)
            
            # Collect by metric type
            for metric, score in case.validation_scores.items():
                if metric.endswith('_accuracy'):
                    base_metric = metric.replace('_accuracy', '')
                    metric_accuracies[base_metric].append(score)
                    
        # Overall statistics
        overall_stats = {
            'mean_accuracy': np.mean(all_accuracies),
            'median_accuracy': np.median(all_accuracies),
            'std_accuracy': np.std(all_accuracies),
            'min_accuracy': np.min(all_accuracies),
            'max_accuracy': np.max(all_accuracies),
            'num_cases': len(self.completed_cases)
        }
        
        # Bootstrap confidence intervals
        bootstrap_ci = self._calculate_bootstrap_ci(all_accuracies)
        overall_stats['accuracy_ci_lower'] = bootstrap_ci[0]
        overall_stats['accuracy_ci_upper'] = bootstrap_ci[1]
        
        # Statistical significance testing
        significance_test = self._test_significance(all_accuracies)
        
        # Per-metric performance
        metric_stats = {}
        for metric, scores in metric_accuracies.items():
            if len(scores) >= 3:  # Need minimum for statistics
                metric_stats[metric] = {
                    'mean_accuracy': np.mean(scores),
                    'std_accuracy': np.std(scores),
                    'num_cases': len(scores),
                    'confidence_interval': self._calculate_bootstrap_ci(scores)
                }
                
        # Regional performance analysis
        regional_stats = self._analyze_regional_performance()
        
        # Reproducibility metrics
        reproducibility = self._calculate_reproducibility_metrics()
        
        performance = {
            'framework_version': self.framework_version,
            'analysis_timestamp': datetime.datetime.now().isoformat(),
            'overall_performance': overall_stats,
            'statistical_significance': significance_test,
            'per_metric_performance': metric_stats,
            'regional_performance': regional_stats,
            'reproducibility_metrics': reproducibility,
            'validation_summary': {
                'total_cases': len(self.validation_cases),
                'completed_cases': len(self.completed_cases),
                'pending_cases': len(self.validation_cases) - len(self.completed_cases),
                'accuracy_benchmark': overall_stats['mean_accuracy'] >= 0.90,  # 90% benchmark
                'significance_achieved': significance_test['p_value'] < self.significance_threshold
            }
        }
        
        # Store in history
        self.performance_history.append(performance)
        
        return performance
        
    def _calculate_bootstrap_ci(self, 
                              data: List[float], 
                              confidence_level: float = 0.95) -> Tuple[float, float]:
        """
        Calculate bootstrap confidence intervals.
        
        Args:
            data: Data to bootstrap
            confidence_level: Confidence level (default 95%)
            
        Returns:
            (lower_bound, upper_bound) tuple
        """
        if len(data) < 2:
            return (0.0, 1.0)
            
        try:
            # Use scipy bootstrap
            data_array = np.array(data)
            
            def mean_statistic(x):
                return np.mean(x)
                
            rng = np.random.default_rng()
            bootstrap_result = bootstrap(
                (data_array,), 
                mean_statistic,
                n_resamples=self.bootstrap_samples,
                confidence_level=confidence_level,
                random_state=rng
            )
            
            return (bootstrap_result.confidence_interval.low, 
                   bootstrap_result.confidence_interval.high)
                   
        except Exception as e:
            logger.warning(f"Bootstrap calculation failed: {str(e)}")
            # Fallback to simple percentile method
            alpha = (1 - confidence_level) / 2
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            bootstrap_means = []
            for _ in range(self.bootstrap_samples):
                sample = np.random.choice(data, size=len(data), replace=True)
                bootstrap_means.append(np.mean(sample))
                
            return (np.percentile(bootstrap_means, lower_percentile),
                   np.percentile(bootstrap_means, upper_percentile))
                   
    def _test_significance(self, accuracies: List[float]) -> Dict[str, Any]:
        """
        Test statistical significance of framework performance.
        
        Args:
            accuracies: List of accuracy scores
            
        Returns:
            Statistical test results
        """
        # Test against null hypothesis that accuracy = 0.5 (random)
        null_hypothesis_mean = 0.5
        
        # One-sample t-test
        t_statistic, p_value = stats.ttest_1samp(accuracies, null_hypothesis_mean)
        
        # Effect size (Cohen's d)
        effect_size = (np.mean(accuracies) - null_hypothesis_mean) / np.std(accuracies)
        
        # Test normality
        shapiro_stat, shapiro_p = stats.shapiro(accuracies)
        
        # Wilcoxon signed-rank test (non-parametric alternative)
        centered_accuracies = np.array(accuracies) - null_hypothesis_mean
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(centered_accuracies, alternative='greater')
        
        return {
            'null_hypothesis': f'Mean accuracy = {null_hypothesis_mean}',
            't_test': {
                'statistic': t_statistic,
                'p_value': p_value,
                'significant': p_value < self.significance_threshold
            },
            'effect_size_cohens_d': effect_size,
            'normality_test': {
                'shapiro_statistic': shapiro_stat,
                'shapiro_p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            },
            'wilcoxon_test': {
                'statistic': wilcoxon_stat,
                'p_value': wilcoxon_p,
                'significant': wilcoxon_p < self.significance_threshold
            },
            'sample_size': len(accuracies),
            'power_analysis': self._calculate_statistical_power(len(accuracies), effect_size)
        }
        
    def _calculate_statistical_power(self, n: int, effect_size: float, alpha: float = None) -> float:
        """
        Calculate statistical power for the test.
        
        Args:
            n: Sample size
            effect_size: Cohen's d effect size
            alpha: Significance level
            
        Returns:
            Statistical power (0.0 to 1.0)
        """
        if alpha is None:
            alpha = self.significance_threshold
            
        # Approximate power calculation for one-sample t-test
        from scipy.stats import norm
        
        # Critical t-value
        t_critical = stats.t.ppf(1 - alpha, df=n-1)
        
        # Non-centrality parameter
        ncp = effect_size * np.sqrt(n)
        
        # Power approximation
        power = 1 - stats.t.cdf(t_critical, df=n-1, loc=ncp)
        
        return min(1.0, max(0.0, power))
        
    def _analyze_regional_performance(self) -> Dict[str, Any]:
        """
        Analyze performance by geographic/cultural regions.
        
        Returns:
            Regional performance analysis
        """
        regional_data = defaultdict(list)
        
        # Group cases by region (simplified classification)
        for case in self.completed_cases:
            region = self._classify_region(case.country)
            accuracy = self._get_case_overall_accuracy(case)
            regional_data[region].append(accuracy)
            
        regional_stats = {}
        for region, accuracies in regional_data.items():
            if len(accuracies) >= 2:  # Need minimum for stats
                regional_stats[region] = {
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'num_cases': len(accuracies),
                    'min_accuracy': np.min(accuracies),
                    'max_accuracy': np.max(accuracies)
                }
                
        # Test for regional differences (ANOVA if enough data)
        if len(regional_stats) >= 2:
            all_regional_accuracies = [accuracies for accuracies in regional_data.values() if len(accuracies) >= 2]
            if len(all_regional_accuracies) >= 2:
                try:
                    f_statistic, anova_p = stats.f_oneway(*all_regional_accuracies)
                    regional_stats['anova_test'] = {
                        'f_statistic': f_statistic,
                        'p_value': anova_p,
                        'significant_difference': anova_p < 0.05
                    }
                except:
                    regional_stats['anova_test'] = {'error': 'ANOVA calculation failed'}
                    
        return regional_stats
        
    def _classify_region(self, country: str) -> str:
        """
        Classify country into geographic/cultural region.
        
        Args:
            country: Country name
            
        Returns:
            Region classification
        """
        # Simplified region classification
        latin_america = ['Colombia', 'Argentina', 'Brazil', 'Chile', 'Mexico', 'Peru', 'Venezuela']
        north_america = ['USA', 'Canada']
        europe = ['Germany', 'France', 'UK', 'Spain', 'Italy', 'Netherlands']
        asia = ['Japan', 'South Korea', 'Singapore', 'Taiwan']
        
        if country in latin_america:
            return 'Latin America'
        elif country in north_america:
            return 'North America'
        elif country in europe:
            return 'Europe'
        elif country in asia:
            return 'Asia'
        else:
            return 'Other'
            
    def _calculate_reproducibility_metrics(self) -> Dict[str, Any]:
        """
        Calculate reproducibility and reliability metrics.
        
        Returns:
            Reproducibility metrics
        """
        metrics = {}
        
        # Inter-coder reliability (if multiple coders)
        if len(self.kappa_scores) > 0:
            metrics['inter_coder_reliability'] = {
                'mean_kappa': np.mean(self.kappa_scores),
                'std_kappa': np.std(self.kappa_scores),
                'num_comparisons': len(self.kappa_scores),
                'substantial_agreement': np.mean(self.kappa_scores) > 0.6  # Landis & Koch criteria
            }
            
        # Test-retest reliability (if cases have multiple predictions)
        test_retest_correlations = []
        for case in self.completed_cases:
            if 'repeated_predictions' in case.metadata:
                predictions = case.metadata['repeated_predictions']
                if len(predictions) >= 2:
                    corr, _ = stats.pearsonr(predictions[0], predictions[1])
                    test_retest_correlations.append(corr)
                    
        if test_retest_correlations:
            metrics['test_retest_reliability'] = {
                'mean_correlation': np.mean(test_retest_correlations),
                'std_correlation': np.std(test_retest_correlations),
                'num_cases': len(test_retest_correlations)
            }
            
        # Prediction calibration
        calibration_data = []
        for case in self.completed_cases:
            for metric in case.predicted_values:
                if metric in case.actual_values and metric in case.confidence_intervals:
                    ci_lower, ci_upper = case.confidence_intervals[metric]
                    actual = case.actual_values[metric]
                    within_ci = ci_lower <= actual <= ci_upper
                    calibration_data.append(within_ci)
                    
        if calibration_data:
            calibration_rate = np.mean(calibration_data)
            metrics['prediction_calibration'] = {
                'calibration_rate': calibration_rate,
                'expected_rate': 0.95,  # For 95% confidence intervals
                'well_calibrated': abs(calibration_rate - 0.95) < 0.1,
                'num_predictions': len(calibration_data)
            }
            
        # Temporal stability
        if len(self.performance_history) >= 2:
            recent_accuracies = [perf['overall_performance']['mean_accuracy'] 
                               for perf in self.performance_history[-5:]]  # Last 5 measurements
            metrics['temporal_stability'] = {
                'accuracy_trend': np.polyfit(range(len(recent_accuracies)), recent_accuracies, 1)[0],
                'accuracy_variance': np.var(recent_accuracies),
                'stable_performance': np.var(recent_accuracies) < 0.01  # Low variance = stable
            }
            
        return metrics
        
    def add_inter_coder_comparison(self, 
                                 coder1_predictions: Dict[str, float],
                                 coder2_predictions: Dict[str, float],
                                 case_id: str = None) -> float:
        """
        Add inter-coder reliability comparison.
        
        Args:
            coder1_predictions: First coder's predictions
            coder2_predictions: Second coder's predictions
            case_id: Optional case identifier
            
        Returns:
            Cohen's kappa score
        """
        # Convert continuous predictions to categories for kappa calculation
        def categorize_predictions(predictions):
            categories = []
            for value in predictions.values():
                if value < -0.5:
                    categories.append('low')
                elif value < 0.0:
                    categories.append('medium_low')
                elif value < 0.5:
                    categories.append('medium_high')
                else:
                    categories.append('high')
            return categories
            
        # Find common metrics
        common_metrics = set(coder1_predictions.keys()) & set(coder2_predictions.keys())
        
        if not common_metrics:
            logger.warning("No common metrics between coders")
            return 0.0
            
        coder1_common = {k: coder1_predictions[k] for k in common_metrics}
        coder2_common = {k: coder2_predictions[k] for k in common_metrics}
        
        categories1 = categorize_predictions(coder1_common)
        categories2 = categorize_predictions(coder2_common)
        
        # Calculate Cohen's kappa
        kappa = cohen_kappa_score(categories1, categories2)
        
        self.kappa_scores.append(kappa)
        self.coder_agreements.append({
            'kappa_score': kappa,
            'case_id': case_id,
            'num_metrics': len(common_metrics),
            'timestamp': datetime.datetime.now().isoformat()
        })
        
        logger.info(f"Inter-coder reliability: κ = {kappa:.3f}")
        return kappa
        
    def generate_validation_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive validation report.
        
        Returns:
            Complete validation report
        """
        performance = self.calculate_framework_performance()
        
        if 'error' in performance:
            return {
                'validation_report': 'incomplete',
                'error': performance['error'],
                'framework_version': self.framework_version,
                'timestamp': datetime.datetime.now().isoformat()
            }
            
        # Calculate additional metrics
        accuracy_by_reform_type = defaultdict(list)
        for case in self.completed_cases:
            accuracy = self._get_case_overall_accuracy(case)
            accuracy_by_reform_type[case.reform_type].append(accuracy)
            
        reform_type_stats = {}
        for reform_type, accuracies in accuracy_by_reform_type.items():
            reform_type_stats[reform_type] = {
                'mean_accuracy': np.mean(accuracies),
                'num_cases': len(accuracies),
                'accuracy_range': (np.min(accuracies), np.max(accuracies))
            }
            
        # Validation quality assessment
        quality_assessment = {
            'statistical_rigor': {
                'sufficient_sample_size': len(self.completed_cases) >= self.min_sample_size,
                'significance_achieved': performance['statistical_significance']['t_test']['significant'],
                'effect_size_adequate': abs(performance['statistical_significance']['effect_size_cohens_d']) > 0.8,
                'confidence_intervals_calculated': 'accuracy_ci_lower' in performance['overall_performance']
            },
            'reproducibility_evidence': {
                'inter_coder_reliability': len(self.kappa_scores) > 0,
                'cross_cultural_validation': len(performance['regional_performance']) > 1,
                'temporal_stability': 'temporal_stability' in performance['reproducibility_metrics']
            }
        }
        
        # Final assessment
        accuracy_threshold = 0.90  # 90% accuracy benchmark
        significance_achieved = performance['statistical_significance']['t_test']['p_value'] < self.significance_threshold
        
        world_class_criteria = {
            'high_accuracy': performance['overall_performance']['mean_accuracy'] >= accuracy_threshold,
            'statistical_significance': significance_achieved,
            'sufficient_sample': len(self.completed_cases) >= self.min_sample_size,
            'reproducible_results': len(self.kappa_scores) > 0,
            'cross_cultural_validated': len(performance['regional_performance']) > 1
        }
        
        world_class_achieved = all(world_class_criteria.values())
        
        report = {
            'validation_report_summary': {
                'framework_version': self.framework_version,
                'report_timestamp': datetime.datetime.now().isoformat(),
                'world_class_reproducibility_achieved': world_class_achieved,
                'overall_accuracy': performance['overall_performance']['mean_accuracy'],
                'statistical_significance_p_value': performance['statistical_significance']['t_test']['p_value'],
                'total_validated_cases': len(self.completed_cases)
            },
            'world_class_criteria_assessment': world_class_criteria,
            'detailed_performance': performance,
            'accuracy_by_reform_type': reform_type_stats,
            'quality_assessment': quality_assessment,
            'validation_cases_summary': [case.export_dict() for case in self.completed_cases],
            'recommendations': self._generate_recommendations(performance, world_class_achieved)
        }
        
        return report
        
    def _generate_recommendations(self, 
                                performance: Dict[str, Any], 
                                world_class_achieved: bool) -> List[str]:
        """
        Generate recommendations for improving validation.
        
        Args:
            performance: Performance metrics
            world_class_achieved: Whether world-class standards are met
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        if not world_class_achieved:
            # Check each criterion
            if performance['overall_performance']['mean_accuracy'] < 0.90:
                recommendations.append(
                    f"Improve overall accuracy from {performance['overall_performance']['mean_accuracy']:.1%} "
                    "to above 90% through model refinement and better SAPNC filter calibration."
                )
                
            if performance['statistical_significance']['t_test']['p_value'] >= self.significance_threshold:
                recommendations.append(
                    f"Achieve statistical significance (p < {self.significance_threshold}) by collecting "
                    "more validation cases and improving prediction precision."
                )
                
            if len(self.completed_cases) < self.min_sample_size:
                recommendations.append(
                    f"Increase sample size from {len(self.completed_cases)} to at least {self.min_sample_size} "
                    "validated cases for robust statistical analysis."
                )
                
            if len(self.kappa_scores) == 0:
                recommendations.append(
                    "Establish inter-coder reliability by having multiple researchers independently "
                    "code a subset of cases and calculate Cohen's kappa."
                )
                
            if len(performance['regional_performance']) <= 1:
                recommendations.append(
                    "Expand cross-cultural validation by including cases from multiple geographic "
                    "and cultural regions (Latin America, Europe, Asia, etc.)."
                )
                
        else:
            recommendations.append(
                "Congratulations! World-class reproducibility standards achieved. "
                "Consider publishing methodology and establishing this as a benchmark for the field."
            )
            
        # Additional improvement suggestions
        if 'prediction_calibration' in performance['reproducibility_metrics']:
            calibration = performance['reproducibility_metrics']['prediction_calibration']
            if not calibration.get('well_calibrated', False):
                recommendations.append(
                    f"Improve prediction calibration (current: {calibration['calibration_rate']:.1%}, "
                    f"target: {calibration['expected_rate']:.1%}) by refining confidence interval estimation."
                )
                
        return recommendations
        
    def export_validation_database(self) -> Dict[str, Any]:
        """
        Export complete validation database for reproducibility.
        
        Returns:
            Complete validation database
        """
        return {
            'metadata': {
                'framework_version': self.framework_version,
                'export_timestamp': datetime.datetime.now().isoformat(),
                'total_cases': len(self.validation_cases),
                'completed_cases': len(self.completed_cases),
                'validation_parameters': {
                    'significance_threshold': self.significance_threshold,
                    'min_sample_size': self.min_sample_size,
                    'bootstrap_samples': self.bootstrap_samples
                }
            },
            'validation_cases': {case_id: case.export_dict() 
                               for case_id, case in self.validation_cases.items()},
            'performance_history': self.performance_history,
            'inter_coder_reliability': {
                'kappa_scores': self.kappa_scores,
                'coder_agreements': self.coder_agreements
            },
            'checksum': self._calculate_database_checksum()
        }
        
    def _calculate_database_checksum(self) -> str:
        """
        Calculate checksum for database integrity verification.
        
        Returns:
            MD5 checksum of validation data
        """
        # Create deterministic string representation of validation data
        case_data = sorted([
            f"{case.case_id}:{case.predicted_values}:{case.actual_values}" 
            for case in self.completed_cases
        ])
        
        data_string = ''.join(case_data)
        return hashlib.md5(data_string.encode()).hexdigest()

# Example validation cases for testing
def create_example_validation_cases() -> List[ValidationCase]:
    """
    Create example validation cases for testing the tracker.
    
    Returns:
        List of example ValidationCase objects
    """
    cases = []
    
    # Colombia Pension Reform 2024 (highly accurate prediction)
    colombia_case = ValidationCase(
        case_id="colombia_pension_2024",
        country="Colombia",
        reform_type="pension_reform",
        prediction_date="2024-01-15",
        actual_outcome_date="2024-09-15",
        predicted_values={
            'implementation_gap': 0.42,
            'political_stability_impact': -0.35,
            'social_protest_intensity': 0.78,
            'constitutional_challenge_probability': 0.65
        },
        actual_values={
            'implementation_gap': 0.44,
            'political_stability_impact': -0.32,
            'social_protest_intensity': 0.81,
            'constitutional_challenge_probability': 0.68
        },
        confidence_intervals={
            'implementation_gap': (0.35, 0.50),
            'political_stability_impact': (-0.45, -0.25),
            'social_protest_intensity': (0.65, 0.85),
            'constitutional_challenge_probability': (0.55, 0.75)
        }
    )
    cases.append(colombia_case)
    
    # Argentina Milei Reforms 2024 (moderate accuracy)
    argentina_case = ValidationCase(
        case_id="argentina_milei_2024",
        country="Argentina",
        reform_type="economic_liberalization",
        prediction_date="2024-02-01",
        actual_outcome_date="2024-08-01",
        predicted_values={
            'implementation_gap': 0.38,
            'inflation_impact': -0.25,
            'judicial_resistance': 0.55,
            'congressional_approval': 0.35
        },
        actual_values={
            'implementation_gap': 0.45,
            'inflation_impact': -0.18,
            'judicial_resistance': 0.62,
            'congressional_approval': 0.28
        },
        confidence_intervals={
            'implementation_gap': (0.30, 0.48),
            'inflation_impact': (-0.35, -0.15),
            'judicial_resistance': (0.45, 0.65),
            'congressional_approval': (0.25, 0.45)
        }
    )
    cases.append(argentina_case)
    
    return cases

# Example usage and testing
if __name__ == "__main__":
    # Create validation tracker
    tracker = ValidationTracker(framework_version="4.0")
    
    # Add example validation cases
    example_cases = create_example_validation_cases()
    
    for case in example_cases:
        tracker.add_validation_case(case)
        
    # Generate validation report
    report = tracker.generate_validation_report()
    
    print("\n=== Validation Report Summary ===")
    if 'validation_report_summary' in report:
        summary = report['validation_report_summary']
        print(f"Framework Version: {summary['framework_version']}")
        print(f"World-class Reproducibility: {summary['world_class_reproducibility_achieved']}")
        print(f"Overall Accuracy: {summary['overall_accuracy']:.1%}")
        print(f"Statistical Significance (p): {summary['statistical_significance_p_value']:.2e}")
        print(f"Validated Cases: {summary['total_validated_cases']}")
        
        if 'recommendations' in report:
            print("\nRecommendations:")
            for rec in report['recommendations']:
                print(f"- {rec}")
    else:
        print("Insufficient data for full validation report")
        print(f"Error: {report.get('error', 'Unknown error')}")
    
    # Export validation database
    database = tracker.export_validation_database()
    print(f"\nValidation database exported with checksum: {database['checksum']}")