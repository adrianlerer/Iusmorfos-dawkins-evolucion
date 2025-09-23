#!/usr/bin/env python3
"""
Diagnostic Plots and Statistical Validation for Iusmorfos
=========================================================

Comprehensive diagnostic system for validating evolutionary legal system models.
Following gold-standard model validation practices:

1. Model diagnostics (residuals, QQ-plots, leverage)
2. Cross-validation diagnostics
3. Bootstrap validation
4. Convergence diagnostics
5. Sensitivity analysis plots
6. Model assumption testing

Author: Adrian Lerer & AI Assistant
Date: 2025-09-23
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import jarque_bera, shapiro, anderson, kstest
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import mean_squared_error, mean_absolute_error
from typing import Dict, List, Tuple, Any, Optional, Union, Callable
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from robustness import RobustnessAnalyzer


class ModelDiagnostics:
    """
    Comprehensive model diagnostics and validation system.
    
    Provides statistical validation plots and tests to assess:
    - Model assumptions (normality, homoscedasticity, independence)
    - Model performance (bias, variance, convergence) 
    - Data quality (outliers, leverage points, influential observations)
    - Robustness (sensitivity to assumptions, bootstrap stability)
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize diagnostics system."""
        self.config = get_config()
        
        if random_seed is None:
            random_seed = self.config.config['reproducibility']['random_seed']
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Configure plotting
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Storage for diagnostic results
        self.diagnostic_results = {}
        
        print(f"🔍 Model Diagnostics initialized")
        print(f"   🔒 Random seed: {random_seed}")
    
    def plot_residual_diagnostics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                model_name: str = "Model") -> Dict[str, Any]:
        """
        Create comprehensive residual diagnostic plots.
        
        Args:
            y_true: True target values
            y_pred: Predicted values
            model_name: Name of the model for plot titles
            
        Returns:
            Dictionary with diagnostic statistics
        """
        print(f"🔄 Creating residual diagnostics for {model_name}...")
        
        # Calculate residuals
        residuals = y_true - y_pred
        standardized_residuals = residuals / np.std(residuals) if np.std(residuals) > 0 else residuals
        
        # Create diagnostic plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Residuals vs Fitted Values
        ax1.scatter(y_pred, residuals, alpha=0.6, s=30)
        ax1.axhline(y=0, color='red', linestyle='--', linewidth=2)
        
        # Add LOWESS smooth line
        try:
            from sklearn.preprocessing import PolynomialFeatures
            from sklearn.linear_model import LinearRegression
            from sklearn.pipeline import Pipeline
            
            # Simple polynomial smoothing as LOWESS alternative
            sorted_indices = np.argsort(y_pred)
            y_pred_sorted = y_pred[sorted_indices]
            residuals_sorted = residuals[sorted_indices]
            
            # Fit polynomial to show trend
            z = np.polyfit(y_pred_sorted, residuals_sorted, 2)
            trend = np.poly1d(z)
            ax1.plot(y_pred_sorted, trend(y_pred_sorted), 'blue', linewidth=2, alpha=0.8, label='Trend')
            ax1.legend()
        except:
            pass
        
        ax1.set_xlabel('Fitted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title(f'📊 Residuals vs Fitted ({model_name})', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. Q-Q Plot for Normality
        stats.probplot(residuals, dist="norm", plot=ax2)
        ax2.set_title(f'📈 Normal Q-Q Plot ({model_name})', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Calculate R-squared for normality assessment
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        r_squared = stats.pearsonr(theoretical_quantiles, sample_quantiles)[0]**2
        ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax2.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # 3. Scale-Location Plot (Standardized residuals vs fitted)
        sqrt_abs_resid = np.sqrt(np.abs(standardized_residuals))
        ax3.scatter(y_pred, sqrt_abs_resid, alpha=0.6, s=30)
        
        # Add trend line
        try:
            z = np.polyfit(y_pred, sqrt_abs_resid, 1)
            trend = np.poly1d(z)
            ax3.plot(np.sort(y_pred), trend(np.sort(y_pred)), 'red', linewidth=2, alpha=0.8)
        except:
            pass
        
        ax3.set_xlabel('Fitted Values')
        ax3.set_ylabel('√|Standardized Residuals|')
        ax3.set_title(f'📏 Scale-Location Plot ({model_name})', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Residual Distribution
        ax4.hist(residuals, bins=20, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        
        # Overlay normal distribution
        mu, sigma = stats.norm.fit(residuals)
        x_norm = np.linspace(residuals.min(), residuals.max(), 100)
        ax4.plot(x_norm, stats.norm.pdf(x_norm, mu, sigma), 'red', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
        
        ax4.set_xlabel('Residuals')
        ax4.set_ylabel('Density')
        ax4.set_title(f'📊 Residual Distribution ({model_name})', fontsize=14, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../outputs/residual_diagnostics_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Statistical tests
        diagnostic_stats = self._calculate_diagnostic_statistics(residuals, y_true, y_pred)
        diagnostic_stats['model_name'] = model_name
        
        self.diagnostic_results[f'residual_diagnostics_{model_name}'] = diagnostic_stats
        
        print(f"✅ Residual diagnostics completed for {model_name}")
        return diagnostic_stats
    
    def _calculate_diagnostic_statistics(self, residuals: np.ndarray, 
                                       y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calculate comprehensive diagnostic statistics."""
        
        stats_dict = {}
        
        # Basic residual statistics
        stats_dict['residual_stats'] = {
            'mean': float(np.mean(residuals)),
            'std': float(np.std(residuals)),
            'min': float(np.min(residuals)),
            'max': float(np.max(residuals)),
            'skewness': float(stats.skew(residuals)),
            'kurtosis': float(stats.kurtosis(residuals))
        }
        
        # Model performance metrics
        stats_dict['performance'] = {
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
            'r2': float(stats.pearsonr(y_true, y_pred)[0]**2) if len(y_true) > 1 else 0
        }
        
        # Normality tests
        try:
            shapiro_stat, shapiro_p = shapiro(residuals)
            stats_dict['normality_tests'] = {
                'shapiro_wilk': {'statistic': float(shapiro_stat), 'p_value': float(shapiro_p)},
                'interpretation': 'normal' if shapiro_p > 0.05 else 'non_normal'
            }
        except:
            stats_dict['normality_tests'] = {'error': 'Failed to calculate normality tests'}
        
        # Homoscedasticity test (Breusch-Pagan test approximation)
        try:
            # Simple correlation test between squared residuals and fitted values
            squared_residuals = residuals ** 2
            corr_coef, corr_p = stats.pearsonr(y_pred, squared_residuals)
            
            stats_dict['homoscedasticity'] = {
                'correlation_coefficient': float(corr_coef),
                'p_value': float(corr_p),
                'interpretation': 'homoscedastic' if corr_p > 0.05 else 'heteroscedastic'
            }
        except:
            stats_dict['homoscedasticity'] = {'error': 'Failed to test homoscedasticity'}
        
        # Outlier detection (using IQR method)
        try:
            q1, q3 = np.percentile(residuals, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            outliers = (residuals < lower_bound) | (residuals > upper_bound)
            
            stats_dict['outliers'] = {
                'count': int(np.sum(outliers)),
                'percentage': float(np.mean(outliers) * 100),
                'indices': np.where(outliers)[0].tolist()
            }
        except:
            stats_dict['outliers'] = {'error': 'Failed to detect outliers'}
        
        return stats_dict
    
    def plot_convergence_diagnostics(self, evolution_data: List[float],
                                   generation_labels: Optional[List[int]] = None,
                                   model_name: str = "Evolution") -> Dict[str, Any]:
        """
        Plot convergence diagnostics for evolutionary models.
        
        Args:
            evolution_data: Time series of evolution metrics
            generation_labels: Optional generation labels
            model_name: Name of the evolutionary model
            
        Returns:
            Convergence diagnostic statistics
        """
        print(f"🔄 Creating convergence diagnostics for {model_name}...")
        
        if len(evolution_data) < 5:
            print("⚠️ Insufficient data for convergence analysis")
            return {}
        
        evolution_array = np.array(evolution_data)
        
        if generation_labels is None:
            generation_labels = list(range(len(evolution_data)))
        
        # Create convergence plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Evolution Trajectory with Confidence Bands
        ax1.plot(generation_labels, evolution_data, 'b-', linewidth=2, marker='o', markersize=4, label='Evolution')
        
        # Add moving average
        window_size = max(3, len(evolution_data) // 10)
        moving_avg = pd.Series(evolution_data).rolling(window=window_size, center=True).mean()
        ax1.plot(generation_labels, moving_avg, 'r--', linewidth=2, alpha=0.7, label=f'Moving Average ({window_size})')
        
        # Add confidence bands (using moving standard deviation)
        moving_std = pd.Series(evolution_data).rolling(window=window_size, center=True).std()
        upper_band = moving_avg + 1.96 * moving_std
        lower_band = moving_avg - 1.96 * moving_std
        ax1.fill_between(generation_labels, lower_band, upper_band, alpha=0.2, color='gray', label='95% CI')
        
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Evolution Metric')
        ax1.set_title(f'📈 Evolution Trajectory with Confidence Bands ({model_name})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. First Differences (Generation-to-Generation Changes)
        if len(evolution_data) > 1:
            first_diff = np.diff(evolution_data)
            ax2.plot(generation_labels[1:], first_diff, 'g-', linewidth=2, marker='s', markersize=4)
            ax2.axhline(y=0, color='red', linestyle='--', alpha=0.5)
            
            # Add trend line
            z = np.polyfit(generation_labels[1:], first_diff, 1)
            trend = np.poly1d(z)
            ax2.plot(generation_labels[1:], trend(generation_labels[1:]), 'orange', linewidth=2, alpha=0.7, label='Trend')
            
            ax2.set_xlabel('Generation')
            ax2.set_ylabel('Change from Previous Generation')
            ax2.set_title(f'📊 First Differences ({model_name})', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Convergence Rate Analysis
        if len(evolution_data) >= 10:
            # Calculate convergence rate using exponential decay model
            generations = np.array(generation_labels)
            
            # Fit exponential decay to the rate of change
            try:
                # Simple approach: fit to absolute differences
                abs_diff = np.abs(first_diff) if len(evolution_data) > 1 else [0]
                
                if len(abs_diff) > 5:
                    # Exponential fit: y = a * exp(-b * x) + c
                    from scipy.optimize import curve_fit
                    
                    def exp_decay(x, a, b, c):
                        return a * np.exp(-b * x) + c
                    
                    try:
                        popt, _ = curve_fit(exp_decay, generations[1:len(abs_diff)+1], abs_diff, 
                                          p0=[1, 0.1, 0], maxfev=1000)
                        
                        x_fit = np.linspace(generations[1], generations[-1], 100)
                        y_fit = exp_decay(x_fit, *popt)
                        
                        ax3.plot(generations[1:len(abs_diff)+1], abs_diff, 'purple', 
                               linewidth=2, marker='D', markersize=4, label='|Rate of Change|')
                        ax3.plot(x_fit, y_fit, 'red', linewidth=2, alpha=0.7, 
                               label=f'Exponential Fit (τ = {1/popt[1]:.1f})')
                        
                    except:
                        # Fallback: just plot the data
                        ax3.plot(generations[1:len(abs_diff)+1], abs_diff, 'purple', 
                               linewidth=2, marker='D', markersize=4)
                
                ax3.set_xlabel('Generation')
                ax3.set_ylabel('|Rate of Change|')
                ax3.set_title(f'⏱️ Convergence Rate Analysis ({model_name})', fontsize=14, fontweight='bold')
                ax3.set_yscale('log')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
            except Exception as e:
                ax3.text(0.5, 0.5, f'Convergence analysis\nfailed: {str(e)[:50]}...', 
                        ha='center', va='center', transform=ax3.transAxes)
        
        # 4. Autocorrelation Function
        if len(evolution_data) >= 10:
            max_lag = min(len(evolution_data) // 3, 20)
            
            autocorr = []
            lags = []
            
            for lag in range(1, max_lag + 1):
                if len(evolution_data) > lag:
                    corr, _ = stats.pearsonr(evolution_data[:-lag], evolution_data[lag:])
                    autocorr.append(corr)
                    lags.append(lag)
            
            if autocorr:
                ax4.plot(lags, autocorr, 'orange', linewidth=2, marker='o', markersize=5)
                ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax4.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Strong correlation')
                ax4.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)
                
                ax4.set_xlabel('Lag (Generations)')
                ax4.set_ylabel('Autocorrelation')
                ax4.set_title(f'🔄 Autocorrelation Function ({model_name})', fontsize=14, fontweight='bold')
                ax4.set_ylim(-1, 1)
                ax4.legend()
                ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'../outputs/convergence_diagnostics_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate convergence statistics
        convergence_stats = self._calculate_convergence_statistics(evolution_data)
        convergence_stats['model_name'] = model_name
        
        self.diagnostic_results[f'convergence_diagnostics_{model_name}'] = convergence_stats
        
        print(f"✅ Convergence diagnostics completed for {model_name}")
        return convergence_stats
    
    def _calculate_convergence_statistics(self, evolution_data: List[float]) -> Dict[str, Any]:
        """Calculate convergence diagnostic statistics."""
        
        stats_dict = {}
        evolution_array = np.array(evolution_data)
        
        # Basic convergence metrics
        if len(evolution_data) > 1:
            first_diff = np.diff(evolution_array)
            
            stats_dict['convergence_metrics'] = {
                'final_rate_of_change': float(first_diff[-1]) if len(first_diff) > 0 else 0,
                'mean_rate_of_change': float(np.mean(first_diff)),
                'std_rate_of_change': float(np.std(first_diff)),
                'trend_slope': float(stats.linregress(range(len(evolution_data)), evolution_data)[0]),
                'is_converging': float(first_diff[-1]) < 0.01 if len(first_diff) > 0 else False
            }
        
        # Stability metrics
        if len(evolution_data) >= 5:
            last_quarter = evolution_array[-len(evolution_array)//4:]
            
            stats_dict['stability'] = {
                'coefficient_of_variation': float(np.std(last_quarter) / np.mean(last_quarter)) if np.mean(last_quarter) != 0 else 0,
                'range_ratio': float((np.max(last_quarter) - np.min(last_quarter)) / np.mean(last_quarter)) if np.mean(last_quarter) != 0 else 0,
                'is_stable': np.std(last_quarter) / np.mean(last_quarter) < 0.05 if np.mean(last_quarter) != 0 else False
            }
        
        return stats_dict
    
    def plot_bootstrap_diagnostics(self, data: np.ndarray, 
                                 statistic_func: Callable = np.mean,
                                 n_bootstrap: int = 1000,
                                 confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Create bootstrap diagnostic plots for statistical robustness.
        
        Args:
            data: Input data for bootstrap analysis
            statistic_func: Function to calculate statistic (default: mean)
            n_bootstrap: Number of bootstrap samples
            confidence_level: Confidence level for intervals
            
        Returns:
            Bootstrap diagnostic results
        """
        print(f"🔄 Creating bootstrap diagnostics (n={n_bootstrap})...")
        
        if len(data) < 3:
            print("⚠️ Insufficient data for bootstrap analysis")
            return {}
        
        # Generate bootstrap samples
        bootstrap_stats = []
        
        for i in range(n_bootstrap):
            bootstrap_sample = np.random.choice(data, size=len(data), replace=True)
            stat_value = statistic_func(bootstrap_sample)
            bootstrap_stats.append(stat_value)
        
        bootstrap_array = np.array(bootstrap_stats)
        original_stat = statistic_func(data)
        
        # Calculate confidence intervals
        alpha = 1 - confidence_level
        ci_lower = np.percentile(bootstrap_array, (alpha/2) * 100)
        ci_upper = np.percentile(bootstrap_array, (1 - alpha/2) * 100)
        
        # Create bootstrap diagnostic plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Bootstrap Distribution
        ax1.hist(bootstrap_array, bins=50, density=True, alpha=0.7, color='lightblue', edgecolor='black')
        ax1.axvline(original_stat, color='red', linestyle='-', linewidth=2, label=f'Original = {original_stat:.4f}')
        ax1.axvline(ci_lower, color='orange', linestyle='--', linewidth=2, label=f'CI Lower = {ci_lower:.4f}')
        ax1.axvline(ci_upper, color='orange', linestyle='--', linewidth=2, label=f'CI Upper = {ci_upper:.4f}')
        
        ax1.set_xlabel('Bootstrap Statistic')
        ax1.set_ylabel('Density')
        ax1.set_title(f'📊 Bootstrap Distribution (n={n_bootstrap})', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Bootstrap Convergence (Running Average)
        running_means = np.cumsum(bootstrap_array) / np.arange(1, len(bootstrap_array) + 1)
        
        ax2.plot(range(1, len(running_means) + 1), running_means, 'blue', linewidth=1, alpha=0.7)
        ax2.axhline(original_stat, color='red', linestyle='-', linewidth=2, label='Original Statistic')
        ax2.axhline(np.mean(bootstrap_array), color='green', linestyle='--', linewidth=2, label='Bootstrap Mean')
        
        ax2.set_xlabel('Bootstrap Sample Number')
        ax2.set_ylabel('Running Mean')
        ax2.set_title('🔄 Bootstrap Convergence', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Q-Q Plot for Bootstrap Distribution Normality
        stats.probplot(bootstrap_array, dist="norm", plot=ax3)
        ax3.set_title('📈 Bootstrap Distribution Q-Q Plot', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Bias and Variance Analysis
        bias = np.mean(bootstrap_array) - original_stat
        variance = np.var(bootstrap_array)
        
        # Create bias-variance visualization
        ax4.bar(['Bias', 'Std Error', 'Original Stat'], 
               [abs(bias), np.sqrt(variance), abs(original_stat)], 
               color=['red', 'orange', 'blue'], alpha=0.7)
        
        ax4.set_ylabel('Magnitude')
        ax4.set_title('📊 Bias-Variance Analysis', fontsize=14, fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (label, value) in enumerate(zip(['Bias', 'Std Error', 'Original Stat'], 
                                             [abs(bias), np.sqrt(variance), abs(original_stat)])):
            ax4.text(i, value + 0.01, f'{value:.4f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('../outputs/bootstrap_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate bootstrap statistics
        bootstrap_results = {
            'original_statistic': float(original_stat),
            'bootstrap_mean': float(np.mean(bootstrap_array)),
            'bootstrap_std': float(np.std(bootstrap_array)),
            'bias': float(bias),
            'variance': float(variance),
            'confidence_interval': [float(ci_lower), float(ci_upper)],
            'confidence_level': confidence_level,
            'n_bootstrap': n_bootstrap
        }
        
        # Statistical tests on bootstrap distribution
        try:
            shapiro_stat, shapiro_p = shapiro(bootstrap_array)
            bootstrap_results['normality_test'] = {
                'statistic': float(shapiro_stat),
                'p_value': float(shapiro_p),
                'is_normal': shapiro_p > 0.05
            }
        except:
            bootstrap_results['normality_test'] = {'error': 'Failed to test normality'}
        
        self.diagnostic_results['bootstrap_diagnostics'] = bootstrap_results
        
        print("✅ Bootstrap diagnostics completed")
        return bootstrap_results
    
    def create_model_comparison_diagnostics(self, models_performance: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create diagnostic plots comparing multiple models.
        
        Args:
            models_performance: Dictionary with model names and their performance metrics
            
        Returns:
            Model comparison diagnostic results
        """
        print("🔄 Creating model comparison diagnostics...")
        
        if not models_performance:
            print("⚠️ No model performance data provided")
            return {}
        
        # Extract performance metrics
        model_names = list(models_performance.keys())
        
        # Collect metrics
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for model_name, perf in models_performance.items():
            accuracies.append(perf.get('accuracy', 0))
            precisions.append(perf.get('precision', 0))
            recalls.append(perf.get('recall', 0))
            f1_scores.append(perf.get('f1_score', 0))
        
        # Create comparison plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # 1. Performance Metrics Comparison
        x_pos = np.arange(len(model_names))
        width = 0.2
        
        ax1.bar(x_pos - 1.5*width, accuracies, width, label='Accuracy', alpha=0.8)
        ax1.bar(x_pos - 0.5*width, precisions, width, label='Precision', alpha=0.8)
        ax1.bar(x_pos + 0.5*width, recalls, width, label='Recall', alpha=0.8)
        ax1.bar(x_pos + 1.5*width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Performance Metrics')
        ax1.set_title('📊 Model Performance Comparison', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels([name.replace('_', ' ').title() for name in model_names], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
        
        # 2. Performance Distribution
        all_metrics = [accuracies, precisions, recalls, f1_scores]
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        ax2.boxplot(all_metrics, labels=metric_names)
        ax2.set_ylabel('Score')
        ax2.set_title('📦 Performance Distribution Across Models', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. Model Ranking
        # Calculate average performance across all metrics
        avg_performances = []
        for i in range(len(model_names)):
            avg_perf = np.mean([accuracies[i], precisions[i], recalls[i], f1_scores[i]])
            avg_performances.append(avg_perf)
        
        # Sort by performance
        sorted_indices = np.argsort(avg_performances)[::-1]  # Descending order
        sorted_names = [model_names[i] for i in sorted_indices]
        sorted_perfs = [avg_performances[i] for i in sorted_indices]
        
        colors = ['gold', 'silver', '#CD7F32'] + ['lightblue'] * (len(sorted_names) - 3)
        colors = colors[:len(sorted_names)]
        
        bars = ax3.barh(range(len(sorted_names)), sorted_perfs, color=colors, alpha=0.8)
        ax3.set_yticks(range(len(sorted_names)))
        ax3.set_yticklabels([name.replace('_', ' ').title() for name in sorted_names])
        ax3.set_xlabel('Average Performance')
        ax3.set_title('🏆 Model Ranking', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Add performance labels
        for i, (bar, perf) in enumerate(zip(bars, sorted_perfs)):
            ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                    f'{perf:.3f}', va='center', fontweight='bold')
        
        # 4. Performance Correlation Matrix
        if len(model_names) > 1:
            # Create performance matrix
            perf_matrix = np.array([accuracies, precisions, recalls, f1_scores])
            
            # Calculate correlation between metrics
            corr_matrix = np.corrcoef(perf_matrix)
            
            im = ax4.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax4.set_xticks(range(len(metric_names)))
            ax4.set_yticks(range(len(metric_names)))
            ax4.set_xticklabels(metric_names, rotation=45)
            ax4.set_yticklabels(metric_names)
            ax4.set_title('🔗 Metric Correlation Matrix', fontsize=14, fontweight='bold')
            
            # Add correlation values
            for i in range(len(metric_names)):
                for j in range(len(metric_names)):
                    ax4.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                           ha='center', va='center', fontweight='bold',
                           color='white' if abs(corr_matrix[i, j]) > 0.5 else 'black')
            
            plt.colorbar(im, ax=ax4, shrink=0.8)
        
        plt.tight_layout()
        plt.savefig('../outputs/model_comparison_diagnostics.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Generate comparison statistics
        comparison_stats = {
            'best_model': sorted_names[0] if sorted_names else None,
            'best_accuracy': max(accuracies) if accuracies else 0,
            'performance_spread': max(avg_performances) - min(avg_performances) if avg_performances else 0,
            'model_rankings': {name: rank + 1 for rank, name in enumerate(sorted_names)},
            'average_performances': {name: perf for name, perf in zip(sorted_names, sorted_perfs)}
        }
        
        self.diagnostic_results['model_comparison'] = comparison_stats
        
        print("✅ Model comparison diagnostics completed")
        return comparison_stats
    
    def generate_comprehensive_diagnostic_report(self) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report from all analyses."""
        
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'diagnostic_results': self.diagnostic_results,
            'summary': {
                'total_diagnostics_performed': len(self.diagnostic_results),
                'key_findings': [],
                'recommendations': []
            }
        }
        
        # Analyze results and generate findings
        findings = []
        recommendations = []
        
        # Check residual diagnostics
        for key, result in self.diagnostic_results.items():
            if 'residual_diagnostics' in key and 'normality_tests' in result:
                if result['normality_tests'].get('interpretation') == 'normal':
                    findings.append(f"Residuals are normally distributed for {result['model_name']}")
                else:
                    findings.append(f"Non-normal residuals detected for {result['model_name']}")
                    recommendations.append(f"Consider robust regression methods for {result['model_name']}")
        
        # Check convergence diagnostics
        for key, result in self.diagnostic_results.items():
            if 'convergence_diagnostics' in key and 'convergence_metrics' in result:
                if result['convergence_metrics'].get('is_converging'):
                    findings.append(f"Model {result['model_name']} shows convergence")
                else:
                    findings.append(f"Model {result['model_name']} may not be converging")
                    recommendations.append(f"Extend evolution time or adjust parameters for {result['model_name']}")
        
        # General recommendations
        recommendations.extend([
            "Implement cross-validation for all model evaluations",
            "Consider ensemble methods to improve robustness",
            "Validate findings with external datasets",
            "Monitor model performance over time"
        ])
        
        report['summary']['key_findings'] = findings
        report['summary']['recommendations'] = recommendations
        
        # Save report
        output_path = self.config.get_path('results_dir') / f"diagnostic_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Comprehensive diagnostic report saved: {output_path}")
        return report


def demo_model_diagnostics():
    """Demonstration of model diagnostics system."""
    print("🔍 Model Diagnostics Demo")
    print("=" * 25)
    
    # Create diagnostics system
    diagnostics = ModelDiagnostics()
    
    # Generate mock data
    np.random.seed(42)
    
    # Mock model predictions vs true values
    n_samples = 100
    y_true = np.random.normal(5, 2, n_samples)
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)  # Add some prediction error
    
    # Mock evolution data
    evolution_data = [1.0 + i * 0.1 + np.random.normal(0, 0.05) for i in range(30)]
    
    # Mock bootstrap data
    bootstrap_data = np.random.gamma(2, 1.5, 50)
    
    # Run diagnostics
    print("\n1. Residual diagnostics:")
    residual_results = diagnostics.plot_residual_diagnostics(y_true, y_pred, "Demo Model")
    
    print("\n2. Convergence diagnostics:")
    convergence_results = diagnostics.plot_convergence_diagnostics(evolution_data, model_name="Demo Evolution")
    
    print("\n3. Bootstrap diagnostics:")
    bootstrap_results = diagnostics.plot_bootstrap_diagnostics(bootstrap_data, n_bootstrap=500)
    
    # Mock model comparison data
    models_perf = {
        'dummy_classifier': {'accuracy': 0.60, 'precision': 0.58, 'recall': 0.55, 'f1_score': 0.56},
        'logistic_regression': {'accuracy': 0.75, 'precision': 0.73, 'recall': 0.72, 'f1_score': 0.72},
        'random_forest': {'accuracy': 0.82, 'precision': 0.81, 'recall': 0.80, 'f1_score': 0.80},
        'iusmorfos_model': {'accuracy': 0.85, 'precision': 0.84, 'recall': 0.83, 'f1_score': 0.83}
    }
    
    print("\n4. Model comparison diagnostics:")
    comparison_results = diagnostics.create_model_comparison_diagnostics(models_perf)
    
    # Generate comprehensive report
    print("\n5. Generating comprehensive diagnostic report:")
    report = diagnostics.generate_comprehensive_diagnostic_report()
    
    print("\n📊 Diagnostics Complete!")
    print(f"   🔍 Total analyses: {report['summary']['total_diagnostics_performed']}")
    print(f"   📋 Key findings: {len(report['summary']['key_findings'])}")
    print(f"   💡 Recommendations: {len(report['summary']['recommendations'])}")
    
    return diagnostics


if __name__ == "__main__":
    demo_model_diagnostics()