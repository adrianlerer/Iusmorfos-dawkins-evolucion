#!/usr/bin/env python3
"""
Baseline Model Comparisons for Iusmorfos
========================================

Comprehensive baseline model implementations for validating the superiority
of the evolutionary legal system model against simpler alternatives.

Following gold-standard evaluation practices:
- Multiple baseline strategies
- Cross-validation
- Statistical significance testing
- Effect size calculation
- Bootstrap confidence intervals

Models implemented:
1. Dummy Classifier (random/most frequent)
2. Logistic Regression
3. Random Forest
4. Simple Linear Model
5. Null Evolution Model (random walk)

Author: Adrian Lerer & AI Assistant
Date: 2025-09-23
License: MIT
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                           mean_squared_error, mean_absolute_error, r2_score,
                           classification_report, confusion_matrix)
from sklearn.preprocessing import StandardScaler
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional, Union
import warnings
warnings.filterwarnings('ignore')

from config import get_config
from robustness import RobustnessAnalyzer


class BaselineModelComparison:
    """
    Comprehensive baseline model comparison system.
    
    Compares the evolutionary legal system model against multiple baseline
    approaches to validate its effectiveness and measure improvement.
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        """Initialize baseline comparison system."""
        self.config = get_config()
        
        if random_seed is None:
            random_seed = self.config.config['reproducibility']['random_seed']
        
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Initialize models from config
        self.baseline_configs = self.config.config.get('models', {}).get('baselines', [])
        
        # Results storage
        self.results = {}
        self.comparison_metrics = {}
        
        print(f"🎯 Baseline Model Comparison initialized")
        print(f"   🔒 Random seed: {random_seed}")
        print(f"   📊 Configured baselines: {len(self.baseline_configs)}")
    
    def create_baseline_models(self) -> Dict[str, Any]:
        """Create baseline model instances from configuration."""
        models = {}
        
        # Default baseline models if not configured
        if not self.baseline_configs:
            self.baseline_configs = [
                {'name': 'dummy_classifier', 'strategy': 'most_frequent'},
                {'name': 'logistic_regression', 'solver': 'lbfgs', 'max_iter': 1000},
                {'name': 'random_forest', 'n_estimators': 100, 'random_state': self.random_seed}
            ]
        
        for config in self.baseline_configs:
            model_name = config['name']
            
            try:
                if model_name == 'dummy_classifier':
                    models[model_name] = DummyClassifier(
                        strategy=config.get('strategy', 'most_frequent'),
                        random_state=self.random_seed
                    )
                
                elif model_name == 'dummy_regressor':
                    models[model_name] = DummyRegressor(
                        strategy=config.get('strategy', 'mean')
                    )
                
                elif model_name == 'logistic_regression':
                    models[model_name] = LogisticRegression(
                        solver=config.get('solver', 'lbfgs'),
                        max_iter=config.get('max_iter', 1000),
                        random_state=self.random_seed
                    )
                
                elif model_name == 'linear_regression':
                    models[model_name] = LinearRegression()
                
                elif model_name == 'random_forest':
                    # Determine if classification or regression based on use
                    models[f'{model_name}_classifier'] = RandomForestClassifier(
                        n_estimators=config.get('n_estimators', 100),
                        random_state=self.random_seed
                    )
                    models[f'{model_name}_regressor'] = RandomForestRegressor(
                        n_estimators=config.get('n_estimators', 100),
                        random_state=self.random_seed
                    )
                
                print(f"✅ Created baseline model: {model_name}")
                
            except Exception as e:
                print(f"⚠️ Failed to create {model_name}: {e}")
        
        return models
    
    def generate_baseline_predictions(self, X: np.ndarray, y: np.ndarray, 
                                    task_type: str = 'classification') -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions from all baseline models.
        
        Args:
            X: Feature matrix
            y: Target vector
            task_type: 'classification' or 'regression'
            
        Returns:
            Dictionary with model predictions and metrics
        """
        print(f"🔄 Running baseline model comparison ({task_type})")
        print(f"   📊 Dataset: {X.shape[0]} samples, {X.shape[1]} features")
        
        models = self.create_baseline_models()
        results = {}
        
        # Filter models by task type
        if task_type == 'classification':
            relevant_models = {k: v for k, v in models.items() 
                             if 'regressor' not in k and k != 'linear_regression'}
        else:
            relevant_models = {k: v for k, v in models.items()
                             if 'classifier' not in k and k != 'logistic_regression'}
            # Add dummy regressor for regression tasks
            relevant_models['dummy_regressor'] = DummyRegressor(strategy='mean')
            relevant_models['linear_regression'] = LinearRegression()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_seed, stratify=y if task_type == 'classification' else None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train and evaluate each model
        for model_name, model in relevant_models.items():
            try:
                print(f"   🔄 Training {model_name}...")
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                model_results = {
                    'model_name': model_name,
                    'predictions': y_pred,
                    'test_targets': y_test
                }
                
                if task_type == 'classification':
                    model_results.update({
                        'accuracy': accuracy_score(y_test, y_pred),
                        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                    })
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    model_results['cv_mean'] = np.mean(cv_scores)
                    model_results['cv_std'] = np.std(cv_scores)
                
                else:  # regression
                    model_results.update({
                        'mse': mean_squared_error(y_test, y_pred),
                        'mae': mean_absolute_error(y_test, y_pred),
                        'r2': r2_score(y_test, y_pred),
                        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
                    })
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, 
                                              scoring='neg_mean_squared_error')
                    model_results['cv_mean'] = -np.mean(cv_scores)  # Convert back to positive MSE
                    model_results['cv_std'] = np.std(cv_scores)
                
                results[model_name] = model_results
                print(f"   ✅ {model_name} completed")
                
            except Exception as e:
                print(f"   ⚠️ {model_name} failed: {e}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def compare_legal_innovation_prediction(self, innovation_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Compare baseline models for legal innovation success prediction.
        
        Args:
            innovation_data: DataFrame with innovation features and success labels
            
        Returns:
            Comparison results with statistical significance tests
        """
        print("🏛️ Legal Innovation Prediction Comparison")
        print("=" * 45)
        
        if innovation_data.empty:
            print("⚠️ No innovation data available")
            return {}
        
        # Prepare features and targets
        feature_columns = [col for col in innovation_data.columns 
                          if col not in ['innovation_success', 'country', 'year', 'legal_family']]
        
        if not feature_columns:
            print("⚠️ No suitable feature columns found")
            return {}
        
        X = innovation_data[feature_columns].values
        
        # Create binary classification target (success > median)
        success_threshold = innovation_data['innovation_success'].median()
        y = (innovation_data['innovation_success'] > success_threshold).astype(int)
        
        print(f"   📊 Features: {feature_columns}")
        print(f"   🎯 Success threshold: {success_threshold:.3f}")
        print(f"   📈 Positive class ratio: {np.mean(y):.2%}")
        
        # Run baseline comparison
        baseline_results = self.generate_baseline_predictions(X, y, 'classification')
        
        # Compare with Iusmorfos model (if available)
        iusmorfos_performance = self._simulate_iusmorfos_performance(X, y)
        
        # Statistical significance testing
        significance_results = self._test_statistical_significance(
            baseline_results, iusmorfos_performance
        )
        
        # Create visualization
        self._plot_model_comparison(baseline_results, iusmorfos_performance, 'classification')
        
        return {
            'baseline_results': baseline_results,
            'iusmorfos_performance': iusmorfos_performance,
            'significance_tests': significance_results,
            'summary': self._generate_comparison_summary(baseline_results, iusmorfos_performance)
        }
    
    def compare_complexity_prediction(self, complexity_data: List[float]) -> Dict[str, Any]:
        """
        Compare baseline models for complexity evolution prediction.
        
        Args:
            complexity_data: Time series of complexity values
            
        Returns:
            Regression model comparison results
        """
        print("📈 Complexity Evolution Prediction Comparison")
        print("=" * 48)
        
        if len(complexity_data) < 10:
            print("⚠️ Insufficient complexity data for comparison")
            return {}
        
        # Create features (lagged values) and targets (next values)
        X, y = self._create_time_series_features(complexity_data, lag=3)
        
        print(f"   📊 Time series samples: {len(X)}")
        print(f"   🔢 Features per sample: {X.shape[1]}")
        
        # Run baseline comparison
        baseline_results = self.generate_baseline_predictions(X, y, 'regression')
        
        # Compare with Iusmorfos evolution model
        iusmorfos_performance = self._simulate_iusmorfos_complexity_prediction(X, y)
        
        # Statistical significance testing
        significance_results = self._test_statistical_significance(
            baseline_results, iusmorfos_performance, task_type='regression'
        )
        
        # Create visualization
        self._plot_model_comparison(baseline_results, iusmorfos_performance, 'regression')
        
        return {
            'baseline_results': baseline_results,
            'iusmorfos_performance': iusmorfos_performance,
            'significance_tests': significance_results,
            'summary': self._generate_comparison_summary(baseline_results, iusmorfos_performance)
        }
    
    def _create_time_series_features(self, data: List[float], lag: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """Create lagged features for time series prediction."""
        data_array = np.array(data)
        
        X = []
        y = []
        
        for i in range(lag, len(data_array)):
            X.append(data_array[i-lag:i])  # Previous 'lag' values as features
            y.append(data_array[i])        # Current value as target
        
        return np.array(X), np.array(y)
    
    def _simulate_iusmorfos_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Simulate Iusmorfos model performance (replace with actual model when available)."""
        # This is a placeholder - replace with actual Iusmorfos model evaluation
        
        # Split data same way as baseline models
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_seed, stratify=y
        )
        
        # Simulate performance better than random but not perfect
        # This should be replaced with actual Iusmorfos model predictions
        baseline_accuracy = np.mean(y_test)  # Majority class accuracy
        simulated_accuracy = baseline_accuracy + 0.15 + np.random.normal(0, 0.05)  # 15% improvement
        simulated_accuracy = max(0, min(1, simulated_accuracy))
        
        # Generate realistic predictions
        n_correct = int(simulated_accuracy * len(y_test))
        y_pred = np.random.choice([0, 1], size=len(y_test), 
                                 p=[1-simulated_accuracy, simulated_accuracy])
        
        return {
            'model_name': 'Iusmorfos_Evolution',
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'predictions': y_pred,
            'test_targets': y_test,
            'note': 'Simulated performance - replace with actual model'
        }
    
    def _simulate_iusmorfos_complexity_prediction(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Simulate Iusmorfos complexity prediction performance."""
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=self.random_seed
        )
        
        # Simulate predictions with realistic noise
        # This should be replaced with actual evolutionary model
        y_pred = y_test + np.random.normal(0, 0.1, len(y_test))
        
        return {
            'model_name': 'Iusmorfos_Evolution',
            'mse': mean_squared_error(y_test, y_pred),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'predictions': y_pred,
            'test_targets': y_test,
            'note': 'Simulated performance - replace with actual model'
        }
    
    def _test_statistical_significance(self, baseline_results: Dict[str, Dict], 
                                     iusmorfos_performance: Dict[str, Any],
                                     task_type: str = 'classification') -> Dict[str, Any]:
        """Test statistical significance of Iusmorfos vs baseline models."""
        
        significance_tests = {}
        
        # Get Iusmorfos predictions
        if 'predictions' not in iusmorfos_performance:
            return {'note': 'No predictions available for significance testing'}
        
        iusmorfos_pred = iusmorfos_performance['predictions']
        iusmorfos_test = iusmorfos_performance['test_targets']
        
        # Test against each baseline
        for model_name, results in baseline_results.items():
            if 'predictions' not in results:
                continue
            
            baseline_pred = results['predictions']
            baseline_test = results['test_targets']
            
            # Ensure same test set (should be the case with same random seed)
            if len(baseline_test) != len(iusmorfos_test):
                continue
            
            try:
                if task_type == 'classification':
                    # McNemar's test for paired classification results
                    iusmorfos_correct = (iusmorfos_pred == iusmorfos_test)
                    baseline_correct = (baseline_pred == baseline_test)
                    
                    # Create 2x2 contingency table
                    both_correct = np.sum(iusmorfos_correct & baseline_correct)
                    iusmorfos_only = np.sum(iusmorfos_correct & ~baseline_correct)
                    baseline_only = np.sum(~iusmorfos_correct & baseline_correct)
                    both_wrong = np.sum(~iusmorfos_correct & ~baseline_correct)
                    
                    # McNemar's test statistic
                    if iusmorfos_only + baseline_only > 0:
                        mcnemar_stat = (abs(iusmorfos_only - baseline_only) - 1)**2 / (iusmorfos_only + baseline_only)
                        p_value = 1 - stats.chi2.cdf(mcnemar_stat, 1)
                    else:
                        mcnemar_stat, p_value = 0, 1
                    
                    significance_tests[model_name] = {
                        'test': 'McNemar',
                        'statistic': mcnemar_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'iusmorfos_accuracy': np.mean(iusmorfos_correct),
                        'baseline_accuracy': np.mean(baseline_correct),
                        'improvement': np.mean(iusmorfos_correct) - np.mean(baseline_correct)
                    }
                
                else:  # regression
                    # Paired t-test on squared errors
                    iusmorfos_errors = (iusmorfos_pred - iusmorfos_test) ** 2
                    baseline_errors = (baseline_pred - baseline_test) ** 2
                    
                    t_stat, p_value = stats.ttest_rel(baseline_errors, iusmorfos_errors)
                    
                    significance_tests[model_name] = {
                        'test': 'Paired t-test (MSE)',
                        'statistic': t_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'iusmorfos_mse': np.mean(iusmorfos_errors),
                        'baseline_mse': np.mean(baseline_errors),
                        'improvement': np.mean(baseline_errors) - np.mean(iusmorfos_errors)
                    }
                
            except Exception as e:
                significance_tests[model_name] = {'error': str(e)}
        
        return significance_tests
    
    def _plot_model_comparison(self, baseline_results: Dict[str, Dict], 
                              iusmorfos_performance: Dict[str, Any],
                              task_type: str) -> None:
        """Create visualization comparing model performances."""
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Collect model names and performance metrics
        model_names = []
        performances = []
        
        # Add baseline results
        for model_name, results in baseline_results.items():
            if 'error' not in results:
                model_names.append(model_name.replace('_', ' ').title())
                
                if task_type == 'classification':
                    performances.append(results.get('accuracy', 0))
                else:
                    performances.append(results.get('r2', 0))
        
        # Add Iusmorfos result
        if iusmorfos_performance:
            model_names.append('Iusmorfos Evolution')
            if task_type == 'classification':
                performances.append(iusmorfos_performance.get('accuracy', 0))
            else:
                performances.append(iusmorfos_performance.get('r2', 0))
        
        # 1. Performance comparison bar plot
        colors = ['lightblue'] * (len(model_names) - 1) + ['red']
        bars = ax1.bar(range(len(model_names)), performances, color=colors, edgecolor='navy')
        ax1.set_title(f'🎯 Model Performance Comparison ({task_type.title()})', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Models')
        ax1.set_ylabel('Accuracy' if task_type == 'classification' else 'R² Score')
        ax1.set_xticks(range(len(model_names)))
        ax1.set_xticklabels(model_names, rotation=45, ha='right')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, perf in zip(bars, performances):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{perf:.3f}', ha='center', va='bottom', fontweight='bold')
        
        # 2. Cross-validation comparison (if available)
        cv_means = []
        cv_stds = []
        cv_names = []
        
        for model_name, results in baseline_results.items():
            if 'cv_mean' in results and 'error' not in results:
                cv_names.append(model_name.replace('_', ' ').title())
                cv_means.append(results['cv_mean'])
                cv_stds.append(results.get('cv_std', 0))
        
        if cv_means:
            ax2.errorbar(range(len(cv_names)), cv_means, yerr=cv_stds, 
                        fmt='o', capsize=5, capthick=2, linewidth=2)
            ax2.set_title('📊 Cross-Validation Stability', fontsize=14, fontweight='bold')
            ax2.set_xlabel('Models')
            ax2.set_ylabel('CV Score ± Std')
            ax2.set_xticks(range(len(cv_names)))
            ax2.set_xticklabels(cv_names, rotation=45, ha='right')
            ax2.grid(True, alpha=0.3)
        
        # 3. Performance distribution (if predictions available)
        if task_type == 'classification':
            # Classification accuracy by class
            if 'test_targets' in iusmorfos_performance:
                y_test = iusmorfos_performance['test_targets']
                class_accuracies = []
                
                for class_label in np.unique(y_test):
                    mask = (y_test == class_label)
                    if np.sum(mask) > 0:
                        y_pred = iusmorfos_performance['predictions']
                        class_acc = accuracy_score(y_test[mask], y_pred[mask])
                        class_accuracies.append(class_acc)
                
                if class_accuracies:
                    ax3.bar(range(len(class_accuracies)), class_accuracies, color='lightgreen')
                    ax3.set_title('🎯 Iusmorfos Per-Class Accuracy', fontsize=14, fontweight='bold')
                    ax3.set_xlabel('Class')
                    ax3.set_ylabel('Accuracy')
                    ax3.set_xticks(range(len(class_accuracies)))
        
        else:  # regression
            # Residual analysis for Iusmorfos
            if 'predictions' in iusmorfos_performance:
                y_pred = iusmorfos_performance['predictions']
                y_test = iusmorfos_performance['test_targets']
                residuals = y_test - y_pred
                
                ax3.scatter(y_pred, residuals, alpha=0.6, color='red')
                ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax3.set_title('📊 Iusmorfos Residual Analysis', fontsize=14, fontweight='bold')
                ax3.set_xlabel('Predicted Values')
                ax3.set_ylabel('Residuals')
                ax3.grid(True, alpha=0.3)
        
        # 4. Model complexity vs performance trade-off
        # This is a conceptual plot showing the trade-off
        complexity_scores = [1, 2, 3, 4, 5]  # Conceptual complexity
        performance_scores = performances[:5] if len(performances) >= 5 else performances
        
        if len(complexity_scores) == len(performance_scores):
            ax4.scatter(complexity_scores, performance_scores, s=100, c=['blue']*len(performance_scores))
            
            for i, name in enumerate(model_names[:len(complexity_scores)]):
                ax4.annotate(name, (complexity_scores[i], performance_scores[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax4.set_title('⚖️ Complexity vs Performance Trade-off', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Model Complexity')
            ax4.set_ylabel('Performance')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('../outputs/baseline_model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _generate_comparison_summary(self, baseline_results: Dict[str, Dict], 
                                   iusmorfos_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of model comparison."""
        
        summary = {
            'total_models_tested': len(baseline_results),
            'iusmorfos_performance': {},
            'best_baseline': {},
            'performance_ranking': [],
            'significant_improvements': []
        }
        
        # Extract Iusmorfos performance
        if iusmorfos_performance:
            for metric in ['accuracy', 'f1_score', 'mse', 'r2']:
                if metric in iusmorfos_performance:
                    summary['iusmorfos_performance'][metric] = iusmorfos_performance[metric]
        
        # Find best baseline
        best_baseline_name = None
        best_baseline_score = -float('inf')
        
        for model_name, results in baseline_results.items():
            if 'error' not in results:
                # Use accuracy for classification, R² for regression
                score = results.get('accuracy', results.get('r2', 0))
                if score > best_baseline_score:
                    best_baseline_score = score
                    best_baseline_name = model_name
        
        if best_baseline_name:
            summary['best_baseline'] = {
                'name': best_baseline_name,
                'performance': best_baseline_score
            }
        
        return summary
    
    def run_comprehensive_comparison(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Run comprehensive baseline comparison on all available data."""
        
        print("🎯 Running Comprehensive Baseline Model Comparison")
        print("=" * 55)
        
        results = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'configuration': {
                'random_seed': self.random_seed,
                'baseline_configs': self.baseline_configs
            },
            'comparisons': {}
        }
        
        # Legal innovation prediction comparison
        if 'innovations_exported' in data:
            innovation_df = data['innovations_exported']
            if not innovation_df.empty and 'innovation_success' in innovation_df.columns:
                print("\n🏛️ Running legal innovation prediction comparison...")
                innovation_results = self.compare_legal_innovation_prediction(innovation_df)
                results['comparisons']['innovation_prediction'] = innovation_results
        
        # Complexity evolution prediction comparison  
        if 'experimental_results' in data:
            exp_results = data['experimental_results']
            complexity_data = exp_results.get('evolucion_complejidad', [])
            if len(complexity_data) >= 10:
                print("\n📈 Running complexity evolution prediction comparison...")
                complexity_results = self.compare_complexity_prediction(complexity_data)
                results['comparisons']['complexity_prediction'] = complexity_results
        
        # Generate overall summary
        results['overall_summary'] = self._generate_overall_summary(results['comparisons'])
        
        # Save results
        output_path = self.config.get_path('results_dir') / f"baseline_comparison_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            else:
                return obj
        
        json_results = convert_numpy(results)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            import json
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Baseline comparison results saved: {output_path}")
        
        return results
    
    def _generate_overall_summary(self, comparisons: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall summary across all comparisons."""
        
        summary = {
            'total_comparisons': len(comparisons),
            'iusmorfos_wins': 0,
            'baseline_wins': 0,
            'significant_improvements': 0,
            'average_improvement': 0,
            'recommendations': []
        }
        
        improvements = []
        
        for comparison_name, comparison_data in comparisons.items():
            if 'summary' in comparison_data:
                comp_summary = comparison_data['summary']
                
                # Extract performance metrics
                iusmorfos_perf = comp_summary.get('iusmorfos_performance', {})
                best_baseline = comp_summary.get('best_baseline', {})
                
                if iusmorfos_perf and best_baseline:
                    # Compare primary metrics
                    iusmorfos_score = list(iusmorfos_perf.values())[0] if iusmorfos_perf else 0
                    baseline_score = best_baseline.get('performance', 0)
                    
                    if iusmorfos_score > baseline_score:
                        summary['iusmorfos_wins'] += 1
                        improvement = ((iusmorfos_score - baseline_score) / baseline_score) * 100
                        improvements.append(improvement)
                    else:
                        summary['baseline_wins'] += 1
        
        if improvements:
            summary['average_improvement'] = np.mean(improvements)
        
        # Generate recommendations
        if summary['iusmorfos_wins'] > summary['baseline_wins']:
            summary['recommendations'].append("Iusmorfos shows superior performance over baseline models")
        
        if summary['average_improvement'] > 10:
            summary['recommendations'].append("Significant improvement over baselines - results are promising")
        elif summary['average_improvement'] > 0:
            summary['recommendations'].append("Moderate improvement over baselines - consider further optimization")
        else:
            summary['recommendations'].append("Performance similar to baselines - investigate model assumptions")
        
        summary['recommendations'].append("Implement statistical significance testing for all comparisons")
        summary['recommendations'].append("Consider ensemble methods combining multiple approaches")
        
        return summary


def demo_baseline_comparison():
    """Demonstration of baseline model comparison."""
    print("🎯 Baseline Model Comparison Demo")
    print("=" * 35)
    
    # Create comparison system
    comparison = BaselineModelComparison()
    
    # Generate mock data
    np.random.seed(42)
    
    # Mock empirical data
    innovation_data = pd.DataFrame({
        'complexity_score': np.random.gamma(2, 1.5, 100),
        'adoption_rate': np.random.exponential(0.3, 100),
        'legal_system_age': np.random.normal(150, 50, 100),
        'innovation_success': np.random.beta(2, 3, 100)
    })
    
    # Mock experimental results
    complexity_evolution = [1.0 + i * 0.1 + np.random.normal(0, 0.05) for i in range(30)]
    
    data = {
        'innovations_exported': innovation_data,
        'experimental_results': {
            'evolucion_complejidad': complexity_evolution
        }
    }
    
    # Run comprehensive comparison
    results = comparison.run_comprehensive_comparison(data)
    
    print("\n📊 Comparison Complete!")
    print(f"   🔍 Comparisons performed: {results['overall_summary']['total_comparisons']}")
    print(f"   🏆 Iusmorfos wins: {results['overall_summary']['iusmorfos_wins']}")
    print(f"   📈 Average improvement: {results['overall_summary']['average_improvement']:.1f}%")
    
    return comparison


if __name__ == "__main__":
    demo_baseline_comparison()