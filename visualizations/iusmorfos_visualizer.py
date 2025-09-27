#!/usr/bin/env python3
"""
Iusmorfos Visualizer - Framework v4.0
=====================================

Suite de visualización para el framework Iusmorfos con capacidades
de análisis visual de trayectorias institucionales, cuencas de atracción,
y dinámicas evolutivas.

Author: Adrian Lerer & Claude (AI Assistant)
Version: 4.0
Date: September 2024

Features:
- 9D IusSpace trajectory visualization
- Attractor basin mapping
- Competitive evolution dynamics
- Validation metrics dashboards
- Real-time prediction interfaces
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple, Optional, Any
import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IusmorfosVisualizer:
    """
    Main visualization class for Iusmorfos framework.
    """
    
    def __init__(self, style: str = 'academic'):
        """
        Initialize visualizer with academic styling.
        
        Args:
            style: Visualization style ('academic', 'presentation', 'publication')
        """
        self.style = style
        self.setup_styling()
        
        # Dimension names for 9D IusSpace
        self.dimension_names = [
            'Federal Structure',
            'Judicial Independence', 
            'Democratic Participation',
            'Individual Rights',
            'Separation of Powers',
            'Constitutional Stability',
            'Rule of Law',
            'Social Rights',
            'Checks & Balances'
        ]
        
        # Color schemes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728',
            'info': '#9467bd',
            'gradient': ['#440154', '#31688e', '#35b779', '#fde725']
        }
        
        logger.info(f"Iusmorfos Visualizer initialized with {style} style")
        
    def setup_styling(self):
        """Set up matplotlib and seaborn styling."""
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        # Academic publication style
        if self.style == 'academic':
            plt.rcParams.update({
                'font.size': 10,
                'axes.titlesize': 12,
                'axes.labelsize': 10,
                'xtick.labelsize': 9,
                'ytick.labelsize': 9,
                'legend.fontsize': 9,
                'figure.titlesize': 14,
                'font.family': 'serif'
            })
            
    def plot_9d_trajectory_pca(self, 
                              trajectory: np.ndarray,
                              title: str = "9D Institutional Trajectory (PCA Projection)",
                              save_path: str = None) -> plt.Figure:
        """
        Plot 9D trajectory using PCA projection to 3D.
        
        Args:
            trajectory: Trajectory array (time_steps x 9_dimensions)
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        from sklearn.decomposition import PCA
        
        # Apply PCA to reduce to 3D
        pca = PCA(n_components=3)
        trajectory_3d = pca.fit_transform(trajectory)
        
        # Create 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot trajectory
        ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], 
               'b-', linewidth=2, alpha=0.7, label='Trajectory')
        
        # Mark start and end points
        ax.scatter(trajectory_3d[0, 0], trajectory_3d[0, 1], trajectory_3d[0, 2],
                  c='green', s=100, marker='o', label='Start')
        ax.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2],
                  c='red', s=100, marker='s', label='End')
        
        # Add time-based coloring
        time_colors = np.linspace(0, 1, len(trajectory_3d))
        scatter = ax.scatter(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2],
                           c=time_colors, cmap='viridis', s=30, alpha=0.6)
        
        # Labels and formatting
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        ax.set_zlabel(f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)')
        ax.set_title(title)
        ax.legend()
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
        cbar.set_label('Time Progress')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_dimensional_evolution(self,
                                 trajectory: np.ndarray,
                                 time_points: np.ndarray = None,
                                 title: str = "Institutional Dimensions Evolution",
                                 save_path: str = None) -> plt.Figure:
        """
        Plot evolution of each dimension over time.
        
        Args:
            trajectory: Trajectory array (time_steps x 9_dimensions)
            time_points: Time points array (optional)
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if time_points is None:
            time_points = np.arange(len(trajectory))
            
        # Create subplots
        fig, axes = plt.subplots(3, 3, figsize=(15, 12))
        axes = axes.flatten()
        
        for i, (ax, dim_name) in enumerate(zip(axes, self.dimension_names)):
            if i < trajectory.shape[1]:
                ax.plot(time_points, trajectory[:, i], 
                       linewidth=2, color=self.colors['primary'])
                ax.set_title(dim_name, fontsize=10)
                ax.set_ylabel('Value')
                ax.grid(True, alpha=0.3)
                ax.set_ylim(-1.1, 1.1)
                
                # Add horizontal lines at key values
                ax.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                ax.axhline(y=0.5, color='green', linestyle=':', alpha=0.5)
                ax.axhline(y=-0.5, color='red', linestyle=':', alpha=0.5)
                
        # Set x-axis labels for bottom row
        for ax in axes[6:9]:
            ax.set_xlabel('Time')
            
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_attractor_basins_2d(self,
                                basins: Dict[str, Any],
                                projection_dims: Tuple[int, int] = (0, 1),
                                title: str = "Attractor Basins (2D Projection)",
                                save_path: str = None) -> plt.Figure:
        """
        Plot attractor basins in 2D projection.
        
        Args:
            basins: Dictionary of AttractorBasin objects
            projection_dims: Which dimensions to project (dim1, dim2)
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(basins)))
        
        for i, (basin_id, basin_data) in enumerate(basins.items()):
            if hasattr(basin_data, 'attractor_point'):
                attractor = basin_data.attractor_point
                boundaries = basin_data.basin_boundaries
            else:
                # Handle dictionary format
                attractor = np.array(basin_data['attractor_point'])
                boundaries = basin_data['basin_boundaries']
                
            # Plot attractor point
            ax.scatter(attractor[projection_dims[0]], attractor[projection_dims[1]],
                      c=[colors[i]], s=200, marker='*', 
                      label=f'{basin_id} (attractor)', edgecolors='black')
            
            # Plot basin boundary (approximate as ellipse)
            if isinstance(boundaries, dict) and len(boundaries) >= 2:
                dim_names = list(boundaries.keys())
                if len(dim_names) > max(projection_dims):
                    dim1_bounds = boundaries[dim_names[projection_dims[0]]]
                    dim2_bounds = boundaries[dim_names[projection_dims[1]]]
                    
                    # Draw approximate basin boundary
                    center_x = (dim1_bounds[0] + dim1_bounds[1]) / 2
                    center_y = (dim2_bounds[0] + dim2_bounds[1]) / 2
                    width = dim1_bounds[1] - dim1_bounds[0]
                    height = dim2_bounds[1] - dim2_bounds[0]
                    
                    ellipse = plt.Circle((center_x, center_y), 
                                       max(width, height) / 4,
                                       color=colors[i], alpha=0.3)
                    ax.add_patch(ellipse)
                    
        # Formatting
        ax.set_xlabel(self.dimension_names[projection_dims[0]])
        ax.set_ylabel(self.dimension_names[projection_dims[1]])
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_competitive_evolution(self,
                                  evolution_results: Dict[str, Any],
                                  title: str = "Competitive Evolution Dynamics",
                                  save_path: str = None) -> plt.Figure:
        """
        Plot competitive evolution results.
        
        Args:
            evolution_results: Results from competitive arena simulation
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        # Extract data
        generation_stats = evolution_results.get('generation_stats', [])
        if not generation_stats:
            logger.warning("No generation stats available for plotting")
            return None
            
        generations = [stat['generation'] for stat in generation_stats]
        num_species = [stat['num_species'] for stat in generation_stats]
        mean_fitness = [stat['mean_fitness'] for stat in generation_stats]
        diversity = [stat['diversity'] for stat in generation_stats]
        
        # Create subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        # Species count over time
        ax1.plot(generations, num_species, 'b-', linewidth=2)
        ax1.set_title('Number of Species')
        ax1.set_xlabel('Generation')
        ax1.set_ylabel('Species Count')
        ax1.grid(True, alpha=0.3)
        
        # Mean fitness over time
        ax2.plot(generations, mean_fitness, 'g-', linewidth=2)
        ax2.set_title('Mean Fitness')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness')
        ax2.grid(True, alpha=0.3)
        
        # Diversity over time
        ax3.plot(generations, diversity, 'r-', linewidth=2)
        ax3.set_title('Genetic Diversity')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Diversity')
        ax3.grid(True, alpha=0.3)
        
        # Final population composition
        final_pop = evolution_results.get('final_population', {})
        if 'species_population' in final_pop:
            species_names = list(final_pop['species_population'].keys())
            fitness_values = [final_pop['species_population'][name]['fitness'] 
                            for name in species_names]
            
            if species_names:
                bars = ax4.bar(range(len(species_names)), fitness_values)
                ax4.set_title('Final Population Fitness')
                ax4.set_xlabel('Species')
                ax4.set_ylabel('Fitness')
                ax4.set_xticks(range(len(species_names)))
                ax4.set_xticklabels([name[:15] + '...' if len(name) > 15 else name 
                                   for name in species_names], rotation=45)
                
                # Color bars by fitness
                colors = plt.cm.viridis(np.linspace(0, 1, len(bars)))
                for bar, color in zip(bars, colors):
                    bar.set_color(color)
                    
        plt.suptitle(title, fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def plot_validation_metrics(self,
                              validation_report: Dict[str, Any],
                              title: str = "Validation Metrics Dashboard",
                              save_path: str = None) -> plt.Figure:
        """
        Plot validation metrics and accuracy trends.
        
        Args:
            validation_report: Validation report from ValidationTracker
            title: Plot title
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if 'detailed_performance' not in validation_report:
            logger.warning("No detailed performance data available")
            return None
            
        performance = validation_report['detailed_performance']
        
        # Create dashboard
        fig = plt.figure(figsize=(16, 12))
        
        # Overall accuracy metrics
        ax1 = plt.subplot(2, 3, 1)
        if 'overall_performance' in performance:
            overall = performance['overall_performance']
            metrics = ['mean_accuracy', 'median_accuracy', 'min_accuracy', 'max_accuracy']
            values = [overall.get(metric, 0) for metric in metrics]
            
            bars = ax1.bar(metrics, values, color=self.colors['primary'])
            ax1.set_title('Overall Accuracy Metrics')
            ax1.set_ylabel('Accuracy')
            ax1.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{value:.2f}', ha='center', va='bottom')
                        
        # Per-metric performance
        ax2 = plt.subplot(2, 3, 2)
        if 'per_metric_performance' in performance:
            metric_perf = performance['per_metric_performance']
            if metric_perf:
                metric_names = list(metric_perf.keys())
                accuracies = [metric_perf[m]['mean_accuracy'] for m in metric_names]
                
                bars = ax2.bar(range(len(metric_names)), accuracies, 
                             color=self.colors['secondary'])
                ax2.set_title('Per-Metric Accuracy')
                ax2.set_ylabel('Mean Accuracy')
                ax2.set_xticks(range(len(metric_names)))
                ax2.set_xticklabels([name[:10] + '...' if len(name) > 10 else name 
                                   for name in metric_names], rotation=45)
                ax2.set_ylim(0, 1)
                
        # Regional performance
        ax3 = plt.subplot(2, 3, 3)
        if 'regional_performance' in performance:
            regional = performance['regional_performance']
            if regional and any(isinstance(v, dict) for v in regional.values()):
                regions = [k for k, v in regional.items() if isinstance(v, dict)]
                reg_accuracies = [regional[r]['mean_accuracy'] for r in regions]
                
                bars = ax3.bar(regions, reg_accuracies, color=self.colors['success'])
                ax3.set_title('Regional Performance')
                ax3.set_ylabel('Mean Accuracy')
                ax3.set_ylim(0, 1)
                plt.setp(ax3.get_xticklabels(), rotation=45)
                
        # Statistical significance
        ax4 = plt.subplot(2, 3, 4)
        if 'statistical_significance' in performance:
            sig_test = performance['statistical_significance']
            
            # P-value visualization
            p_value = sig_test.get('t_test', {}).get('p_value', 1.0)
            significance_threshold = 0.0001
            
            # Bar showing p-value vs threshold
            ax4.bar(['P-value', 'Threshold'], [p_value, significance_threshold],
                   color=['red' if p_value > significance_threshold else 'green', 'blue'])
            ax4.set_yscale('log')
            ax4.set_title('Statistical Significance')
            ax4.set_ylabel('P-value (log scale)')
            
            # Add significance indicator
            if p_value < significance_threshold:
                ax4.text(0.5, 0.8, 'SIGNIFICANT', transform=ax4.transAxes,
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
                        
        # Reproducibility metrics
        ax5 = plt.subplot(2, 3, 5)
        if 'reproducibility_metrics' in performance:
            reprod = performance['reproducibility_metrics']
            
            metrics = []
            values = []
            
            if 'inter_coder_reliability' in reprod:
                metrics.append('Inter-coder\nReliability')
                values.append(reprod['inter_coder_reliability'].get('mean_kappa', 0))
                
            if 'prediction_calibration' in reprod:
                metrics.append('Prediction\nCalibration')
                values.append(reprod['prediction_calibration'].get('calibration_rate', 0))
                
            if 'temporal_stability' in reprod:
                metrics.append('Temporal\nStability')
                # Convert stability to positive metric (1 - variance)
                variance = reprod['temporal_stability'].get('accuracy_variance', 0.1)
                values.append(max(0, 1 - variance * 10))
                
            if metrics:
                bars = ax5.bar(metrics, values, color=self.colors['info'])
                ax5.set_title('Reproducibility Metrics')
                ax5.set_ylabel('Score')
                ax5.set_ylim(0, 1)
                
        # World-class criteria assessment
        ax6 = plt.subplot(2, 3, 6)
        if 'world_class_criteria_assessment' in validation_report:
            criteria = validation_report['world_class_criteria_assessment']
            
            criterion_names = list(criteria.keys())
            criterion_values = [float(criteria[name]) for name in criterion_names]
            
            # Create pie chart of achieved criteria
            achieved = sum(criterion_values)
            total = len(criterion_values)
            
            labels = ['Achieved', 'Not Achieved']
            sizes = [achieved, total - achieved]
            colors = ['green', 'red']
            
            ax6.pie(sizes, labels=labels, colors=colors, autopct='%1.0f%%', startangle=90)
            ax6.set_title(f'World-Class Criteria\n({achieved}/{total} achieved)')
            
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
        
    def create_interactive_trajectory_plot(self,
                                         trajectory: np.ndarray,
                                         time_points: np.ndarray = None,
                                         title: str = "Interactive 9D Trajectory") -> go.Figure:
        """
        Create interactive 3D trajectory plot using Plotly.
        
        Args:
            trajectory: Trajectory array (time_steps x 9_dimensions)
            time_points: Time points array
            title: Plot title
            
        Returns:
            Plotly figure
        """
        if time_points is None:
            time_points = np.arange(len(trajectory))
            
        # Use PCA for 3D projection
        from sklearn.decomposition import PCA
        pca = PCA(n_components=3)
        trajectory_3d = pca.fit_transform(trajectory)
        
        # Create interactive 3D plot
        fig = go.Figure()
        
        # Add trajectory line
        fig.add_trace(go.Scatter3d(
            x=trajectory_3d[:, 0],
            y=trajectory_3d[:, 1], 
            z=trajectory_3d[:, 2],
            mode='lines+markers',
            line=dict(color='blue', width=4),
            marker=dict(
                size=4,
                color=time_points,
                colorscale='Viridis',
                colorbar=dict(title="Time"),
                showscale=True
            ),
            name='Trajectory',
            hovertemplate='<b>Time: %{marker.color}</b><br>' +
                         'PC1: %{x:.3f}<br>' +
                         'PC2: %{y:.3f}<br>' +
                         'PC3: %{z:.3f}<br>' +
                         '<extra></extra>'
        ))
        
        # Mark start and end points
        fig.add_trace(go.Scatter3d(
            x=[trajectory_3d[0, 0]],
            y=[trajectory_3d[0, 1]],
            z=[trajectory_3d[0, 2]],
            mode='markers',
            marker=dict(size=12, color='green', symbol='circle'),
            name='Start',
            hovertemplate='<b>Start Point</b><extra></extra>'
        ))
        
        fig.add_trace(go.Scatter3d(
            x=[trajectory_3d[-1, 0]],
            y=[trajectory_3d[-1, 1]], 
            z=[trajectory_3d[-1, 2]],
            mode='markers',
            marker=dict(size=12, color='red', symbol='square'),
            name='End',
            hovertemplate='<b>End Point</b><extra></extra>'
        ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title=f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)',
                yaxis_title=f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)',
                zaxis_title=f'PC3 ({pca.explained_variance_ratio_[2]:.1%} variance)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            width=800,
            height=600
        )
        
        return fig
        
    def create_dashboard_summary(self, 
                               analysis_results: Dict[str, Any],
                               save_path: str = None) -> plt.Figure:
        """
        Create comprehensive dashboard summarizing all analysis results.
        
        Args:
            analysis_results: Complete analysis results
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure with dashboard
        """
        fig = plt.figure(figsize=(20, 14))
        
        # Title
        fig.suptitle('Iusmorfos Framework v4.0 - Comprehensive Analysis Dashboard', 
                    fontsize=18, fontweight='bold')
        
        # Analysis overview (top left)
        ax1 = plt.subplot(3, 4, 1)
        overview_data = analysis_results.get('analysis_metadata', {})
        
        ax1.text(0.1, 0.9, f"Country: {overview_data.get('country', 'N/A')}", 
                transform=ax1.transAxes, fontsize=12, fontweight='bold')
        ax1.text(0.1, 0.8, f"Period: {overview_data.get('analysis_period', 'N/A')}", 
                transform=ax1.transAxes, fontsize=10)
        ax1.text(0.1, 0.7, f"Framework: v{overview_data.get('framework_version', '4.0')}", 
                transform=ax1.transAxes, fontsize=10)
        ax1.text(0.1, 0.6, f"Timestamp: {overview_data.get('report_timestamp', 'N/A')[:10]}", 
                transform=ax1.transAxes, fontsize=10)
        
        ax1.set_title('Analysis Overview')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')
        
        # Key findings (top center-left)
        ax2 = plt.subplot(3, 4, 2)
        exec_summary = analysis_results.get('executive_summary', {})
        key_findings = exec_summary.get('key_findings', [])
        
        for i, finding in enumerate(key_findings[:4]):  # Show first 4 findings
            ax2.text(0.05, 0.9 - i*0.2, f"• {finding[:50]}{'...' if len(finding) > 50 else ''}", 
                    transform=ax2.transAxes, fontsize=9, wrap=True)
        
        ax2.set_title('Key Findings')
        ax2.axis('off')
        
        # Implementation probabilities (top center-right)
        ax3 = plt.subplot(3, 4, 3)
        trajectory_pred = analysis_results.get('reform_trajectory_prediction', {})
        impl_probs = trajectory_pred.get('implementation_probability_analysis', {})
        
        if impl_probs:
            dims = list(impl_probs.keys())[:5]  # Show top 5
            probs = [impl_probs[dim] for dim in dims]
            
            bars = ax3.barh(dims, probs, color=self.colors['primary'])
            ax3.set_title('Implementation Probabilities')
            ax3.set_xlabel('Probability')
            ax3.set_xlim(0, 1)
            
            # Add value labels
            for i, (bar, prob) in enumerate(zip(bars, probs)):
                ax3.text(prob + 0.02, i, f'{prob:.2f}', 
                        va='center', fontsize=8)
                        
        # Risk assessment (top right)
        ax4 = plt.subplot(3, 4, 4)
        risk_summary = analysis_results.get('risk_assessment_summary', {})
        top_risks = risk_summary.get('top_risks', [])
        
        if top_risks:
            risk_names = [risk['name'][:15] for risk in top_risks[:3]]
            risk_scores = [risk['risk_score'] for risk in top_risks[:3]]
            
            bars = ax4.bar(risk_names, risk_scores, color=self.colors['warning'])
            ax4.set_title('Top Risks')
            ax4.set_ylabel('Risk Score')
            plt.setp(ax4.get_xticklabels(), rotation=45)
            
        # Validation metrics summary (middle left)
        ax5 = plt.subplot(3, 4, 5)
        validation = analysis_results.get('validation_framework', {})
        
        if 'validation_report_summary' in validation:
            val_summary = validation['validation_report_summary']
            
            metrics = {
                'Accuracy': val_summary.get('overall_accuracy', 0),
                'World-Class': float(val_summary.get('world_class_reproducibility_achieved', False)),
                'Cases': min(1.0, val_summary.get('total_validated_cases', 0) / 10)  # Normalized
            }
            
            bars = ax5.bar(metrics.keys(), metrics.values(), 
                          color=[self.colors['success'], self.colors['info'], self.colors['secondary']])
            ax5.set_title('Validation Status')
            ax5.set_ylabel('Score/Status')
            ax5.set_ylim(0, 1)
            
            # Add value labels
            for bar, (name, value) in zip(bars, metrics.items()):
                if name == 'Cases':
                    label = f"{val_summary.get('total_validated_cases', 0)}"
                elif name == 'World-Class':
                    label = 'YES' if value == 1.0 else 'NO'
                else:
                    label = f'{value:.2f}'
                ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                        label, ha='center', va='bottom', fontsize=8)
                        
        # Competitive evolution summary (middle center-left)
        ax6 = plt.subplot(3, 4, 6)
        competition = analysis_results.get('competitive_evolution_analysis', {})
        
        if 'survival_analysis' in competition:
            survival = competition['survival_analysis']
            survivors = len(survival.get('survivors', []))
            extinctions = len(survival.get('extinctions', []))
            new_species = len(survival.get('new_species', []))
            
            categories = ['Survivors', 'Extinctions', 'New Species']
            counts = [survivors, extinctions, new_species]
            colors = [self.colors['success'], self.colors['warning'], self.colors['info']]
            
            bars = ax6.bar(categories, counts, color=colors)
            ax6.set_title('Evolution Summary')
            ax6.set_ylabel('Count')
            
            # Add value labels
            for bar, count in zip(bars, counts):
                ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(count), ha='center', va='bottom')
                        
        # Scenario comparison (middle center-right)
        ax7 = plt.subplot(3, 4, 7)
        scenario_analysis = trajectory_pred.get('scenario_analysis', {})
        
        if 'scenarios' in scenario_analysis:
            scenarios = scenario_analysis['scenarios']
            scenario_names = list(scenarios.keys())[:3]  # Show first 3 scenarios
            
            # Extract some metric for comparison (placeholder)
            scenario_scores = []
            for name in scenario_names:
                scenario_data = scenarios[name]
                prediction = scenario_data.get('prediction', {})
                if prediction.get('success', False):
                    stats = prediction.get('trajectory_stats', {})
                    score = 1.0 - stats.get('total_change', 1.0)  # Lower change = higher score
                    scenario_scores.append(max(0, score))
                else:
                    scenario_scores.append(0)
                    
            if scenario_scores:
                bars = ax7.bar([name[:8] for name in scenario_names], scenario_scores,
                             color=self.colors['primary'])
                ax7.set_title('Scenario Stability')
                ax7.set_ylabel('Stability Score')
                ax7.set_ylim(0, 1)
                plt.setp(ax7.get_xticklabels(), rotation=45)
                
        # Critical junctures timeline (middle right)
        ax8 = plt.subplot(3, 4, 8)
        critical_junctures = trajectory_pred.get('critical_junctures', [])
        
        if critical_junctures:
            # Extract timing information
            juncture_names = [cj['name'][:15] for cj in critical_junctures[:4]]
            juncture_probs = [cj['probability_threshold'] for cj in critical_junctures[:4]]
            
            bars = ax8.barh(juncture_names, juncture_probs, color=self.colors['warning'])
            ax8.set_title('Critical Junctures')
            ax8.set_xlabel('Probability')
            ax8.set_xlim(0, 1)
            
        # Policy recommendations (bottom section)
        ax9 = plt.subplot(3, 2, 5)
        policy_recs = analysis_results.get('policy_recommendations', [])
        
        if policy_recs:
            high_priority_recs = [rec for rec in policy_recs if rec.get('priority') == 'high']
            
            rec_text = "High Priority Recommendations:\n"
            for i, rec in enumerate(high_priority_recs[:3]):
                rec_text += f"{i+1}. {rec.get('title', 'N/A')}\n"
                
            ax9.text(0.05, 0.95, rec_text, transform=ax9.transAxes, 
                    fontsize=10, verticalalignment='top', wrap=True)
            
        ax9.set_title('Policy Recommendations')
        ax9.axis('off')
        
        # Methodology summary (bottom right)
        ax10 = plt.subplot(3, 2, 6)
        methodology = analysis_results.get('methodology_notes', {})
        
        method_text = "Framework Components:\n"
        if 'framework_components' in methodology:
            components = methodology['framework_components']
            for comp_name, comp_desc in list(components.items())[:4]:
                method_text += f"• {comp_name}: {comp_desc[:30]}...\n"
                
        method_text += f"\nStatistical Standards:\n"
        if 'statistical_standards' in methodology:
            standards = methodology['statistical_standards']
            method_text += f"• Significance: {standards.get('significance_threshold', 'p < 0.0001')}\n"
            method_text += f"• Min. samples: {standards.get('minimum_sample_size', 10)}\n"
            
        ax10.text(0.05, 0.95, method_text, transform=ax10.transAxes,
                 fontsize=9, verticalalignment='top')
        
        ax10.set_title('Methodology')
        ax10.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig

# Utility functions for quick visualization
def quick_trajectory_plot(trajectory: np.ndarray, 
                         title: str = "Institutional Trajectory") -> plt.Figure:
    """Quick 2D trajectory plot using first two dimensions."""
    visualizer = IusmorfosVisualizer()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2, alpha=0.7)
    ax.scatter(trajectory[0, 0], trajectory[0, 1], c='green', s=100, marker='o', label='Start')
    ax.scatter(trajectory[-1, 0], trajectory[-1, 1], c='red', s=100, marker='s', label='End')
    
    ax.set_xlabel(visualizer.dimension_names[0])
    ax.set_ylabel(visualizer.dimension_names[1])
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    return fig

def quick_validation_summary(validation_report: Dict[str, Any]) -> str:
    """Quick text summary of validation results."""
    if 'validation_report_summary' in validation_report:
        summary = validation_report['validation_report_summary']
        
        text = f"""
IUSMORFOS VALIDATION SUMMARY
============================
Framework Version: {summary.get('framework_version', 'N/A')}
World-Class Reproducibility: {'✓ YES' if summary.get('world_class_reproducibility_achieved') else '✗ NO'}
Overall Accuracy: {summary.get('overall_accuracy', 0):.1%}
Statistical Significance: p = {summary.get('statistical_significance_p_value', 1.0):.2e}
Validated Cases: {summary.get('total_validated_cases', 0)}

Status: {'MEETS WORLD-CLASS STANDARDS' if summary.get('world_class_reproducibility_achieved') else 'NEEDS IMPROVEMENT'}
        """
        return text.strip()
    else:
        return "Validation report not available or incomplete."

# Example usage and testing
if __name__ == "__main__":
    print("=== Iusmorfos Visualizer v4.0 Test ===")
    
    # Create test visualizer
    visualizer = IusmorfosVisualizer(style='academic')
    
    # Generate test trajectory data
    np.random.seed(42)
    time_steps = 100
    trajectory = np.zeros((time_steps, 9))
    
    # Simulate institutional evolution
    initial_state = np.array([0.3, 0.7, 0.6, 0.8, 0.5, -0.2, 0.4, 0.9, 0.6])
    target_state = np.array([0.7, 0.8, 0.4, 0.9, 0.6, 0.8, 0.8, -0.5, 0.5])
    
    for t in range(time_steps):
        alpha = t / (time_steps - 1)
        trajectory[t] = initial_state + alpha * (target_state - initial_state)
        # Add some noise
        trajectory[t] += np.random.normal(0, 0.05, 9)
        # Keep in bounds
        trajectory[t] = np.clip(trajectory[t], -1, 1)
    
    # Test trajectory visualization
    print("Creating trajectory visualizations...")
    fig1 = visualizer.plot_9d_trajectory_pca(trajectory, 
                                             title="Test 9D Trajectory (PCA)")
    print("✓ 9D trajectory PCA plot created")
    
    fig2 = visualizer.plot_dimensional_evolution(trajectory,
                                               title="Test Dimensional Evolution")
    print("✓ Dimensional evolution plot created")
    
    # Test quick visualization
    fig3 = quick_trajectory_plot(trajectory, "Quick Test Trajectory")
    print("✓ Quick trajectory plot created")
    
    # Test validation summary
    test_validation = {
        'validation_report_summary': {
            'framework_version': '4.0',
            'world_class_reproducibility_achieved': True,
            'overall_accuracy': 0.94,
            'statistical_significance_p_value': 0.00001,
            'total_validated_cases': 15
        }
    }
    
    summary_text = quick_validation_summary(test_validation)
    print("✓ Validation summary generated")
    print(summary_text)
    
    print("\n=== Visualizer Test Complete ===")
    print("All visualization components functional")
    print("Ready for integration with analysis results")