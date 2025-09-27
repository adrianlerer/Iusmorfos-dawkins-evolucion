#!/usr/bin/env python3
"""
Visualization Suite - Iusmorfos Framework v4.0
===============================================

Suite completa de visualización para el análisis de sistemas institucionales
en espacio 9-dimensional con capacidades interactivas y análisis temporal.

Author: Adrian Lerer & Claude (AI Assistant)
Version: 4.0
Date: September 2024

Key Features:
- Interactive 9D constitutional space visualization
- Institutional trajectory plotting with temporal analysis
- Attractor basin mapping and competitive dynamics
- SAPNC reality filter visualization
- Cross-cultural comparative analysis
- Real-time prediction interface
- Validation metrics dashboard
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional, Any
import json
import datetime
import warnings
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConstitutionalSpaceVisualizer:
    """
    Visualizer for 9-dimensional constitutional space analysis.
    """
    
    def __init__(self, style_theme: str = "plotly_white"):
        """
        Initialize visualizer with styling preferences.
        
        Args:
            style_theme: Plotly theme for visualizations
        """
        self.style_theme = style_theme
        self.dimension_names = [
            'Federal Structure', 'Judicial Independence', 'Democratic Participation',
            'Individual Rights', 'Separation of Powers', 'Constitutional Stability',
            'Rule of Law', 'Social Rights', 'Checks & Balances'
        ]
        
        # Color schemes for different analyses
        self.color_schemes = {
            'institutions': px.colors.qualitative.Set3,
            'trajectories': px.colors.sequential.Viridis,
            'basins': px.colors.qualitative.Pastel,
            'validation': px.colors.sequential.RdYlGn,
            'risks': px.colors.sequential.Reds
        }
        
        # Set default template
        self._setup_plotting_style()
        
        logger.info("Constitutional Space Visualizer initialized")
        
    def _setup_plotting_style(self):
        """Set up plotting style and templates."""
        # Matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
        
        # Custom color palettes
        self.custom_colors = {
            'argentina': '#74ACDF',  # Light blue (Argentina flag)
            'colombia': '#FCDD09',   # Yellow (Colombia flag)  
            'chile': '#D52B1E',      # Red (Chile flag)
            'usa': '#B22234',        # Red (USA flag)
            'prediction': '#1f77b4', # Blue
            'actual': '#ff7f0e',     # Orange
            'confidence': '#d62728'  # Red
        }
        
    def plot_constitutional_radar(self, 
                                constitutional_data: Dict[str, Dict[str, float]],
                                title: str = "Constitutional Comparison",
                                show_confidence: bool = True) -> go.Figure:
        """
        Create radar chart for constitutional parameters comparison.
        
        Args:
            constitutional_data: Dict with countries/cases and their 9D parameters
            title: Chart title
            show_confidence: Whether to show confidence intervals
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Add trace for each country/case
        for i, (case_name, parameters) in enumerate(constitutional_data.items()):
            # Extract values in dimension order
            values = []
            for dim_name in self.dimension_names:
                # Convert dimension name to parameter key
                param_key = dim_name.lower().replace(' ', '_').replace('&', 'and')
                values.append(parameters.get(param_key, 0.0))
                
            # Close the radar chart
            values_closed = values + [values[0]]
            dimensions_closed = self.dimension_names + [self.dimension_names[0]]
            
            # Add trace
            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=dimensions_closed,
                fill='toself',
                name=case_name,
                line=dict(color=self.color_schemes['institutions'][i % len(self.color_schemes['institutions'])]),
                fillcolor=f"rgba{(*px.colors.hex_to_rgb(self.color_schemes['institutions'][i % len(self.color_schemes['institutions'])]), 0.1)}"
            ))
            
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-1, 1],
                    tickmode='linear',
                    tick0=-1,
                    dtick=0.5
                )
            ),
            showlegend=True,
            title=dict(text=title, x=0.5),
            template=self.style_theme,
            width=800,
            height=600
        )
        
        return fig
        
    def plot_institutional_trajectory(self,
                                   trajectory_data: Dict[str, Any],
                                   dimensions_to_plot: List[str] = None,
                                   title: str = "Institutional Evolution Trajectory") -> go.Figure:
        """
        Plot institutional evolution trajectory over time.
        
        Args:
            trajectory_data: Trajectory data with time series
            dimensions_to_plot: Specific dimensions to visualize (default: all)
            title: Chart title
            
        Returns:
            Plotly figure with trajectory visualization
        """
        if dimensions_to_plot is None:
            dimensions_to_plot = self.dimension_names
            
        # Create subplots
        n_dims = len(dimensions_to_plot)
        rows = int(np.ceil(n_dims / 3))
        cols = min(3, n_dims)
        
        fig = make_subplots(
            rows=rows, 
            cols=cols,
            subplot_titles=dimensions_to_plot,
            vertical_spacing=0.08,
            horizontal_spacing=0.06
        )
        
        # Extract time and trajectory data
        if 'time' in trajectory_data:
            time = trajectory_data['time']
            trajectory = trajectory_data['trajectory']
        else:
            trajectory = trajectory_data
            time = np.arange(len(trajectory))
            
        # Plot each dimension
        for i, dim_name in enumerate(dimensions_to_plot):
            row = i // cols + 1
            col = i % cols + 1
            
            # Get dimension index
            dim_key = dim_name.lower().replace(' ', '_').replace('&', 'and')
            
            if isinstance(trajectory, np.ndarray) and trajectory.ndim == 2:
                # Trajectory is 2D array (time x dimensions)
                dim_index = self.dimension_names.index(dim_name)
                y_values = trajectory[:, dim_index]
            elif isinstance(trajectory, dict) and dim_key in trajectory:
                y_values = trajectory[dim_key]
            else:
                logger.warning(f"Could not find data for dimension {dim_name}")
                continue
                
            # Add trajectory line
            fig.add_trace(
                go.Scatter(
                    x=time,
                    y=y_values,
                    mode='lines+markers',
                    name=dim_name,
                    line=dict(color=self.color_schemes['trajectories'][i % len(self.color_schemes['trajectories'])]),
                    showlegend=False
                ),
                row=row, col=col
            )
            
            # Add horizontal reference lines
            fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=row, col=col)
            fig.add_hline(y=0.5, line_dash="dot", line_color="green", opacity=0.3, row=row, col=col)
            fig.add_hline(y=-0.5, line_dash="dot", line_color="red", opacity=0.3, row=row, col=col)
            
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template=self.style_theme,
            height=200 * rows + 100,
            width=1200
        )
        
        # Update axes labels
        fig.update_xaxes(title_text="Time")
        fig.update_yaxes(title_text="Value", range=[-1.1, 1.1])
        
        return fig
        
    def plot_attractor_basins_3d(self,
                               basins_data: Dict[str, Any],
                               sample_points: np.ndarray = None,
                               title: str = "Attractor Basins (PCA Projection)") -> go.Figure:
        """
        Plot attractor basins in 3D using PCA projection.
        
        Args:
            basins_data: Dictionary with basin information
            sample_points: Optional sample points to show basin membership
            title: Chart title
            
        Returns:
            Plotly 3D scatter plot
        """
        fig = go.Figure()
        
        # Extract attractor points
        attractor_points = []
        attractor_labels = []
        
        for basin_id, basin_info in basins_data.items():
            if 'attractor_point' in basin_info:
                attractor_points.append(basin_info['attractor_point'])
                attractor_labels.append(basin_id)
                
        if not attractor_points:
            logger.warning("No attractor points found in basin data")
            return fig
            
        # Convert to numpy array and apply PCA
        attractor_array = np.array(attractor_points)
        
        # Apply PCA for 3D visualization
        pca = PCA(n_components=3)
        attractor_3d = pca.fit_transform(attractor_array)
        
        # Add attractor points
        fig.add_trace(go.Scatter3d(
            x=attractor_3d[:, 0],
            y=attractor_3d[:, 1],
            z=attractor_3d[:, 2],
            mode='markers+text',
            marker=dict(
                size=12,
                color=np.arange(len(attractor_labels)),
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Basin ID")
            ),
            text=attractor_labels,
            textposition="top center",
            name="Attractors",
            hovertemplate="<b>%{text}</b><br>" +
                         "PC1: %{x:.3f}<br>" +
                         "PC2: %{y:.3f}<br>" +
                         "PC3: %{z:.3f}<extra></extra>"
        ))
        
        # Add sample points if provided
        if sample_points is not None and len(sample_points) > 0:
            sample_3d = pca.transform(sample_points)
            
            fig.add_trace(go.Scatter3d(
                x=sample_3d[:, 0],
                y=sample_3d[:, 1],
                z=sample_3d[:, 2],
                mode='markers',
                marker=dict(
                    size=3,
                    color='lightgray',
                    opacity=0.3
                ),
                name="Sample Points",
                showlegend=True
            ))
            
        fig.update_layout(
            title=dict(text=title, x=0.5),
            scene=dict(
                xaxis_title=f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)",
                yaxis_title=f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)",
                zaxis_title=f"PC3 ({pca.explained_variance_ratio_[2]:.1%} var)",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            template=self.style_theme,
            width=800,
            height=600
        )
        
        return fig
        
    def plot_sapnc_impact_analysis(self,
                                 predicted_vs_actual: Dict[str, Tuple[float, float]],
                                 sapnc_coefficients: Dict[str, float],
                                 title: str = "SAPNC Reality Filter Impact Analysis") -> go.Figure:
        """
        Visualize impact of SAPNC reality filter on predictions.
        
        Args:
            predicted_vs_actual: Dict mapping metrics to (predicted, actual) tuples
            sapnc_coefficients: SAPNC coefficients by country/case
            title: Chart title
            
        Returns:
            Plotly figure showing SAPNC impact
        """
        # Create subplot with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        metrics = list(predicted_vs_actual.keys())
        predicted_values = [v[0] for v in predicted_vs_actual.values()]
        actual_values = [v[1] for v in predicted_vs_actual.values()]
        
        # Calculate implementation gaps
        implementation_gaps = [abs(pred - actual) for pred, actual in zip(predicted_values, actual_values)]
        
        # Add predicted vs actual scatter
        fig.add_trace(
            go.Scatter(
                x=predicted_values,
                y=actual_values,
                mode='markers+text',
                text=metrics,
                textposition="top center",
                marker=dict(
                    size=12,
                    color=implementation_gaps,
                    colorscale='RdYlBu_r',
                    showscale=True,
                    colorbar=dict(title="Implementation Gap")
                ),
                name="Predicted vs Actual",
                hovertemplate="<b>%{text}</b><br>" +
                             "Predicted: %{x:.3f}<br>" +
                             "Actual: %{y:.3f}<br>" +
                             "Gap: %{marker.color:.3f}<extra></extra>"
            ),
            secondary_y=False
        )
        
        # Add perfect prediction line
        min_val = min(min(predicted_values), min(actual_values))
        max_val = max(max(predicted_values), max(actual_values))
        
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(color='red', dash='dash'),
                name="Perfect Prediction",
                showlegend=True
            ),
            secondary_y=False
        )
        
        # Add SAPNC coefficient bars on secondary axis
        if sapnc_coefficients:
            countries = list(sapnc_coefficients.keys())
            coefficients = list(sapnc_coefficients.values())
            
            fig.add_trace(
                go.Bar(
                    x=countries,
                    y=coefficients,
                    name="SAPNC Coefficient",
                    marker=dict(color='lightblue', opacity=0.7),
                    yaxis="y2"
                ),
                secondary_y=True
            )
            
        # Update layout
        fig.update_xaxes(title_text="Predicted Values")
        fig.update_yaxes(title_text="Actual Values", secondary_y=False)
        fig.update_yaxes(title_text="SAPNC Coefficient", secondary_y=True)
        
        fig.update_layout(
            title=dict(text=title, x=0.5),
            template=self.style_theme,
            width=900,
            height=600
        )
        
        return fig

class CompetitiveDynamicsVisualizer:
    """
    Visualizer for competitive evolutionary dynamics between institutional forms.
    """
    
    def __init__(self):
        """Initialize competitive dynamics visualizer."""
        self.setup_style()
        
    def setup_style(self):
        """Setup visualization style."""
        self.colors = {
            'fitness': px.colors.sequential.Viridis,
            'population': px.colors.sequential.Blues,
            'diversity': px.colors.sequential.Greens,
            'extinction': px.colors.sequential.Reds
        }
        
    def plot_fitness_evolution(self,
                             evolution_data: Dict[str, Any],
                             title: str = "Fitness Evolution Over Generations") -> go.Figure:
        """
        Plot fitness evolution of species over generations.
        
        Args:
            evolution_data: Evolution simulation results
            title: Chart title
            
        Returns:
            Plotly figure with fitness evolution
        """
        fig = go.Figure()
        
        # Extract generation statistics
        if 'generation_stats' in evolution_data:
            generations = [stat['generation'] for stat in evolution_data['generation_stats']]
            mean_fitness = [stat['mean_fitness'] for stat in evolution_data['generation_stats']]
            max_fitness = [stat['max_fitness'] for stat in evolution_data['generation_stats']]
            min_fitness = [stat['min_fitness'] for stat in evolution_data['generation_stats']]
            
            # Add fitness traces
            fig.add_trace(go.Scatter(
                x=generations,
                y=mean_fitness,
                mode='lines+markers',
                name='Mean Fitness',
                line=dict(color='blue', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=generations,
                y=max_fitness,
                mode='lines',
                name='Max Fitness',
                line=dict(color='green', width=2, dash='dash')
            ))
            
            fig.add_trace(go.Scatter(
                x=generations,
                y=min_fitness,
                mode='lines',
                name='Min Fitness',
                line=dict(color='red', width=2, dash='dot')
            ))
            
            # Add fitness range area
            fig.add_trace(go.Scatter(
                x=generations + generations[::-1],
                y=max_fitness + min_fitness[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.1)',
                line=dict(color='rgba(255,255,255,0)'),
                name='Fitness Range',
                showlegend=False
            ))
            
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Generation",
            yaxis_title="Fitness Score",
            template="plotly_white",
            width=800,
            height=500
        )
        
        return fig
        
    def plot_species_competition_network(self,
                                      citation_network_data: Dict[str, Any],
                                      title: str = "Species Competition Network") -> go.Figure:
        """
        Plot network of species competition and citations.
        
        Args:
            citation_network_data: Network data from competitive arena
            title: Chart title
            
        Returns:
            Plotly network visualization
        """
        fig = go.Figure()
        
        # Extract network information
        if 'num_nodes' in citation_network_data and citation_network_data['num_nodes'] > 0:
            # For demonstration, create a simple network layout
            # In real implementation, would use actual network data
            
            num_nodes = min(citation_network_data['num_nodes'], 20)  # Limit for visualization
            
            # Generate circular layout for nodes
            angles = np.linspace(0, 2*np.pi, num_nodes, endpoint=False)
            x_pos = np.cos(angles)
            y_pos = np.sin(angles)
            
            # Node sizes based on degree (if available)
            node_sizes = np.random.uniform(10, 30, num_nodes)  # Placeholder
            node_colors = np.random.uniform(0, 1, num_nodes)   # Placeholder
            
            # Add nodes
            fig.add_trace(go.Scatter(
                x=x_pos,
                y=y_pos,
                mode='markers+text',
                marker=dict(
                    size=node_sizes,
                    color=node_colors,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Citation Count")
                ),
                text=[f"Species_{i+1}" for i in range(num_nodes)],
                textposition="middle center",
                name="Species",
                hovertemplate="<b>%{text}</b><br>" +
                             "Citations: %{marker.color:.0f}<extra></extra>"
            ))
            
            # Add edges (simplified)
            edge_x = []
            edge_y = []
            
            for i in range(num_nodes):
                for j in range(i+1, min(i+4, num_nodes)):  # Connect to nearby nodes
                    edge_x.extend([x_pos[i], x_pos[j], None])
                    edge_y.extend([y_pos[i], y_pos[j], None])
                    
            fig.add_trace(go.Scatter(
                x=edge_x,
                y=edge_y,
                mode='lines',
                line=dict(color='lightgray', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
            
        fig.update_layout(
            title=dict(text=title, x=0.5),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            template="plotly_white",
            width=700,
            height=700
        )
        
        return fig

class ValidationDashboard:
    """
    Interactive dashboard for validation metrics and performance tracking.
    """
    
    def __init__(self):
        """Initialize validation dashboard."""
        self.setup_dashboard_style()
        
    def setup_dashboard_style(self):
        """Setup dashboard styling."""
        self.colors = {
            'accuracy': '#2E8B57',     # Sea Green
            'error': '#DC143C',        # Crimson
            'confidence': '#4169E1',   # Royal Blue
            'neutral': '#708090'       # Slate Gray
        }
        
    def create_accuracy_timeline(self,
                               validation_history: List[Dict[str, Any]],
                               title: str = "Framework Accuracy Over Time") -> go.Figure:
        """
        Create timeline of framework accuracy evolution.
        
        Args:
            validation_history: List of validation results over time
            title: Chart title
            
        Returns:
            Plotly timeline figure
        """
        fig = go.Figure()
        
        if validation_history:
            timestamps = []
            accuracies = []
            confidence_lower = []
            confidence_upper = []
            num_cases = []
            
            for entry in validation_history:
                if 'analysis_timestamp' in entry:
                    timestamps.append(entry['analysis_timestamp'])
                    
                    overall_perf = entry.get('overall_performance', {})
                    accuracies.append(overall_perf.get('mean_accuracy', 0))
                    confidence_lower.append(overall_perf.get('accuracy_ci_lower', 0))
                    confidence_upper.append(overall_perf.get('accuracy_ci_upper', 1))
                    num_cases.append(overall_perf.get('num_cases', 0))
                    
            # Convert timestamps to datetime
            timestamps = pd.to_datetime(timestamps)
            
            # Add accuracy line with confidence intervals
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=accuracies,
                mode='lines+markers',
                name='Mean Accuracy',
                line=dict(color=self.colors['accuracy'], width=3),
                marker=dict(size=8)
            ))
            
            # Add confidence interval
            fig.add_trace(go.Scatter(
                x=list(timestamps) + list(timestamps[::-1]),
                y=confidence_upper + confidence_lower[::-1],
                fill='toself',
                fillcolor=f"rgba(46,139,87,0.2)",
                line=dict(color="rgba(255,255,255,0)"),
                name='95% Confidence Interval',
                showlegend=True
            ))
            
            # Add benchmark line (90% accuracy)
            fig.add_hline(
                y=0.9, 
                line_dash="dash", 
                line_color="red",
                annotation_text="World-class Benchmark (90%)"
            )
            
        fig.update_layout(
            title=dict(text=title, x=0.5),
            xaxis_title="Time",
            yaxis_title="Accuracy Score",
            yaxis=dict(range=[0, 1]),
            template="plotly_white",
            width=900,
            height=500
        )
        
        return fig
        
    def create_validation_metrics_dashboard(self,
                                          validation_report: Dict[str, Any]) -> Dict[str, go.Figure]:
        """
        Create comprehensive validation metrics dashboard.
        
        Args:
            validation_report: Complete validation report
            
        Returns:
            Dictionary of dashboard figures
        """
        dashboard_figures = {}
        
        # 1. Overall Performance Gauge
        if 'overall_performance' in validation_report.get('detailed_performance', {}):
            performance = validation_report['detailed_performance']['overall_performance']
            
            gauge_fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = performance.get('mean_accuracy', 0) * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Overall Accuracy (%)"},
                delta = {'reference': 90},  # World-class benchmark
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': self.colors['accuracy']},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 90], 'color': "yellow"},
                        {'range': [90, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            gauge_fig.update_layout(
                template="plotly_white",
                height=400,
                width=400
            )
            
            dashboard_figures['performance_gauge'] = gauge_fig
            
        # 2. Regional Performance Comparison
        if 'regional_performance' in validation_report.get('detailed_performance', {}):
            regional_perf = validation_report['detailed_performance']['regional_performance']
            
            if regional_perf and isinstance(regional_perf, dict):
                regions = []
                accuracies = []
                num_cases = []
                
                for region, stats in regional_perf.items():
                    if isinstance(stats, dict) and 'mean_accuracy' in stats:
                        regions.append(region)
                        accuracies.append(stats['mean_accuracy'])
                        num_cases.append(stats.get('num_cases', 0))
                        
                if regions:
                    regional_fig = go.Figure()
                    
                    regional_fig.add_trace(go.Bar(
                        x=regions,
                        y=accuracies,
                        text=[f"n={n}" for n in num_cases],
                        textposition='auto',
                        marker_color=self.colors['accuracy'],
                        name="Regional Accuracy"
                    ))
                    
                    regional_fig.add_hline(
                        y=0.9,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="World-class Benchmark"
                    )
                    
                    regional_fig.update_layout(
                        title="Regional Performance Comparison",
                        xaxis_title="Region",
                        yaxis_title="Mean Accuracy",
                        yaxis=dict(range=[0, 1]),
                        template="plotly_white",
                        width=600,
                        height=400
                    )
                    
                    dashboard_figures['regional_performance'] = regional_fig
                    
        # 3. Statistical Significance Indicator
        if 'statistical_significance' in validation_report.get('detailed_performance', {}):
            sig_test = validation_report['detailed_performance']['statistical_significance']
            
            p_value = sig_test.get('t_test', {}).get('p_value', 1.0)
            is_significant = p_value < 0.0001
            
            sig_fig = go.Figure()
            
            sig_fig.add_trace(go.Indicator(
                mode = "number+gauge",
                value = -np.log10(p_value) if p_value > 0 else 10,
                title = {'text': "Statistical Significance<br>(-log10 p-value)"},
                gauge = {
                    'axis': {'range': [0, 10]},
                    'bar': {'color': self.colors['confidence'] if is_significant else self.colors['error']},
                    'steps': [
                        {'range': [0, 4], 'color': "lightgray"},
                        {'range': [4, 10], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 4  # p < 0.0001
                    }
                }
            ))
            
            sig_fig.update_layout(
                template="plotly_white",
                height=400,
                width=400
            )
            
            dashboard_figures['significance_indicator'] = sig_fig
            
        return dashboard_figures

class PredictionInterface:
    """
    Interactive interface for making institutional predictions.
    """
    
    def __init__(self):
        """Initialize prediction interface."""
        self.dimension_names = [
            'Federal Structure', 'Judicial Independence', 'Democratic Participation',
            'Individual Rights', 'Separation of Powers', 'Constitutional Stability',
            'Rule of Law', 'Social Rights', 'Checks & Balances'
        ]
        
    def create_prediction_input_interface(self) -> go.Figure:
        """
        Create interactive interface for inputting institutional parameters.
        
        Returns:
            Plotly figure with interactive sliders
        """
        # Create figure with sliders (simplified version)
        fig = go.Figure()
        
        # Initialize with neutral values
        initial_values = [0.0] * len(self.dimension_names)
        
        # Create radar chart that will update with slider values
        fig.add_trace(go.Scatterpolar(
            r=initial_values + [initial_values[0]],
            theta=self.dimension_names + [self.dimension_names[0]],
            fill='toself',
            name='Current Configuration',
            line=dict(color='blue')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[-1, 1],
                    tickmode='linear',
                    tick0=-1,
                    dtick=0.5
                )
            ),
            title="Interactive Institutional Configuration",
            template="plotly_white",
            width=600,
            height=600
        )
        
        return fig
        
    def create_prediction_results_display(self,
                                        prediction_results: Dict[str, Any]) -> go.Figure:
        """
        Display prediction results with confidence intervals.
        
        Args:
            prediction_results: Results from prediction engine
            
        Returns:
            Plotly figure with prediction visualization
        """
        fig = go.Figure()
        
        if 'trajectory_stats' in prediction_results:
            stats = prediction_results['trajectory_stats']
            
            # Create timeline of predicted evolution
            if 'trajectory' in stats and 'time' in stats:
                trajectory = stats['trajectory']
                time = stats['time']
                
                # Plot first few dimensions as examples
                for i, dim_name in enumerate(self.dimension_names[:3]):
                    if i < trajectory.shape[1]:
                        fig.add_trace(go.Scatter(
                            x=time,
                            y=trajectory[:, i],
                            mode='lines',
                            name=dim_name,
                            line=dict(width=2)
                        ))
                        
            # Add convergence information
            if 'convergence_time' in stats:
                conv_time = stats['convergence_time']
                if np.isfinite(conv_time):
                    fig.add_vline(
                        x=conv_time,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Convergence: {conv_time:.1f}"
                    )
                    
        fig.update_layout(
            title="Predicted Institutional Evolution",
            xaxis_title="Time (years)",
            yaxis_title="Institutional Parameter Value",
            template="plotly_white",
            width=800,
            height=500
        )
        
        return fig

# Utility functions for comprehensive visualization suite
def create_comprehensive_analysis_report(analysis_data: Dict[str, Any]) -> Dict[str, go.Figure]:
    """
    Create comprehensive visual analysis report.
    
    Args:
        analysis_data: Complete analysis data from framework
        
    Returns:
        Dictionary of all visualization figures
    """
    figures = {}
    
    # Initialize visualizers
    const_viz = ConstitutionalSpaceVisualizer()
    comp_viz = CompetitiveDynamicsVisualizer()
    val_dash = ValidationDashboard()
    pred_interface = PredictionInterface()
    
    # Constitutional space analysis
    if 'constitutional_comparison' in analysis_data:
        figures['constitutional_radar'] = const_viz.plot_constitutional_radar(
            analysis_data['constitutional_comparison'],
            title="Constitutional Systems Comparison"
        )
        
    # Trajectory analysis
    if 'trajectory_data' in analysis_data:
        figures['trajectory_evolution'] = const_viz.plot_institutional_trajectory(
            analysis_data['trajectory_data'],
            title="Institutional Evolution Trajectory"
        )
        
    # Attractor basin visualization
    if 'attractor_basins' in analysis_data:
        figures['attractor_basins_3d'] = const_viz.plot_attractor_basins_3d(
            analysis_data['attractor_basins'],
            title="Attractor Basins in Constitutional Space"
        )
        
    # SAPNC impact analysis
    if 'sapnc_analysis' in analysis_data:
        figures['sapnc_impact'] = const_viz.plot_sapnc_impact_analysis(
            analysis_data['sapnc_analysis'].get('predicted_vs_actual', {}),
            analysis_data['sapnc_analysis'].get('sapnc_coefficients', {}),
            title="SAPNC Reality Filter Impact"
        )
        
    # Competitive dynamics
    if 'evolution_results' in analysis_data:
        figures['fitness_evolution'] = comp_viz.plot_fitness_evolution(
            analysis_data['evolution_results'],
            title="Institutional Fitness Evolution"
        )
        
    if 'citation_network' in analysis_data:
        figures['competition_network'] = comp_viz.plot_species_competition_network(
            analysis_data['citation_network'],
            title="Institutional Competition Network"
        )
        
    # Validation dashboard
    if 'validation_report' in analysis_data:
        val_figures = val_dash.create_validation_metrics_dashboard(
            analysis_data['validation_report']
        )
        figures.update(val_figures)
        
    if 'validation_history' in analysis_data:
        figures['accuracy_timeline'] = val_dash.create_accuracy_timeline(
            analysis_data['validation_history'],
            title="Framework Accuracy Evolution"
        )
        
    # Prediction interface
    figures['prediction_interface'] = pred_interface.create_prediction_input_interface()
    
    if 'prediction_results' in analysis_data:
        figures['prediction_display'] = pred_interface.create_prediction_results_display(
            analysis_data['prediction_results']
        )
        
    return figures

def export_figures_to_html(figures: Dict[str, go.Figure], 
                          output_file: str = "iusmorfos_analysis_report.html") -> str:
    """
    Export all figures to a comprehensive HTML report.
    
    Args:
        figures: Dictionary of Plotly figures
        output_file: Output HTML file path
        
    Returns:
        Path to generated HTML file
    """
    # Create HTML report with all figures
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Iusmorfos Framework Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            h1, h2 {{ color: #2E8B57; }}
            .figure-container {{ margin-bottom: 30px; border: 1px solid #ddd; padding: 10px; }}
            .metadata {{ background-color: #f5f5f5; padding: 10px; margin: 10px 0; }}
        </style>
    </head>
    <body>
        <h1>Iusmorfos Framework v4.0 - Comprehensive Analysis Report</h1>
        <div class="metadata">
            <p><strong>Generated:</strong> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            <p><strong>Framework Version:</strong> 4.0</p>
            <p><strong>Analysis Type:</strong> World-class Reproducibility Standard</p>
        </div>
    """
    
    # Add each figure
    for i, (fig_name, fig) in enumerate(figures.items()):
        fig_html = fig.to_html(include_plotlyjs=False, div_id=f"fig_{i}")
        
        # Extract just the div part
        div_start = fig_html.find('<div')
        div_end = fig_html.find('</div>') + 6
        fig_div = fig_html[div_start:div_end]
        
        html_content += f"""
        <div class="figure-container">
            <h2>{fig_name.replace('_', ' ').title()}</h2>
            {fig_div}
        </div>
        """
    
    html_content += """
        </body>
    </html>
    """
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
        
    logger.info(f"Comprehensive analysis report exported to {output_file}")
    return output_file

# Example usage and testing
if __name__ == "__main__":
    print("=== Iusmorfos Visualization Suite v4.0 ===")
    
    # Create sample data for testing
    sample_constitutional_data = {
        'Argentina_PreMilei': {
            'federal_structure': 0.4,
            'judicial_independence': 0.3,
            'democratic_participation': 0.6,
            'individual_rights': 0.5,
            'separation_of_powers': 0.2,
            'constitutional_stability': -0.3,
            'rule_of_law': 0.2,
            'social_rights': 0.7,
            'checks_and_balances': 0.3
        },
        'Argentina_Milei_Target': {
            'federal_structure': 0.7,
            'judicial_independence': 0.8,
            'democratic_participation': 0.4,
            'individual_rights': 0.9,
            'separation_of_powers': 0.6,
            'constitutional_stability': 0.8,
            'rule_of_law': 0.8,
            'social_rights': -0.5,
            'checks_and_balances': 0.5
        }
    }
    
    # Test constitutional radar chart
    const_viz = ConstitutionalSpaceVisualizer()
    radar_fig = const_viz.plot_constitutional_radar(
        sample_constitutional_data,
        title="Argentina: Pre-Milei vs. Milei Target"
    )
    
    print("✓ Constitutional radar chart created")
    
    # Test trajectory visualization
    sample_trajectory = {
        'time': np.linspace(0, 24, 100),
        'trajectory': np.random.randn(100, 9).cumsum(axis=0) * 0.1
    }
    
    trajectory_fig = const_viz.plot_institutional_trajectory(
        sample_trajectory,
        title="Sample Institutional Evolution"
    )
    
    print("✓ Trajectory visualization created")
    
    # Test validation dashboard
    sample_validation = {
        'detailed_performance': {
            'overall_performance': {
                'mean_accuracy': 0.92,
                'accuracy_ci_lower': 0.88,
                'accuracy_ci_upper': 0.95,
                'num_cases': 15
            },
            'statistical_significance': {
                't_test': {'p_value': 0.00005}
            },
            'regional_performance': {
                'Latin America': {'mean_accuracy': 0.90, 'num_cases': 8},
                'Europe': {'mean_accuracy': 0.94, 'num_cases': 4},
                'Asia': {'mean_accuracy': 0.93, 'num_cases': 3}
            }
        }
    }
    
    val_dash = ValidationDashboard()
    dashboard_figs = val_dash.create_validation_metrics_dashboard(sample_validation)
    
    print(f"✓ Validation dashboard created ({len(dashboard_figs)} figures)")
    
    # Export sample report
    all_figures = {
        'constitutional_comparison': radar_fig,
        'trajectory_evolution': trajectory_fig,
        **dashboard_figs
    }
    
    html_file = export_figures_to_html(all_figures, "sample_iusmorfos_report.html")
    print(f"✓ Sample report exported to: {html_file}")
    
    print("\n=== Visualization Suite Ready ===")
    print("Components available:")
    print("• ConstitutionalSpaceVisualizer - 9D constitutional analysis")
    print("• CompetitiveDynamicsVisualizer - Evolutionary dynamics")
    print("• ValidationDashboard - Performance metrics")  
    print("• PredictionInterface - Interactive predictions")
    print("• Comprehensive HTML reporting")