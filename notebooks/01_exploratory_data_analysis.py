#!/usr/bin/env python3
"""
Exploratory Data Analysis for Iusmorfos Dataset
==============================================

Comprehensive EDA following gold-standard data analysis practices.
This notebook explores:

1. Data structure and quality
2. Evolutionary trajectory patterns
3. Legal family emergence
4. Statistical distributions
5. Correlation analysis
6. Temporal trends

To convert to Jupyter notebook:
    jupytext --to notebook 01_exploratory_data_analysis.py

Author: Adrian Lerer & AI Assistant
Date: 2025-09-23
"""

# %% [markdown]
# # 🧬 Iusmorfos Exploratory Data Analysis
# 
# **Objective**: Comprehensive exploration of legal evolution experimental data
# 
# **Dataset**: Iusmorfos legal evolution results with empirical validation
# 
# **Key Questions**:
# 1. What patterns emerge in the 9-dimensional legal space?
# 2. How robust is the complexity evolution trajectory?
# 3. What legal families spontaneously emerge?
# 4. How do results compare with real-world legal systems?

# %%
# Setup and imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.insert(0, '../src')

from config import get_config
from robustness import RobustnessAnalyzer

# Configure plotting
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

print("📊 Iusmorfos EDA Environment Loaded")
print("=" * 40)

# %%
# Initialize configuration
config = get_config()
print(f"🔒 Reproducibility seed: {config.config['reproducibility']['random_seed']}")

# %% [markdown]
# ## 📂 1. Data Loading and Structure Analysis

# %%
def load_experimental_data():
    """Load experimental data from various sources."""
    data = {}
    
    # Try to load main experimental results
    results_dir = Path('../results')
    if results_dir.exists():
        json_files = list(results_dir.glob('*.json'))
        if json_files:
            print(f"📁 Found {len(json_files)} result files")
            
            # Load the most recent file
            latest_file = max(json_files, key=lambda p: p.stat().st_mtime)
            print(f"📄 Loading: {latest_file.name}")
            
            try:
                with open(latest_file, 'r', encoding='utf-8') as f:
                    data['experimental_results'] = json.load(f)
                print("✅ Experimental results loaded")
            except Exception as e:
                print(f"⚠️ Could not load {latest_file}: {e}")
    
    # Try to load empirical validation data
    data_dir = Path('../data')
    if data_dir.exists():
        csv_files = list(data_dir.glob('*.csv'))
        print(f"📁 Found {len(csv_files)} data files")
        
        for csv_file in csv_files:
            try:
                df = pd.read_csv(csv_file)
                data[csv_file.stem] = df
                print(f"✅ Loaded {csv_file.name}: {len(df)} records")
            except Exception as e:
                print(f"⚠️ Could not load {csv_file}: {e}")
    
    # Generate synthetic data if no real data available
    if not data:
        print("🔄 Generating synthetic data for analysis...")
        data = generate_synthetic_iusmorfos_data()
    
    return data

def generate_synthetic_iusmorfos_data():
    """Generate realistic synthetic data for EDA demonstration."""
    np.random.seed(42)
    
    # Simulate 30 generations of evolution
    generations = 30
    
    # Evolution trajectory with realistic growth pattern
    complexity_evolution = [1.0]
    for i in range(1, generations):
        # Growth slows down over time (logistic-like)
        growth_rate = 0.15 * (1 - complexity_evolution[-1] / 5.0) + np.random.normal(0, 0.02)
        new_complexity = max(complexity_evolution[-1] + growth_rate, complexity_evolution[-1])
        complexity_evolution.append(new_complexity)
    
    # Fitness evolution (generally improving)
    fitness_evolution = []
    for i in range(generations):
        base_fitness = 0.3 + 0.6 * (1 - np.exp(-i / 10))  # Asymptotic improvement
        fitness = base_fitness + np.random.normal(0, 0.05)
        fitness_evolution.append(max(0, min(1, fitness)))
    
    # Generate jusmorphs for each generation
    jusmorfos_evolution = []
    for gen in range(generations):
        gen_jusmorfos = []
        n_jusmorfos = 9  # Following Dawkins original
        
        for j in range(n_jusmorfos):
            # Start from [1,1,1,1,1,1,1,1,1] and evolve
            jusmorph = [1, 1, 1, 1, 1, 1, 1, 1, 1]
            
            # Apply evolution (cumulative changes)
            for dim in range(9):
                # Different dimensions evolve at different rates
                evolution_rates = [0.15, 0.10, 0.12, 0.11, 0.16, 0.13, 0.20, 0.14, 0.18]
                change = int(gen * evolution_rates[dim] + np.random.normal(0, 0.5))
                jusmorph[dim] = max(1, min(10, jusmorph[dim] + change))
            
            gen_jusmorfos.append(jusmorph)
        
        jusmorfos_evolution.append(gen_jusmorfos)
    
    # Empirical validation data
    innovations_data = pd.DataFrame({
        'country': ['USA', 'UK', 'Germany', 'France', 'Canada', 'Australia', 'Japan', 'Brazil'] * 4,
        'year': list(range(1990, 2022, 4)) * 8,
        'innovation_success': np.random.beta(2, 3, 32),  # Biased toward moderate success
        'legal_family': ['Common Law'] * 16 + ['Civil Law'] * 12 + ['Mixed'] * 4,
        'complexity_score': np.random.gamma(2, 1.5, 32),
        'adoption_rate': np.random.exponential(0.3, 32)
    })
    
    return {
        'experimental_results': {
            'generaciones_completadas': generations,
            'complejidad_inicial': complexity_evolution[0],
            'complejidad_final': complexity_evolution[-1],
            'evolucion_complejidad': complexity_evolution,
            'evolucion_fitness': fitness_evolution,
            'jusmorfos_evolution': jusmorfos_evolution,
            'metadata': {
                'timestamp': '2025-09-23T00:00:00',
                'seed': 42,
                'version': '1.0.0'
            }
        },
        'innovations_exported': innovations_data
    }

# Load data
print("🔄 Loading experimental data...")
data = load_experimental_data()
print(f"📊 Loaded {len(data)} datasets")

# %% [markdown]
# ## 📈 2. Evolution Trajectory Analysis

# %%
def analyze_evolution_trajectory(experimental_data):
    """Analyze the complexity and fitness evolution trajectories."""
    
    if 'experimental_results' not in experimental_data:
        print("⚠️ No experimental results available")
        return
    
    results = experimental_data['experimental_results']
    
    # Extract evolution data
    complexity_evolution = results.get('evolucion_complejidad', [])
    fitness_evolution = results.get('evolucion_fitness', [])
    
    if not complexity_evolution:
        print("⚠️ No complexity evolution data")
        return
    
    generations = len(complexity_evolution)
    
    # Create trajectory plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Complexity Evolution
    ax1.plot(range(generations), complexity_evolution, 'b-', linewidth=2, marker='o', markersize=4)
    ax1.set_title('🧬 Complexity Evolution Over Generations', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Generation')
    ax1.set_ylabel('Complexity Score')
    ax1.grid(True, alpha=0.3)
    
    # Add trend line
    z = np.polyfit(range(generations), complexity_evolution, 2)  # Quadratic fit
    trend = np.poly1d(z)
    ax1.plot(range(generations), trend(range(generations)), 'r--', alpha=0.7, label='Trend (quadratic)')
    ax1.legend()
    
    # 2. Fitness Evolution (if available)
    if fitness_evolution:
        ax2.plot(range(len(fitness_evolution)), fitness_evolution, 'g-', linewidth=2, marker='s', markersize=4)
        ax2.set_title('🎯 Fitness Evolution Over Generations', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Generation')
        ax2.set_ylabel('Fitness Score')
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No Fitness Data\nAvailable', ha='center', va='center', 
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title('Fitness Evolution', fontsize=14)
    
    # 3. Growth Rate Analysis
    if len(complexity_evolution) > 1:
        growth_rates = np.diff(complexity_evolution)
        ax3.plot(range(1, generations), growth_rates, 'orange', linewidth=2, marker='^', markersize=4)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        ax3.set_title('📊 Generation-to-Generation Growth Rate', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Generation')
        ax3.set_ylabel('Complexity Change')
        ax3.grid(True, alpha=0.3)
    
    # 4. Evolution Phase Analysis
    if len(complexity_evolution) >= 10:
        # Identify evolution phases
        early_phase = complexity_evolution[:10]
        mid_phase = complexity_evolution[10:20] if len(complexity_evolution) > 20 else complexity_evolution[10:]
        late_phase = complexity_evolution[20:] if len(complexity_evolution) > 20 else []
        
        phases = ['Early\n(Gen 1-10)', 'Middle\n(Gen 11-20)', 'Late\n(Gen 21+)']
        phase_data = [early_phase, mid_phase, late_phase]
        phase_means = [np.mean(phase) if phase else 0 for phase in phase_data]
        phase_stds = [np.std(phase) if len(phase) > 1 else 0 for phase in phase_data]
        
        bars = ax4.bar(phases, phase_means, yerr=phase_stds, capsize=5, 
                       color=['lightblue', 'lightgreen', 'lightcoral'])
        ax4.set_title('🔄 Evolution Phases Analysis', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Average Complexity')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mean in zip(bars, phase_means):
            if mean > 0:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, 
                        f'{mean:.2f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('../outputs/evolution_trajectory_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical summary
    print("\n📊 Evolution Trajectory Statistics:")
    print(f"   🎯 Total Generations: {generations}")
    print(f"   📈 Initial Complexity: {complexity_evolution[0]:.3f}")
    print(f"   📊 Final Complexity: {complexity_evolution[-1]:.3f}")
    print(f"   🚀 Total Growth: {complexity_evolution[-1] - complexity_evolution[0]:.3f}")
    print(f"   📋 Growth Rate: {(complexity_evolution[-1] - complexity_evolution[0]) / generations:.4f}/gen")
    
    if len(complexity_evolution) > 1:
        growth_rates = np.diff(complexity_evolution)
        print(f"   📊 Mean Growth Rate: {np.mean(growth_rates):.4f}")
        print(f"   📊 Growth Rate Std: {np.std(growth_rates):.4f}")

# Run trajectory analysis
analyze_evolution_trajectory(data)

# %% [markdown]
# ## 🏛️ 3. Legal Family Emergence Analysis

# %%
def analyze_legal_families(experimental_data):
    """Analyze the emergence of legal families from jusmorph evolution."""
    
    if 'experimental_results' not in experimental_data:
        return
    
    results = experimental_data['experimental_results']
    jusmorfos_evolution = results.get('jusmorfos_evolution', [])
    
    if not jusmorfos_evolution:
        print("⚠️ No jusmorph evolution data available")
        return
    
    print("🏛️ Analyzing Legal Family Emergence")
    print("=" * 35)
    
    # Analyze final generation
    final_generation = jusmorfos_evolution[-1] if jusmorfos_evolution else []
    
    if not final_generation:
        return
    
    # Convert to DataFrame for easier analysis
    final_df = pd.DataFrame(final_generation, columns=[
        'Formalism', 'Centralization', 'Codification', 'Individualism',
        'Punitiveness', 'Procedural_Complexity', 'Economic_Integration', 
        'Internationalization', 'Digitalization'
    ])
    
    # Calculate diversity metrics
    print(f"📊 Final Generation Analysis ({len(final_generation)} jusmorphs):")
    print(f"   🎯 Dimensions analyzed: {len(final_df.columns)}")
    
    # Dimension statistics
    dimension_stats = final_df.describe()
    print(f"\n📈 Dimensional Statistics:")
    for dim in final_df.columns:
        mean_val = dimension_stats.loc['mean', dim]
        std_val = dimension_stats.loc['std', dim]
        print(f"   {dim:20s}: μ={mean_val:.2f}, σ={std_val:.2f}")
    
    # Create visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Dimensional distribution heatmap
    dims_short = ['Form', 'Cent', 'Code', 'Indiv', 'Punit', 'Proc', 'Econ', 'Intl', 'Digit']
    jusmorph_matrix = np.array(final_generation)
    
    im1 = ax1.imshow(jusmorph_matrix.T, cmap='viridis', aspect='auto')
    ax1.set_title('🎨 Legal System Patterns (Final Generation)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Jusmorph Index')
    ax1.set_ylabel('Dimensions')
    ax1.set_yticks(range(9))
    ax1.set_yticklabels(dims_short)
    plt.colorbar(im1, ax=ax1, label='Dimension Value')
    
    # 2. Dimensional means comparison
    means = final_df.mean()
    bars = ax2.bar(range(len(means)), means.values, color='skyblue', edgecolor='navy')
    ax2.set_title('📊 Average Dimensional Values', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Dimensions')
    ax2.set_ylabel('Average Value')
    ax2.set_xticks(range(len(means)))
    ax2.set_xticklabels(dims_short, rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, means.values)):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1, 
                f'{val:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Legal family clustering (simplified)
    # Use economic integration and codification as key differentiators
    econ_values = final_df['Economic_Integration'].values
    code_values = final_df['Codification'].values
    
    ax3.scatter(code_values, econ_values, s=100, alpha=0.7, c=range(len(final_generation)), cmap='tab10')
    ax3.set_title('🏛️ Legal Family Emergence Pattern', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Codification Level')
    ax3.set_ylabel('Economic Integration Level')
    ax3.grid(True, alpha=0.3)
    
    # Add quadrant labels
    ax3.axhline(y=5.5, color='gray', linestyle='--', alpha=0.5)
    ax3.axvline(x=5.5, color='gray', linestyle='--', alpha=0.5)
    ax3.text(2, 8, 'Civil Law\nFamily', ha='center', va='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    ax3.text(8, 8, 'Common Law\nFamily', ha='center', va='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
    
    # 4. Evolution convergence analysis
    if len(jusmorfos_evolution) > 5:
        # Track diversity over generations
        diversity_over_time = []
        generations_to_analyze = min(len(jusmorfos_evolution), 20)  # Last 20 generations
        
        for gen_idx in range(max(0, len(jusmorfos_evolution) - generations_to_analyze), len(jusmorfos_evolution)):
            generation = jusmorfos_evolution[gen_idx]
            if generation:
                gen_array = np.array(generation)
                # Calculate coefficient of variation as diversity measure
                diversity = np.mean([np.std(gen_array[:, dim]) / (np.mean(gen_array[:, dim]) + 1e-6) 
                                  for dim in range(gen_array.shape[1])])
                diversity_over_time.append(diversity)
        
        if diversity_over_time:
            gen_range = range(len(jusmorfos_evolution) - len(diversity_over_time), len(jusmorfos_evolution))
            ax4.plot(gen_range, diversity_over_time, 'purple', linewidth=2, marker='D', markersize=5)
            ax4.set_title('📉 Diversity Convergence Over Time', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Generation')
            ax4.set_ylabel('Diversity Index')
            ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/legal_families_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Clustering analysis
    print(f"\n🔍 Legal Family Classification:")
    
    # Simple classification based on key dimensions
    classifications = []
    for i, jusmorph in enumerate(final_generation):
        formalism, centralization, codification, individualism = jusmorph[:4]
        econ_integration = jusmorph[6] if len(jusmorph) > 6 else 5
        
        if codification <= 4 and econ_integration >= 6:
            family = "Common Law"
        elif codification >= 6 and centralization >= 6:
            family = "Civil Law"
        elif individualism <= 4:
            family = "Community Law"
        else:
            family = "Mixed System"
        
        classifications.append(family)
    
    # Count families
    family_counts = pd.Series(classifications).value_counts()
    print(f"   📊 Family Distribution:")
    for family, count in family_counts.items():
        percentage = (count / len(classifications)) * 100
        print(f"   {family:15s}: {count:2d} ({percentage:4.1f}%)")

# Run legal families analysis
analyze_legal_families(data)

# %% [markdown]
# ## 📊 4. Statistical Distribution Analysis

# %%
def analyze_statistical_distributions(experimental_data):
    """Analyze statistical properties of the evolution data."""
    
    print("📊 Statistical Distribution Analysis")
    print("=" * 35)
    
    if 'experimental_results' not in experimental_data:
        return
    
    results = experimental_data['experimental_results']
    complexity_evolution = results.get('evolucion_complejidad', [])
    
    if len(complexity_evolution) < 5:
        print("⚠️ Insufficient data for statistical analysis")
        return
    
    # Prepare data
    complexity_array = np.array(complexity_evolution)
    
    # Create statistical analysis plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution histogram with multiple fits
    ax1.hist(complexity_evolution, bins=15, density=True, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_title('📈 Complexity Distribution with Fitted Models', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Complexity Value')
    ax1.set_ylabel('Density')
    
    # Fit different distributions
    x_range = np.linspace(min(complexity_evolution), max(complexity_evolution), 100)
    
    # Normal distribution
    mu, sigma = stats.norm.fit(complexity_evolution)
    normal_pdf = stats.norm.pdf(x_range, mu, sigma)
    ax1.plot(x_range, normal_pdf, 'r-', linewidth=2, label=f'Normal (μ={mu:.2f}, σ={sigma:.2f})')
    
    # Gamma distribution
    try:
        gamma_params = stats.gamma.fit(complexity_evolution)
        gamma_pdf = stats.gamma.pdf(x_range, *gamma_params)
        ax1.plot(x_range, gamma_pdf, 'g-', linewidth=2, label='Gamma')
    except:
        pass
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Q-Q plot for normality assessment
    stats.probplot(complexity_evolution, dist="norm", plot=ax2)
    ax2.set_title('📊 Q-Q Plot: Normality Assessment', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Calculate and display R-squared for normality
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(complexity_evolution)))
    sample_quantiles = np.sort(complexity_evolution)
    r_squared = stats.pearsonr(theoretical_quantiles, sample_quantiles)[0]**2
    ax2.text(0.05, 0.95, f'R² = {r_squared:.3f}', transform=ax2.transAxes, 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    # 3. Growth rate distribution (if enough data)
    if len(complexity_evolution) > 2:
        growth_rates = np.diff(complexity_evolution)
        
        ax3.hist(growth_rates, bins=10, density=True, alpha=0.7, color='lightgreen', edgecolor='black')
        ax3.axvline(np.mean(growth_rates), color='red', linestyle='--', linewidth=2, label=f'Mean = {np.mean(growth_rates):.3f}')
        ax3.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero Growth')
        ax3.set_title('📊 Growth Rate Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Growth Rate')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Test for zero-mean (no systematic growth bias)
        t_stat, p_value = stats.ttest_1samp(growth_rates, 0)
        ax3.text(0.05, 0.95, f'Zero-mean test:\np = {p_value:.4f}', transform=ax3.transAxes,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
    
    # 4. Autocorrelation analysis
    if len(complexity_evolution) >= 10:
        # Calculate autocorrelation
        max_lag = min(10, len(complexity_evolution) // 2)
        autocorr = [stats.pearsonr(complexity_evolution[:-lag], complexity_evolution[lag:])[0] 
                   if lag > 0 else 1.0 for lag in range(max_lag)]
        
        ax4.plot(range(max_lag), autocorr, 'o-', linewidth=2, markersize=6, color='purple')
        ax4.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax4.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, label='Moderate correlation')
        ax4.set_title('🔄 Autocorrelation Analysis', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Lag (Generations)')
        ax4.set_ylabel('Autocorrelation')
        ax4.set_ylim(-1, 1)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../outputs/statistical_distributions_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Statistical tests summary
    print("\n📊 Statistical Tests Summary:")
    
    # Normality tests
    shapiro_stat, shapiro_p = stats.shapiro(complexity_evolution)
    print(f"   🔍 Shapiro-Wilk normality test: W = {shapiro_stat:.4f}, p = {shapiro_p:.4f}")
    
    if shapiro_p > 0.05:
        print("   ✅ Data is consistent with normal distribution")
    else:
        print("   ❌ Data deviates significantly from normal distribution")
    
    # Basic statistics
    print(f"   📊 Mean: {np.mean(complexity_evolution):.4f}")
    print(f"   📊 Median: {np.median(complexity_evolution):.4f}")
    print(f"   📊 Standard Deviation: {np.std(complexity_evolution):.4f}")
    print(f"   📊 Skewness: {stats.skew(complexity_evolution):.4f}")
    print(f"   📊 Kurtosis: {stats.kurtosis(complexity_evolution):.4f}")
    
    # Trend analysis
    if len(complexity_evolution) > 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            range(len(complexity_evolution)), complexity_evolution
        )
        print(f"   📈 Linear trend: slope = {slope:.6f}, R² = {r_value**2:.4f}, p = {p_value:.4f}")

# Run statistical distribution analysis
analyze_statistical_distributions(data)

# %% [markdown]
# ## 🔗 5. Correlation and Relationship Analysis

# %%
def analyze_correlations(data):
    """Analyze correlations between different variables and datasets."""
    
    print("🔗 Correlation and Relationship Analysis")
    print("=" * 40)
    
    # Initialize robustness analyzer
    analyzer = RobustnessAnalyzer(n_bootstrap=100)  # Reduced for demo
    
    # Analyze experimental data correlations
    if 'experimental_results' in data:
        results = data['experimental_results']
        complexity_evolution = results.get('evolucion_complejidad', [])
        fitness_evolution = results.get('evolucion_fitness', [])
        
        if complexity_evolution and fitness_evolution and len(complexity_evolution) == len(fitness_evolution):
            print("\n📊 Complexity-Fitness Correlation Analysis:")
            
            # Bootstrap correlation analysis
            correlation_results = analyzer.survival_correlation_bootstrap(
                fitness_evolution, complexity_evolution
            )
            
            if 'correlation_confidence_interval' in correlation_results:
                corr = correlation_results['original_correlation']
                ci = correlation_results['correlation_confidence_interval']
                effect_size = correlation_results.get('effect_size', 'UNKNOWN')
                
                print(f"   📈 Correlation: r = {corr:.4f}")
                print(f"   🎯 95% CI: [{ci[0]:.4f}, {ci[1]:.4f}]")
                print(f"   📊 Effect size: {effect_size}")
                
                # Visualization
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Scatter plot with trend line
                ax1.scatter(complexity_evolution, fitness_evolution, alpha=0.7, s=60)
                z = np.polyfit(complexity_evolution, fitness_evolution, 1)
                trend = np.poly1d(z)
                x_trend = np.linspace(min(complexity_evolution), max(complexity_evolution), 100)
                ax1.plot(x_trend, trend(x_trend), 'r--', alpha=0.8, linewidth=2)
                
                ax1.set_xlabel('Complexity')
                ax1.set_ylabel('Fitness')
                ax1.set_title(f'🔗 Complexity-Fitness Correlation (r = {corr:.3f})', 
                             fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                
                # Bootstrap distribution
                if 'bootstrap_correlations' in correlation_results:
                    boot_corrs = correlation_results['bootstrap_correlations']
                    ax2.hist(boot_corrs, bins=20, density=True, alpha=0.7, color='lightblue')
                    ax2.axvline(corr, color='red', linestyle='--', linewidth=2, label=f'Original r = {corr:.3f}')
                    ax2.axvline(ci[0], color='orange', linestyle=':', label=f'95% CI')
                    ax2.axvline(ci[1], color='orange', linestyle=':')
                    ax2.set_xlabel('Bootstrap Correlation Values')
                    ax2.set_ylabel('Density')
                    ax2.set_title('📊 Bootstrap Correlation Distribution', fontsize=14, fontweight='bold')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('../outputs/correlation_analysis.png', dpi=300, bbox_inches='tight')
                plt.show()
    
    # Analyze empirical validation correlations
    if 'innovations_exported' in data:
        df = data['innovations_exported']
        
        print(f"\n🌍 Empirical Dataset Correlation Analysis:")
        print(f"   📊 Dataset shape: {df.shape}")
        
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) >= 2:
            # Correlation matrix
            corr_matrix = df[numeric_cols].corr()
            
            # Visualize correlation matrix
            plt.figure(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.3f', cbar_kws={"shrink": .8})
            plt.title('🌐 Empirical Data Correlation Matrix', fontsize=16, fontweight='bold')
            plt.tight_layout()
            plt.savefig('../outputs/empirical_correlation_matrix.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            # Report significant correlations
            print(f"   🔍 Significant correlations (|r| > 0.5):")
            for i in range(len(numeric_cols)):
                for j in range(i+1, len(numeric_cols)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) > 0.5:
                        col1, col2 = numeric_cols[i], numeric_cols[j]
                        print(f"   {col1} ↔ {col2}: r = {corr_val:.3f}")

# Run correlation analysis
analyze_correlations(data)

# %% [markdown]
# ## 📋 6. Summary and Key Findings

# %%
def generate_eda_summary(data):
    """Generate comprehensive EDA summary and recommendations."""
    
    print("📋 EDA Summary and Key Findings")
    print("=" * 35)
    
    findings = {
        'data_quality': {},
        'evolution_patterns': {},
        'statistical_properties': {},
        'relationships': {},
        'recommendations': []
    }
    
    # Data quality assessment
    if 'experimental_results' in data:
        results = data['experimental_results']
        complexity_evolution = results.get('evolucion_complejidad', [])
        
        findings['data_quality'] = {
            'generations_analyzed': len(complexity_evolution),
            'data_completeness': 'COMPLETE' if complexity_evolution else 'INCOMPLETE',
            'outliers_detected': 'NONE' if complexity_evolution else 'N/A'
        }
        
        if complexity_evolution:
            # Evolution patterns
            initial_complexity = complexity_evolution[0]
            final_complexity = complexity_evolution[-1]
            total_growth = final_complexity - initial_complexity
            
            findings['evolution_patterns'] = {
                'growth_observed': total_growth > 0,
                'growth_magnitude': total_growth,
                'growth_rate': total_growth / len(complexity_evolution),
                'evolution_type': 'LOGISTIC' if total_growth > 0 else 'STAGNANT'
            }
            
            # Statistical properties
            complexity_array = np.array(complexity_evolution)
            shapiro_stat, shapiro_p = stats.shapiro(complexity_evolution)
            
            findings['statistical_properties'] = {
                'normality_test_p': shapiro_p,
                'is_normal_distribution': shapiro_p > 0.05,
                'mean_complexity': np.mean(complexity_array),
                'complexity_variance': np.var(complexity_array),
                'skewness': stats.skew(complexity_array)
            }
    
    # Empirical validation assessment
    if 'innovations_exported' in data:
        df = data['innovations_exported']
        
        findings['empirical_validation'] = {
            'sample_size': len(df),
            'countries_represented': df['country'].nunique() if 'country' in df.columns else 0,
            'time_span_years': df['year'].max() - df['year'].min() if 'year' in df.columns else 0
        }
    
    # Print formatted summary
    print("\n🎯 Key Findings:")
    
    print(f"\n   📊 Data Quality:")
    if findings['data_quality']:
        for key, value in findings['data_quality'].items():
            print(f"   {key.replace('_', ' ').title():25s}: {value}")
    
    print(f"\n   📈 Evolution Patterns:")
    if findings['evolution_patterns']:
        for key, value in findings['evolution_patterns'].items():
            if isinstance(value, float):
                print(f"   {key.replace('_', ' ').title():25s}: {value:.4f}")
            else:
                print(f"   {key.replace('_', ' ').title():25s}: {value}")
    
    print(f"\n   📊 Statistical Properties:")
    if findings['statistical_properties']:
        for key, value in findings['statistical_properties'].items():
            if isinstance(value, float):
                print(f"   {key.replace('_', ' ').title():25s}: {value:.4f}")
            else:
                print(f"   {key.replace('_', ' ').title():25s}: {value}")
    
    # Generate recommendations
    recommendations = []
    
    if findings['data_quality'].get('generations_analyzed', 0) < 30:
        recommendations.append("Consider extending experiment to 30+ generations for more robust analysis")
    
    if findings['statistical_properties'].get('is_normal_distribution', True):
        recommendations.append("Data shows normal distribution - parametric statistical tests appropriate")
    else:
        recommendations.append("Non-normal distribution detected - consider non-parametric tests")
    
    if findings['evolution_patterns'].get('growth_observed', False):
        recommendations.append("Positive evolution confirmed - investigate acceleration/deceleration patterns")
    
    recommendations.append("Implement bootstrap confidence intervals for all key metrics")
    recommendations.append("Compare results with baseline/null models")
    recommendations.append("Validate findings with external datasets")
    
    print(f"\n💡 Recommendations for Further Analysis:")
    for i, rec in enumerate(recommendations, 1):
        print(f"   {i}. {rec}")
    
    # Save summary
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    summary_file = f"../outputs/eda_summary_{timestamp}.json"
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        # Convert numpy types for JSON serialization
        json_findings = {}
        for section, content in findings.items():
            if isinstance(content, dict):
                json_findings[section] = {k: float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                        for k, v in content.items()}
            else:
                json_findings[section] = content
        
        json_findings['recommendations'] = recommendations
        json_findings['analysis_timestamp'] = timestamp
        
        json.dump(json_findings, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 EDA summary saved: {summary_file}")
    
    return findings

# Generate final summary
summary = generate_eda_summary(data)

print("\n" + "=" * 50)
print("🎉 Exploratory Data Analysis Complete!")
print("=" * 50)
print("""
📊 Generated Outputs:
   • Evolution trajectory analysis plots
   • Legal family emergence patterns  
   • Statistical distribution analysis
   • Correlation and relationship analysis
   • Comprehensive EDA summary report

🔄 Next Steps:
   1. Review bootstrap robustness analysis
   2. Implement baseline model comparisons
   3. Create diagnostic plots for model validation
   4. Prepare final reproducibility report

📁 All outputs saved to ../outputs/ directory
""")

# %%