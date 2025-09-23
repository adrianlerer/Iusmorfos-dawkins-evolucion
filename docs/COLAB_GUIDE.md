# Google Colab Usage Guide for Iusmorfos Framework

## 🚀 Quick Start

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usuario/iusmorfos_public/blob/main/notebooks/Iusmorfos_Cloud_Analysis.ipynb)

This guide helps you get started with the **Iusmorfos Framework** using Google Colab for cloud-based legal system evolution analysis.

## 📋 Prerequisites

- Google account (for Google Colab access)
- Basic understanding of Python and Jupyter notebooks
- Interest in legal system analysis and computational law

## 🔧 Setup Instructions

### 1. Open the Notebook

Click the "Open in Colab" badge above or manually navigate to:
```
https://colab.research.google.com/github/usuario/iusmorfos_public/blob/main/notebooks/Iusmorfos_Cloud_Analysis.ipynb
```

### 2. Runtime Configuration

For optimal performance:
1. Go to **Runtime → Change runtime type**
2. Select **Python 3** as the runtime type
3. Choose **GPU** if available (for faster computations)
4. Click **Save**

### 3. Initial Setup

Run the first few cells to:
- Install required dependencies
- Clone the Iusmorfos repository
- Initialize the framework
- Load sample data

**⚠️ Important**: Always run cells in order, especially during the initial setup.

## 🎮 Interactive Features

### Country Comparison Tool

The notebook includes interactive widgets for comparing countries:

```python
# Use dropdown menus to select:
- Country 1: Choose from Argentina, Chile, South Africa, Sweden, India
- Country 2: Choose comparison country
- Focus Dimension: Select specific IusSpace dimension to analyze
```

### Real-time Analysis

Modify parameters in real-time:
- **Mutation rates**: Adjust evolution simulation parameters
- **Bootstrap samples**: Change validation sample sizes
- **Time periods**: Modify evolution simulation length
- **Cultural dimensions**: Focus on specific Hofstede dimensions

### Visualization Controls

Interactive plots allow you to:
- **Zoom and pan**: Explore detailed regions of plots
- **Toggle data series**: Show/hide specific countries or dimensions  
- **Export plots**: Save visualizations as PNG/PDF files
- **Hover tooltips**: Get detailed information on data points

## 📊 Analysis Workflows

### Workflow 1: Cross-Country Comparison

1. **Initialize Framework** (Run setup cells)
2. **Load Country Data** (Automatic)
3. **Generate Radar Chart** (Compare countries across 9 dimensions)
4. **Calculate Cultural Distances** (Hofstede-based analysis)
5. **Export Results** (Download CSV files)

### Workflow 2: Legal Evolution Simulation

1. **Select Target Country** (Use dropdown or modify code)
2. **Set Parameters** (Mutation rate, generations, random seed)
3. **Run Simulation** (Execute evolution cells)
4. **Analyze Results** (View fitness evolution and genetic changes)
5. **Compare Scenarios** (Run multiple simulations with different parameters)

### Workflow 3: Power-Law Validation

1. **Load Innovation Data** (842 legal innovations)
2. **Fit Power-Law Distribution** (Maximum likelihood estimation)
3. **Bootstrap Validation** (1000+ samples for confidence intervals)
4. **Sensitivity Analysis** (Test robustness to parameter changes)
5. **Generate Report** (Statistical summary with visualizations)

### Workflow 4: Cultural Adaptation Analysis

1. **Cultural Distance Matrix** (Hofstede dimensions)
2. **Legal System Clustering** (Group similar systems)
3. **Adaptation Patterns** (How culture influences legal evolution)
4. **Cross-Validation** (Test patterns across different countries)

## 🛠️ Customization Options

### Adding New Countries

To analyze additional countries:

```python
# Add to the countries dictionary in framework initialization
new_countries = {
    'Your_Country': {
        'tradition': 'Civil Law',  # or 'Common Law', 'Mixed'
        'region': 'Your_Region'
    }
}

# Add Hofstede cultural values [0-100 scale]
cultural_values['Your_Country'] = [PDI, IDV, MAS, UAI, LTO, IVR]
```

### Custom Dimensions

Extend the IusSpace with new dimensions:

```python
# Add to framework dimensions
custom_dimensions = [
    'digital_governance',
    'environmental_law_integration',
    'international_cooperation',
    'transparency_measures'
]

framework.dimensions.extend(custom_dimensions)
```

### Analysis Parameters

Modify key parameters:

```python
# Evolution simulation
mutation_rate = 0.05      # Default: 0.1 (higher = more variation)
generations = 100         # Default: 50 (longer = more evolution)
population_size = 50      # For genetic algorithm extensions

# Statistical validation  
bootstrap_samples = 1000  # Default: 500 (more = better confidence intervals)
confidence_level = 0.95   # Default: 0.95 (95% confidence intervals)

# Power-law fitting
xmin_threshold = 1        # Minimum value for power-law fitting
gamma_expected = 2.3      # Theoretical expected value
```

## 📈 Performance Optimization

### Memory Management

For large datasets:
```python
# Clear variables when not needed
del large_dataset
import gc; gc.collect()

# Use generators for large loops
innovations = (process_innovation(i) for i in innovation_list)
```

### Computation Speed

Speed up analysis:
```python
# Use vectorized operations
numpy_operations = np.vectorize(your_function)
results = numpy_operations(data_array)

# Parallel processing for bootstrap
from joblib import Parallel, delayed
bootstrap_results = Parallel(n_jobs=-1)(delayed(bootstrap_sample)() for _ in range(n_bootstrap))
```

### GPU Acceleration (If Available)

```python
# Check GPU availability
import tensorflow as tf
print("GPU Available: ", tf.config.list_physical_devices('GPU'))

# Use GPU for large matrix operations when available
if tf.config.list_physical_devices('GPU'):
    # GPU-accelerated computations
    pass
```

## 🔍 Troubleshooting

### Common Issues

**1. Installation Errors**
```bash
# If packages fail to install, try:
!pip install --upgrade pip
!pip install --force-reinstall package_name
```

**2. Memory Errors**
```python
# Reduce dataset size for testing
test_innovations = legal_innovations.sample(n=100)
test_countries = countries_data.head(3)
```

**3. Visualization Issues**
```python
# If plots don't display, try:
%matplotlib inline
import matplotlib.pyplot as plt
plt.ion()  # Turn on interactive mode
```

**4. Git Clone Issues**
```bash
# Alternative download method
!wget https://github.com/usuario/iusmorfos_public/archive/main.zip
!unzip main.zip
!mv iusmorfos_public-main iusmorfos_public
```

### Getting Help

1. **Check Error Messages**: Read the full error traceback
2. **Restart Runtime**: Runtime → Restart runtime (clears all variables)
3. **Clear Outputs**: Edit → Clear all outputs
4. **Reset Environment**: Runtime → Factory reset runtime

## 💾 Exporting Results

### Download Files

```python
# Export key results
from google.colab import files

# Download individual files
files.download('results/analysis_report.md')
files.download('results/countries_data.csv')
files.download('results/legal_innovations.csv')

# Create and download zip archive
!zip -r results.zip results/
files.download('results.zip')
```

### Save to Google Drive

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp -r results/ "/content/drive/My Drive/Iusmorfos_Analysis/"
```

### Export Notebooks

1. **File → Download .ipynb**: Download notebook with results
2. **File → Download .py**: Download as Python script  
3. **File → Print**: Generate PDF version
4. **Edit → Clear all outputs**: Clean notebook before sharing

## 🔬 Advanced Usage

### Custom Analysis Functions

Create your own analysis functions:

```python
def custom_legal_metric(country_data, innovation_data):
    """
    Create custom metrics for legal system analysis
    """
    # Your analysis logic here
    return custom_metric

# Apply to all countries
results = {}
for country in framework.countries_data['country']:
    country_innovations = framework.legal_innovations  # Filter as needed
    country_genes = framework.countries_data[framework.countries_data['country'] == country]
    results[country] = custom_legal_metric(country_genes, country_innovations)
```

### Statistical Extensions

```python
# Add new statistical tests
from scipy import stats

def advanced_validation(data1, data2):
    """Advanced statistical comparison"""
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.ks_2samp(data1, data2)
    
    # Mann-Whitney U test
    u_stat, u_p = stats.mannwhitneyu(data1, data2)
    
    # Effect size (Cohen's d)
    cohens_d = (np.mean(data1) - np.mean(data2)) / np.sqrt(((len(data1)-1)*np.var(data1) + (len(data2)-1)*np.var(data2))/(len(data1)+len(data2)-2))
    
    return {
        'ks_statistic': ks_stat,
        'ks_p_value': ks_p, 
        'mannwhitney_u': u_stat,
        'mannwhitney_p': u_p,
        'effect_size': cohens_d
    }
```

### Machine Learning Integration

```python
# Predictive modeling
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score

# Prepare features (cultural + legal dimensions)
X = framework.countries_data[framework.cultural_dims + framework.dimensions]
y = framework.countries_data['some_target_variable']

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=5)

print(f"Cross-validation R²: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
```

## 🎓 Educational Use

### For Students

**Beginner Level**:
1. Run the complete notebook to see results
2. Modify country selections in interactive widgets
3. Change visualization parameters
4. Export and analyze basic results

**Intermediate Level**:
1. Modify evolution simulation parameters
2. Add new countries with cultural data
3. Create custom visualizations
4. Extend statistical analyses

**Advanced Level**:
1. Implement new theoretical models
2. Add machine learning components
3. Integrate external legal databases
4. Develop new validation methods

### For Instructors

**Classroom Integration**:
- Use as interactive teaching tool for computational law
- Assign parameter modification exercises
- Create comparative projects between student groups
- Develop extensions as final projects

**Assessment Ideas**:
- Compare students' analysis of different countries
- Evaluate understanding through parameter sensitivity analysis
- Test interpretation of statistical results
- Assess ability to extend the framework

## 📚 Further Reading

### Academic References

1. **Dawkins, R.** (1986). *The Blind Watchmaker*. Norton & Company.
2. **Hofstede, G.** (2001). *Culture's Consequences*. Sage Publications.
3. **Clauset, A., et al.** (2009). Power-law distributions in empirical data. *SIAM Review*, 51(4), 661-703.
4. **Newman, M.E.J.** (2005). Power laws, Pareto distributions and Zipf's law. *Contemporary Physics*, 46(5), 323-351.

### Technical Resources

- **NumPy Documentation**: https://numpy.org/doc/
- **Pandas Guide**: https://pandas.pydata.org/docs/
- **Matplotlib Tutorials**: https://matplotlib.org/stable/tutorials/
- **SciPy Statistical Functions**: https://docs.scipy.org/doc/scipy/reference/stats.html
- **NetworkX Graph Analysis**: https://networkx.org/documentation/stable/

### Legal Informatics

- **Computational Legal Studies**: Methods and applications
- **Legal Data Science**: Statistical approaches to law
- **Comparative Legal Systems**: Cross-country methodologies  
- **Legal Evolution Theory**: Theoretical frameworks

## 🆘 Support

### Community Resources

- **GitHub Issues**: Report bugs and request features
- **Discussions**: Ask questions and share insights
- **Wiki**: Community-contributed documentation
- **Examples**: Additional use cases and implementations

### Contact Information

For technical support or collaboration inquiries:
- **Repository**: https://github.com/usuario/iusmorfos_public
- **Documentation**: See `docs/` directory for detailed guides
- **Issues**: Use GitHub Issues for bug reports
- **Features**: Submit feature requests via GitHub

---

**📄 License**: This guide is part of the Iusmorfos Framework, released under MIT License.

**🔄 Last Updated**: 2024-01-15

**✨ Version**: 1.0 - Initial comprehensive guide