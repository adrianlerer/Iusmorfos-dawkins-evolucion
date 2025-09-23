# Contributing to Iusmorfos 🤝

Thank you for your interest in contributing to the Iusmorfos project! This document provides comprehensive guidelines for maintaining our world-class reproducibility standards while enabling collaborative development.

## 🎯 Our Standards

The Iusmorfos project adheres to **gold-standard reproducibility** practices. All contributions must maintain:

- ✅ **100% Computational Reproducibility** (Docker + fixed seeds)
- ✅ **95%+ Test Coverage** for all new code
- ✅ **Statistical Robustness** through bootstrap validation
- ✅ **FAIR Compliance** (Findable, Accessible, Interoperable, Reusable)
- ✅ **Cross-Country Generalizability** validation when applicable

## 🚀 Quick Start for Contributors

### 1. Development Environment Setup

```bash
# Fork and clone the repository
git clone https://github.com/YOUR-USERNAME/Iusmorfos-dawkins-evolucion.git
cd Iusmorfos-dawkins-evolucion/iusmorfos_public

# Set up development environment (Docker recommended)
docker build -t iusmorfos-dev -f Dockerfile.dev .
docker run -it -v $(pwd):/workspace iusmorfos-dev

# Or use conda environment
conda env create -f environment.yml
conda activate iusmorfos-dev
pip install -r requirements-dev.txt
```

### 2. Pre-Commit Setup

```bash
# Install pre-commit hooks for quality assurance
pip install pre-commit
pre-commit install

# Verify setup
pre-commit run --all-files
```

### 3. Run Full Test Suite

```bash
# Ensure all tests pass before starting
python -m pytest tests/ -v --cov=src/ --cov-report=html
python scripts/validate_reproducibility.py
```

## 📋 Contribution Types

### 🔬 Scientific Contributions (High Impact)

#### New Country Validation
- **Requirements**: Legal tradition not yet covered (Religious Law, Customary Law)
- **Data Standards**: ≥300 innovations, 1980+ temporal coverage
- **Validation Criteria**: R² > 0.6, transferability score > 0.7
- **Documentation**: Complete cultural dimension profiling required

#### Statistical Method Improvements
- **Focus Areas**: Robustness testing, power-law validation, bootstrap methods
- **Requirements**: Peer-reviewed methodology, simulation validation
- **Integration**: Must maintain backward compatibility with existing results

#### Cross-Cultural Adaptation Mechanisms  
- **Objective**: Improve framework performance in culturally distant countries
- **Requirements**: Theoretical justification + empirical validation
- **Target**: India (current marginal performance) and future expansions

### 💻 Technical Contributions (Medium Impact)

#### Performance Optimization
- **Targets**: Large dataset processing (>10k innovations), parallel computation
- **Requirements**: Benchmark improvements, memory efficiency
- **Validation**: Performance tests + regression tests must pass

#### Infrastructure Improvements
- **Areas**: CI/CD enhancement, container optimization, dependency management
- **Requirements**: Maintain reproducibility, improve developer experience
- **Testing**: All platforms (Linux/macOS/Windows) compatibility

#### API and Usability Enhancements
- **Focus**: Research-friendly interfaces, notebook improvements
- **Requirements**: Maintain scientific rigor, comprehensive documentation
- **Integration**: Backward compatibility essential

### 📚 Documentation Contributions (Essential)

#### Tutorial Development
- **Needed**: Discipline-specific guides (Law, Political Science, Sociology)
- **Format**: Jupyter notebooks with executable examples
- **Quality**: Peer-reviewable content, reproducible outputs

#### Translation and Localization
- **Priority Languages**: Spanish, Portuguese, French (legal research communities)
- **Requirements**: Technical accuracy, cultural adaptation
- **Validation**: Native speaker review required

## 🔬 Scientific Rigor Requirements

### Statistical Contributions Checklist

- [ ] **Methodology Validation**
  - [ ] Theoretical justification with citations
  - [ ] Simulation studies demonstrating validity
  - [ ] Comparison with established methods
  - [ ] Edge case analysis and robustness testing

- [ ] **Implementation Quality**
  - [ ] Unit tests covering all statistical functions
  - [ ] Integration tests with existing framework
  - [ ] Performance benchmarks
  - [ ] Memory usage analysis

- [ ] **Documentation Standards**
  - [ ] Mathematical notation following conventions
  - [ ] Algorithm complexity analysis
  - [ ] Parameter sensitivity discussion
  - [ ] Limitations and assumptions clearly stated

### Data Contributions Checklist

- [ ] **Data Quality**
  - [ ] Complete provenance documentation
  - [ ] Data validation and quality checks implemented
  - [ ] Missing data patterns analyzed and reported
  - [ ] Outlier detection and treatment documented

- [ ] **Metadata Standards**
  - [ ] Dublin Core compliance for discoverability
  - [ ] Temporal and spatial coverage clearly defined
  - [ ] Data collection methodology documented
  - [ ] Legal and ethical compliance verified

- [ ] **Reproducibility**
  - [ ] Processing scripts included and documented
  - [ ] Checksums provided for data integrity
  - [ ] Version control for data updates
  - [ ] Dependency tracking for processing tools

## 🧪 Testing Standards

### Mandatory Testing Requirements

#### 1. Unit Tests (>95% Coverage Required)
```python
# Example test structure
def test_power_law_estimation():
    \"\"\"Test power-law parameter estimation accuracy.\"\"\"
    # Generate known power-law data
    true_gamma = 2.3
    data = generate_power_law_data(gamma=true_gamma, n=1000, seed=42)
    
    # Estimate parameters
    estimated_gamma = estimate_power_law_gamma(data)
    
    # Validate accuracy
    assert abs(estimated_gamma - true_gamma) < 0.1
    assert 2.0 <= estimated_gamma <= 3.0  # Reasonable bounds
```

#### 2. Integration Tests
```python
def test_full_country_validation_pipeline():
    \"\"\"Test complete validation workflow.\"\"\"
    # Test with synthetic country data
    validator = ExternalValidationFramework()
    synthetic_data = validator.generate_synthetic_country_data('TEST', n=100)
    
    result = validator.validate_country_data('TEST', synthetic_data)
    
    # Validate pipeline completion
    assert 'r2_score' in result.metrics
    assert 'transferability_score' in result.metrics
    assert result.validation_completed == True
```

#### 3. Regression Tests (Critical)
```python
def test_argentina_baseline_regression():
    \"\"\"Ensure Argentina results remain stable.\"\"\"
    # Load historical results
    baseline_results = load_baseline_results('argentina_validation_v1.0.json')
    
    # Run current validation
    current_results = run_argentina_validation()
    
    # Compare with tolerance
    assert abs(current_results.r2_score - baseline_results.r2_score) < 0.01
    assert current_results.power_law_gamma == pytest.approx(
        baseline_results.power_law_gamma, abs=0.05
    )
```

#### 4. Reproducibility Tests
```python
def test_deterministic_reproducibility():
    \"\"\"Verify identical results across runs.\"\"\"
    results_1 = run_experiment_with_seed(seed=42)
    results_2 = run_experiment_with_seed(seed=42) 
    
    # Results must be identical, not just similar
    assert results_1 == results_2
    assert hash(str(results_1)) == hash(str(results_2))
```

### Performance Testing

```python
@pytest.mark.performance
def test_large_dataset_performance():
    \"\"\"Ensure reasonable performance with large datasets.\"\"\"
    large_dataset = generate_synthetic_data(n_countries=10, n_innovations=5000)
    
    start_time = time.time()
    results = run_cross_country_validation(large_dataset)
    execution_time = time.time() - start_time
    
    # Performance requirements
    assert execution_time < 300  # Max 5 minutes
    assert results.memory_usage < 2048  # Max 2GB RAM
```

## 📊 Code Quality Standards

### Python Code Standards

#### Style and Formatting
```python
# Use Black formatter (automated)
black --line-length=88 src/ tests/

# Use isort for imports
isort src/ tests/

# Use flake8 for linting
flake8 src/ tests/ --max-line-length=88
```

#### Documentation Requirements
```python
def validate_power_law_distribution(
    data: np.ndarray,
    theoretical_gamma: float = 2.3,
    confidence_level: float = 0.95
) -> PowerLawValidationResult:
    \"\"\"
    Validate power-law distribution hypothesis for citation network data.
    
    Uses maximum likelihood estimation with Kolmogorov-Smirnov goodness-of-fit
    testing to validate power-law distribution hypothesis against empirical data.
    
    Parameters
    ----------
    data : np.ndarray
        Empirical citation count data (positive integers)
    theoretical_gamma : float, default=2.3
        Expected power-law exponent from theoretical predictions
    confidence_level : float, default=0.95  
        Confidence level for statistical tests
        
    Returns
    -------
    PowerLawValidationResult
        Validation results including:
        - estimated_gamma: MLE estimate of power-law exponent
        - ks_statistic: Kolmogorov-Smirnov test statistic
        - p_value: Statistical significance of fit
        - fits_power_law: Boolean validation result
        - confidence_interval: Bootstrap CI for gamma estimate
        
    Raises
    ------
    ValueError
        If data contains non-positive values or is too small (n < 50)
        
    Notes  
    -----
    Implementation follows Clauset, Shalizi & Newman (2009) methodology
    for power-law fitting and validation. Bootstrap confidence intervals
    use 1000 resamples for robust uncertainty quantification.
    
    References
    ----------
    .. [1] Clauset, A., Shalizi, C. R., & Newman, M. E. (2009). Power-law 
           distributions in empirical data. SIAM review, 51(4), 661-703.
           
    Examples
    --------
    >>> citation_data = np.array([1, 2, 1, 5, 3, 8, 2, 1, 4, 12])
    >>> result = validate_power_law_distribution(citation_data)
    >>> print(f"Estimated gamma: {result.estimated_gamma:.3f}")
    >>> print(f"Fits power-law: {result.fits_power_law}")
    \"\"\"
```

#### Type Hints (Required)
```python
from typing import Dict, List, Optional, Union, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class ValidationResult:
    \"\"\"Results from cross-country validation experiment.\"\"\"
    country_code: str
    r2_score: float
    transferability_score: float
    cultural_adaptation_score: float
    validation_passed: bool
    error_message: Optional[str] = None
    
def process_country_data(
    raw_data: pd.DataFrame,
    country_profile: CountryProfile,
    validation_parameters: Dict[str, Union[int, float]]
) -> Tuple[pd.DataFrame, ValidationResult]:
    \"\"\"Process and validate country-specific legal innovation data.\"\"\"
```

### Statistical Computing Best Practices

#### Numerical Stability
```python
# Use numerically stable algorithms
def safe_log_likelihood(data: np.ndarray, gamma: float) -> float:
    \"\"\"Compute log-likelihood with numerical stability.\"\"\"
    # Avoid log(0) and overflow/underflow
    data_clean = data[data > 0]
    if len(data_clean) == 0:
        return -np.inf
        
    log_data = np.log(data_clean)
    return (
        len(data_clean) * np.log(gamma - 1) - 
        gamma * np.sum(log_data) - 
        np.sum(log_data)
    )
```

#### Reproducible Random Number Generation
```python
def generate_bootstrap_samples(
    data: np.ndarray, 
    n_samples: int = 1000,
    random_state: Optional[int] = None
) -> List[np.ndarray]:
    \"\"\"Generate bootstrap samples with reproducible random state.\"\"\"
    rng = np.random.RandomState(random_state)
    n_data = len(data)
    
    return [
        data[rng.choice(n_data, size=n_data, replace=True)]
        for _ in range(n_samples)
    ]
```

## 🌍 Cross-Country Validation Contributions

### Adding New Countries

#### Requirements Checklist
- [ ] **Legal System Coverage**
  - [ ] Represents legal tradition not yet covered OR
  - [ ] Provides significant cultural/economic diversity within existing tradition

- [ ] **Data Requirements** 
  - [ ] Minimum 300 legal innovations
  - [ ] Temporal coverage ≥20 years (preferably 1980+)
  - [ ] Multiple reform types represented
  - [ ] Crisis periods documented

- [ ] **Metadata Requirements**
  - [ ] Cultural dimensions (Hofstede or equivalent)
  - [ ] Economic development indicators (GDP per capita, governance index)
  - [ ] Legal system classification and historical context
  - [ ] Crisis period timeline with severity ratings

#### Implementation Steps

1. **Create Country Profile**
```python
# Add to src/external_validation.py
new_country_profile = CountryProfile(
    code='BR',  # ISO 3166-1 alpha-2 code
    name='Brazil',
    legal_system=LegalSystem.CIVIL_LAW,
    gdp_per_capita=9673,  # 2022 data
    population_millions=215.3,
    governance_index=0.49,  # World Bank WGI
    cultural_dimensions={
        'power_distance': 69,
        'individualism': 38,
        'masculinity': 49,
        'uncertainty_avoidance': 76,
        'long_term_orientation': 44
    },
    expected_patterns={
        'reform_frequency': 'high',
        'adoption_speed': 'slow', 
        'complexity_preference': 'high',
        'crisis_response': 'institutional'
    }
)
```

2. **Implement Data Generation**
```python
def generate_brazil_specific_data(self, n_innovations: int = 500) -> pd.DataFrame:
    \"\"\"Generate Brazil-specific legal innovation data.\"\"\"
    # Implement country-specific patterns based on legal tradition,
    # economic development, cultural dimensions, and historical context
```

3. **Validation Testing**
```python
def test_brazil_validation():
    \"\"\"Test Brazil validation meets quality standards.\"\"\"
    validator = ExternalValidationFramework()
    brazil_data = validator.generate_synthetic_country_data('BR')
    result = validator.validate_argentina_model_on_country('BR', argentina_params)
    
    # Quality thresholds
    assert result['performance_metrics']['r2_score'] > 0.5  # Minimum acceptable
    assert result['transferability_metrics']['overall_transferability_score'] > 0.6
    assert 'cultural_adaptation' in result
```

### Cultural Adaptation Research

Priority research areas for improving cross-cultural transferability:

#### 1. Legal Tradition Adaptation Mechanisms
- **Civil Law → Common Law**: Precedent weighting adjustments
- **Mixed Systems**: Hybrid model architectures  
- **Religious/Customary Law**: Cultural value integration

#### 2. Economic Development Corrections
- **Development Stage Effects**: Resource availability impact on innovation
- **Institutional Capacity**: Governance quality moderation effects
- **Technology Adoption**: Digital transformation patterns

#### 3. Cultural Distance Bridging
- **Hofstede Dimension Corrections**: Power distance and uncertainty avoidance effects
- **Social Trust Metrics**: Collective action capacity measurements
- **Historical Path Dependencies**: Colonial and legal origin effects

## 🔄 Pull Request Process

### Pre-Submission Checklist

#### Code Quality
- [ ] All pre-commit hooks pass (formatting, linting, type checking)
- [ ] Test coverage ≥95% for new code, >90% overall maintained  
- [ ] All regression tests pass without modification
- [ ] Performance tests show no degradation (>10% slowdown requires justification)

#### Documentation
- [ ] Docstrings follow NumPy/SciPy format with complete parameter descriptions
- [ ] README updated for new functionality (if applicable)
- [ ] Jupyter notebooks run without errors and produce expected outputs
- [ ] Mathematical notation properly formatted (LaTeX in markdown cells)

#### Scientific Validity
- [ ] Statistical methods peer-reviewable (citations to methodology papers)
- [ ] Assumptions clearly stated and validated
- [ ] Edge cases identified and handled appropriately
- [ ] Parameter sensitivity analysis completed (if applicable)

#### Reproducibility
- [ ] All random processes use fixed seeds from configuration
- [ ] Docker container builds successfully with new dependencies
- [ ] Cross-platform compatibility verified (Linux/macOS/Windows)
- [ ] New dependencies justified and minimally required versions specified

### Submission Process

1. **Create Feature Branch**
```bash
# Use descriptive branch names
git checkout -b feature/brazil-country-validation
git checkout -b fix/power-law-estimation-edge-cases  
git checkout -b docs/spanish-tutorial-notebooks
```

2. **Commit Standards**
```bash
# Use conventional commits format
git commit -m "feat(validation): add Brazil country profile with cultural adaptation"
git commit -m "fix(statistics): handle edge case in power-law MLE estimation"  
git commit -m "docs(notebooks): add Spanish translation of EDA tutorial"
git commit -m "test(robustness): add bootstrap validation for new statistics"
```

3. **Pull Request Template**

Use our template to ensure all requirements are addressed:

```markdown
## 🎯 Pull Request Summary

**Type**: [Feature/Fix/Documentation/Performance/Refactor]
**Impact**: [High/Medium/Low]
**Breaking Changes**: [Yes/No]

### 📋 Changes Made
- [ ] Brief description of main changes
- [ ] List of new functions/classes added
- [ ] Modified algorithms or statistical methods
- [ ] Updated documentation or notebooks

### 🔬 Scientific Validation
- [ ] Methodology peer-reviewable (include citations)
- [ ] Statistical tests validate new/modified methods
- [ ] Cross-validation results demonstrate robustness
- [ ] Parameter sensitivity analysis completed

### 🧪 Testing
- [ ] All existing regression tests pass
- [ ] New unit tests added with >95% coverage
- [ ] Integration tests validate end-to-end workflows  
- [ ] Performance tests show no degradation

### 📚 Documentation
- [ ] Docstrings complete with examples
- [ ] README updated (if needed)
- [ ] Notebooks run without errors
- [ ] Mathematical notation properly formatted

### 🔄 Reproducibility  
- [ ] Fixed seeds used for all random processes
- [ ] Docker container builds successfully
- [ ] Cross-platform compatibility verified
- [ ] Dependencies minimal and version-pinned

### 🌍 Cross-Country Impact (if applicable)
- [ ] New country validation meets quality thresholds (R² > 0.6)
- [ ] Cultural adaptation mechanisms implemented
- [ ] Transferability analysis completed
- [ ] Impact on overall framework generalizability assessed
```

4. **Review Process**

Your PR will be reviewed for:

- **Scientific Rigor**: Methodology, statistical validity, reproducibility
- **Code Quality**: Style, performance, maintainability, test coverage
- **Documentation**: Completeness, accuracy, clarity
- **Integration**: Compatibility with existing framework, backward compatibility

Expect 1-2 review cycles for substantial contributions. Reviewers may request:
- Additional statistical validation
- Performance benchmarks
- Documentation improvements  
- Test coverage expansion

## 🏆 Recognition and Attribution

### Contributor Recognition

All contributors are recognized in multiple ways:

#### 1. **Git History and GitHub**
- All contributions permanently recorded in git history
- GitHub contributor statistics and profiles
- Issue and PR authorship maintained

#### 2. **Academic Citations**  
- Substantial scientific contributions eligible for co-authorship on papers
- Software contributions cited in academic publications
- Individual contributor ORCID integration

#### 3. **Release Notes**
- Major contributors highlighted in version release notes
- Feature additions attributed to authors
- Maintenance and bug fix contributions acknowledged

### Contribution Categories

#### 🥇 **Gold Contributors** (Co-authorship eligible)
- New country validations with complete cultural analysis
- Novel statistical methods with peer-review quality validation
- Major architectural improvements with demonstrated impact

#### 🥈 **Silver Contributors** (Acknowledgment in papers)
- Significant performance improvements or bug fixes
- Comprehensive documentation or tutorial development
- Substantial testing infrastructure enhancements

#### 🥉 **Bronze Contributors** (Release notes recognition)
- Bug fixes and minor feature additions
- Documentation improvements and translations
- Test additions and quality improvements

## 📞 Getting Help

### Communication Channels

#### 🐛 **Issues and Bug Reports**
- Use GitHub Issues with appropriate labels
- Provide minimal reproducible examples
- Include system information and error traces
- Follow issue template for completeness

#### 💡 **Feature Requests and Discussions** 
- Use GitHub Discussions for ideas and proposals
- Provide scientific justification for new features
- Consider implementation complexity and maintenance burden
- Link to relevant literature or methodological papers

#### 🤝 **Collaboration Inquiries**
- Reach out via GitHub for research collaborations  
- Academic partnerships welcome for cross-country validation
- Statistical methodology collaboration opportunities
- Institutional support for large-scale validation projects

#### 📚 **Academic Questions**
- Open GitHub Issues with "Research" label for methodological questions
- Link to relevant academic literature in discussions
- Provide context for intended use in research projects
- Consider co-development for novel applications

### Response Time Expectations
- **Bug reports**: 48-72 hours initial response
- **Feature requests**: 1 week for initial feedback  
- **Pull requests**: 1-2 weeks for review completion
- **Academic inquiries**: 1 week for research collaboration discussion

## 📄 Legal and Ethical Considerations

### Intellectual Property
- All contributions under MIT License (same as main project)
- Contributors retain copyright to their contributions  
- Academic use requires appropriate citation
- Commercial use permitted with attribution

### Data Ethics
- Only use publicly available or properly licensed legal data
- Respect privacy and confidentiality requirements
- Follow institutional ethics guidelines for research data
- Document data sources and processing procedures

### Academic Integrity
- Properly cite all methodological sources and inspirations
- Acknowledge prior work and existing implementations
- Maintain scientific rigor and reproducibility standards
- Avoid plagiarism or inappropriate use of others' work

---

## 🙏 Thank You

Thank you for contributing to advancing reproducible computational social science! Your contributions help establish new standards for cross-country validation and institutional evolution research.

Together, we're building a framework that demonstrates how legal systems evolve according to discoverable patterns, providing evidence-based foundations for institutional design and comparative law research.

**Happy Contributing! 🚀**

---

*For additional questions or clarifications, please open a GitHub Issue or Discussion. We're committed to supporting contributors in maintaining our world-class reproducibility standards.*