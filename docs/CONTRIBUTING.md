# Contributing to Iusmorfos

Thank you for your interest in contributing to the Iusmorfos project! This document provides guidelines for contributing to this research software.

## 🎯 Ways to Contribute

### 1. Research Extensions
- **New Dimensions**: Add institutional dimensions beyond the current 9
- **Cross-Cultural Analysis**: Extend to non-Western legal traditions
- **Historical Validation**: Apply framework to documented legal evolution
- **Multi-Agent Systems**: Implement co-evolving legal systems

### 2. Technical Improvements
- **Performance Optimization**: Faster algorithms, better memory usage
- **Visualization Enhancement**: Better jusmorph representations
- **User Interface**: GUI for interactive evolution
- **API Development**: REST API for web applications

### 3. Documentation
- **Tutorial Development**: Step-by-step guides for specific use cases
- **Academic Examples**: More replication scenarios
- **Methodology Clarification**: Detailed algorithmic explanations
- **Translation**: Documentation in other languages

### 4. Validation & Testing
- **Dataset Expansion**: More countries, legal families, time periods
- **Parameter Sensitivity**: Systematic analysis of parameter effects
- **Alternative Metrics**: Different fitness functions, complexity measures
- **Robustness Testing**: Edge cases, error conditions

## 📋 Development Guidelines

### Code Style
- **Python**: Follow PEP 8 style guide
- **Documentation**: Comprehensive docstrings for all functions
- **Type Hints**: Use type annotations where appropriate
- **Comments**: Explain complex algorithmic sections

### Testing
- **Unit Tests**: Test individual functions and components
- **Integration Tests**: Test complete experimental pipelines
- **Validation Tests**: Compare against known results
- **Performance Tests**: Benchmark execution time and memory

### Academic Standards
- **Replicability**: All code must be reproducible
- **Documentation**: Clear methodology descriptions
- **Citations**: Proper attribution of sources and inspirations
- **Validation**: Empirical support for claims

## 🚀 Getting Started

### 1. Fork and Clone
```bash
# Fork the repository on GitHub
# Clone your fork
git clone https://github.com/YOUR_USERNAME/Iusmorfos-dawkins-evolucion.git
cd Iusmorfos-dawkins-evolucion
```

### 2. Development Environment
```bash
# Create virtual environment
python -m venv dev_env
source dev_env/bin/activate  # Linux/Mac
# dev_env\Scripts\activate    # Windows

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 sphinx
```

### 3. Create Feature Branch
```bash
# Create branch for your contribution
git checkout -b feature/your-feature-name
```

### 4. Development Workflow
```bash
# Make changes
# Add tests
# Run tests
pytest

# Check code style
black src/
flake8 src/

# Commit changes
git add .
git commit -m "feat: add your feature description"

# Push to your fork
git push origin feature/your-feature-name
```

### 5. Submit Pull Request
- Create pull request from your fork
- Provide clear description of changes
- Include tests and documentation updates
- Reference any related issues

## 📊 Research Contributions

### Proposing New Research Directions

If you want to propose a significant research extension:

1. **Open an Issue**: Describe the research question and approach
2. **Literature Review**: Reference relevant academic work
3. **Methodology**: Outline proposed methods and validation
4. **Expected Impact**: Explain potential contributions

### Academic Collaboration

For academic collaborations:

1. **Contact Author**: Email for complex research questions
2. **Research Proposal**: Formal proposal for significant extensions
3. **Co-authorship**: Discuss authorship for substantial contributions
4. **Publication**: Coordinate publication strategies

## 🔬 Specific Contribution Areas

### High Priority
- [ ] **Multi-Legal-Family Evolution**: Evolve Civil Law, Common Law, Religious Law simultaneously
- [ ] **Crisis Response Modeling**: How legal systems adapt to external shocks
- [ ] **Globalization Effects**: Impact of international law on domestic evolution
- [ ] **Digital Transformation**: Evolution of legal systems in digital age

### Medium Priority  
- [ ] **Alternative Fitness Functions**: Economic efficiency, social welfare, etc.
- [ ] **Visualization Improvements**: 3D space representation, interactive evolution
- [ ] **Performance Optimization**: Parallel processing, GPU acceleration
- [ ] **User Interface**: Web-based interactive evolution experiments

### Documentation Needs
- [ ] **API Documentation**: Complete function reference
- [ ] **Tutorial Series**: Step-by-step research guides
- [ ] **Case Studies**: Applied examples in different legal contexts
- [ ] **Methodology Guide**: Detailed algorithmic explanations

## 🧪 Testing Guidelines

### Required Tests for New Features

#### Unit Tests
```python
def test_new_feature():
    """Test individual function behavior."""
    # Setup
    input_data = create_test_data()
    expected_output = calculate_expected_result()
    
    # Execute
    result = new_feature_function(input_data)
    
    # Assert
    assert result == expected_output
    assert result.meets_quality_criteria()
```

#### Integration Tests
```python
def test_feature_integration():
    """Test feature works within complete system."""
    simulador = SimuladorBiomorfosMejorado()
    # Configure with new feature
    resultado = simulador.ejecutar_experimento_mejorado(10)
    
    # Verify system still works
    assert resultado['generaciones_completadas'] == 10
    assert 'new_feature_results' in resultado
```

#### Validation Tests
```python
def test_empirical_validation():
    """Test against known empirical results."""
    # Use historical legal evolution data
    # Verify new feature improves or maintains accuracy
    validation_score = validate_against_real_data()
    assert validation_score >= MINIMUM_ACCURACY_THRESHOLD
```

## 📝 Documentation Standards

### Code Documentation
```python
def evolve_legal_system(initial_system: GenLegal, 
                       generations: int,
                       fitness_params: Dict[str, float]) -> Dict[str, Any]:
    """
    Evolve a legal system through specified generations.
    
    This function implements the core evolution algorithm based on Dawkins'
    biomorphs methodology, adapted for institutional evolution.
    
    Args:
        initial_system: Starting legal system configuration
        generations: Number of evolutionary generations to run
        fitness_params: Dictionary with fitness function weights
                       {'complexity': float, 'diversity': float, 'balance': float}
    
    Returns:
        Dictionary containing:
            - 'final_system': Evolved legal system
            - 'evolution_history': Generation-by-generation changes
            - 'emergent_families': Legal families that emerged
            - 'validation_metrics': Empirical validation results
    
    Raises:
        ValueError: If fitness parameters don't sum to 1.0
        RuntimeError: If evolution fails to converge
    
    Example:
        >>> initial = GenLegal(1, 1, 1, 1, 1, 1, 1, 1, 1)
        >>> params = {'complexity': 0.4, 'diversity': 0.3, 'balance': 0.3}
        >>> result = evolve_legal_system(initial, 30, params)
        >>> print(f"Final complexity: {result['final_system'].complejidad}")
    
    References:
        Dawkins, R. (1986). The Blind Watchmaker, Chapter 3.
    """
```

### Research Documentation
- **Methodology**: Detailed algorithmic descriptions
- **Assumptions**: Clearly state theoretical assumptions
- **Limitations**: Acknowledge scope and constraints
- **Validation**: Describe empirical testing approach
- **Future Work**: Suggest research extensions

## 🤝 Community Guidelines

### Communication
- **Be Respectful**: Academic discourse, constructive feedback
- **Be Precise**: Clear technical descriptions
- **Be Open**: Acknowledge limitations and uncertainties
- **Be Collaborative**: Support other contributors' work

### Issue Reporting
When reporting bugs or requesting features:

1. **Check Existing Issues**: Avoid duplicates
2. **Provide Context**: System info, use case, expected behavior
3. **Include Code**: Minimal reproducible example
4. **Add Labels**: Bug, feature, documentation, etc.

### Code Review Process
- **Technical Review**: Code quality, performance, correctness
- **Academic Review**: Methodological soundness, validation
- **Documentation Review**: Clarity, completeness, accuracy
- **Testing Review**: Coverage, edge cases, integration

## 📚 Resources

### Academic Background
- [Dawkins, R. (1986). The Blind Watchmaker](https://en.wikipedia.org/wiki/The_Blind_Watchmaker)
- [Watson, A. (1993). Legal Transplants](https://press.uchicago.edu/ucp/books/book/chicago/L/bo3684040.html)
- [Bommarito & Katz (2014). Legal Complexity](https://link.springer.com/article/10.1007/s10506-014-9160-8)

### Technical Resources
- [Python Development Guide](https://devguide.python.org/)
- [Scientific Python Ecosystem](https://www.scipy.org/)
- [Computational Legal Studies](https://computationallegalstudies.com/)

### Project-Specific
- [Methodology Documentation](docs/methodology.md)
- [API Reference](docs/API.md)
- [Replication Guide](docs/REPLICATION.md)

## 📧 Contact

For questions about contributing:
- **GitHub Issues**: Technical questions, bug reports
- **Email Author**: Complex research collaborations
- **Academic Forums**: Methodological discussions

---

**Thank you for contributing to advancing computational legal studies!**

*Together, we can build better tools for understanding institutional evolution and supporting evidence-based legal reform.*