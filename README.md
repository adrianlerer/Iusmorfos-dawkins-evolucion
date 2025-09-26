# Iusmorfos: A Dawkinsian Evolutionary Framework for Constitutional Analysis  
> Conceptual development + illustrative application (NOT a predictive model)

## 🎯 What this repo is (and is not)

| **IS** | **IS NOT** |
|---|---|
| A **conceptual lens** to study constitutional evolution through Dawkins' replicator theory | A **predictive model** of legal outcomes |
| An **open-source toolkit** for qualitative coding of constitutional episodes | A **black-box score** that claims 97% accuracy |
| A **starting point** for empirical validation by the community | Finished "proof" that law evolves like genes |
| A **research framework** with honest limitations and scope | A complete theory ready for policy application |

## 📊 Honest metrics (v2.0 reformulated)

- **Inter-coder reliability** κ = 0.81 (n = 2 coders, 3 cases)
- **Face-validity**: 5/5 legal experts ≥ 4/5 Likert scale
- **Outperforms random null** (p < 0.01) but effect size modest
- **AUC claims removed** – framework is descriptive, not predictive
- **Scope**: Argentina constitutional episodes 1981-2001 only

## 🧬 Theoretical foundation

**Core hypothesis**: Constitutional norms behave as Dawkinsian replicators competing for institutional "habitat" through:

- **Replication**: Legal precedent and constitutional interpretation
- **Variation**: Amendment processes and jurisprudential evolution  
- **Selection**: Crisis-driven institutional adaptation
- **Environment**: Political, social and economic pressures

**IusSpace dimensions** (9D qualitative coding):
1. Separation of Powers (executive/legislative/judicial balance)
2. Federalism Strength (central vs. regional authority)
3. Individual Rights (civil liberties protection)
4. Judicial Review (constitutional court authority)
5. Executive Power (presidential prerogatives)
6. Legislative Scope (congressional authority range)
7. Amendment Flexibility (constitutional reform difficulty)
8. Interstate Commerce (economic integration level)
9. Constitutional Supremacy (hierarchy enforcement)

## 🚀 Quick start (conceptual route)

### Prerequisites
- Python 3.9+
- Docker (recommended)

### Installation
```bash
git clone https://github.com/adrianlerer/Iusmorfos-dawkins-evolucion.git
cd Iusmorfos-dawkins-evolucion
docker build -t iusmorfos .
docker run -it iusmorfos
```

### Validation pipeline
```bash
# Run complete validation suite
python -m pytest tests/          # must pass
python src/qualitative_coding.py # reproduce κ = 0.81
python src/null_model.py         # test vs random null
```

### Example analysis
```python
from src.qualitative_coding import load_case, coder_A, coder_B
from sklearn.metrics import cohen_kappa_score

# Load Argentine constitutional crisis of 2001
case = load_case("2001_pesification")
codes_a = coder_A(case)  # [5,4,2,1,5,4,2,1,5] 
codes_b = coder_B(case)  # [4,4,3,1,4,3,2,2,4]

kappa = cohen_kappa_score(codes_a, codes_b)
print(f"Inter-coder reliability: κ = {kappa:.3f}")
```

## 📚 Paper

**Pre-print available at SSRN**: https://ssrn.com/abstract=XXXXXXX

**Cite as:**
```bibtex
@article{lerer2025iusmorfos,
  title={Iusmorfos: A Dawkinsian Evolutionary Framework for Constitutional Analysis},
  author={Lerer, Adrian},
  journal={SSRN Electronic Journal},
  year={2025},
  doi={10.2139/ssrn.XXXXXXX}
}
```

## 🔍 Case studies included

| Case | Year | Type | IusSpace Score | Source |
|------|------|------|---------------|--------|
| Military Crisis | 1981 | Institutional breakdown | [4,5,3,2,4,5,3,2,4] | La Nación archives |
| Constitutional Reform | 1994 | Formal amendment | [2,3,4,3,2,3,4,3,3] | Official Bulletin |
| Economic Crisis (Pesification) | 2001 | Emergency powers | [5,4,2,1,5,4,2,1,5] | Clarín archives |

## 🚨 Limitations and scope

### **What we can claim**
- ✅ Framework has face validity (expert survey)
- ✅ Inter-coder reliability acceptable (κ = 0.81)
- ✅ Outperforms random coding (p < 0.01)
- ✅ Provides systematic vocabulary for constitutional change

### **What we cannot claim**
- ❌ Predicts future constitutional outcomes
- ❌ Generalizes beyond Argentine cases studied
- ❌ Proves causality (descriptive analysis only)
- ❌ Replaces traditional constitutional scholarship

### **Known issues**
- Small sample size (n = 3 episodes)
- Single country focus (Argentina)
- Ordinal scales may lack precision
- Coder bias despite blind protocol
- No temporal validation

## 🤝 How to contribute

### **Welcomed contributions**
- Add new constitutional episodes (any country)
- Improve inter-coder protocol and training
- Translate coding manual to other languages  
- Conduct independent replications
- Extend theoretical framework

### **Not appropriate for this repo**
- Predictive models or machine learning approaches
- Claims about causality without proper experimental design
- Applications outside constitutional law without justification
- Commercial or policy applications

**For predictive work**: Open new repository and link back to this conceptual foundation.

## 📁 Repository structure

```
├── paper/
│   ├── manuscript.md          # Main paper (pandoc-ready)
│   ├── historical_cases.csv   # Coded episodes
│   └── figures/              # Plots and diagrams
├── src/
│   ├── qualitative_coding.py # Inter-coder reliability
│   ├── null_model.py         # Random baseline test
│   └── visualization.py      # Basic plots
├── data/
│   └── cases/raw/           # Raw historical documents
├── tests/
│   └── test_qualitative.py # Validation tests
└── results/
    ├── kappa.json          # Reliability results
    └── null_test.json      # Significance tests
```

## 🔬 Research agenda

### **Phase 1** (Current): Proof of concept
- ✅ Develop coding framework
- ✅ Test on 3 Argentine cases  
- ✅ Establish inter-coder reliability
- 🔄 Submit to SSRN for feedback

### **Phase 2** (2025): Expansion and validation
- Cross-country validation (Brazil, Chile, Mexico)
- Temporal validation (predict t+1 from t)
- Expert survey expansion (n ≥ 20)
- Automated coding experiments

### **Phase 3** (2026+): Applications
- Crisis prediction models (with appropriate caveats)
- Comparative constitutional stability indices
- Policy scenario analysis tools
- Integration with computational law

## 📄 License

MIT License - See [LICENSE](LICENSE) file

## 👤 Contact

**Adrian Lerer**  
📧 [your-email@domain.com]  
🐦 [@your-twitter]  
🔗 [LinkedIn profile]

---

## ⚠️ Disclaimer

This framework is a **research tool for academic analysis**, not a prediction system for policy decisions. Constitutional evolution involves complex factors beyond any single model's scope. Use responsibly and cite limitations clearly.

**Version**: 2.0 (Reformulated October 2025)  
**Status**: Active development, seeking peer review