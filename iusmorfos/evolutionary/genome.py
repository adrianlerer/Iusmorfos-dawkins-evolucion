"""
Legal Genome Representation for Evolutionary Simulation

Extends legal-memespace's LegalMemeVector with evolutionary capabilities:
- Fitness tracking (implementation success)
- Genealogy tracking (parent-child relationships)
- Template initialization (OECD, World Bank, etc.)
- Conversion to RootFinder format

Author: Adrian Lerer
Date: 2025-10-13
"""

from typing import Optional, Dict, Any, List
import numpy as np
import json
from datetime import datetime
from pathlib import Path


class LegalGenome:
    """
    Genetic representation of legal norm for evolutionary simulation.
    
    Core representation that works standalone (no external dependencies required).
    Can be extended with legal-memespace and RootFinder when available.
    
    Attributes:
        text: Legal text content
        context: Legal context (jurisdiction, legal_family, date)
        features: 89D feature vector (for anti-corruption domain)
        fitness_score: Implementation success prediction
        generation: Evolutionary generation number
        parent_id: ID of parent genome (for genealogy)
        text_id: Unique identifier
    """
    
    def __init__(
        self,
        text: str,
        context: Dict[str, Any],
        text_id: str,
        features: Optional[np.ndarray] = None,
        fitness_score: Optional[float] = None,
        generation: int = 0,
        parent_id: Optional[str] = None
    ):
        """
        Initialize Legal Genome.
        
        Args:
            text: Legal text (statute, regulation, constitutional amendment)
            context: Dict with keys: jurisdiction, legal_family, enactment_date, 
                    cultural_indices (Hofstede dimensions)
            text_id: Unique identifier (e.g., "india_gst_2017")
            features: Pre-computed 89D feature vector (optional)
            fitness_score: Implementation success [0,1] (optional)
            generation: Evolutionary timestep (default: 0)
            parent_id: Parent genome ID for genealogy (optional)
        """
        self.text = text
        self.context = context
        self.text_id = text_id
        self.features = features
        self.fitness_score = fitness_score
        self.generation = generation
        self.parent_id = parent_id
        
        # Extract features if not provided
        if self.features is None:
            self._extract_features_fallback()
    
    def _extract_features_fallback(self):
        """
        Fallback feature extraction using basic text analysis.
        
        Creates 89D feature vector based on:
        - Text length and complexity
        - Keyword presence (enforcement, sanctions, rights, etc.)
        - Structural markers (articles, sections, clauses)
        - Context-based features
        """
        text_lower = self.text.lower()
        
        # Basic text metrics (10 features)
        text_length = len(self.text)
        word_count = len(self.text.split())
        sentence_count = self.text.count('.') + self.text.count('!') + self.text.count('?')
        avg_word_length = text_length / max(word_count, 1)
        avg_sentence_length = word_count / max(sentence_count, 1)
        
        # Keyword density (30 features)
        keywords = {
            'enforcement': ['enforce', 'comply', 'compliance', 'penalty', 'sanction'],
            'rights': ['right', 'entitle', 'guarantee', 'protection', 'freedom'],
            'obligations': ['must', 'shall', 'require', 'obligate', 'duty'],
            'procedures': ['procedure', 'process', 'mechanism', 'appeal', 'review'],
            'scope': ['apply', 'applicable', 'scope', 'jurisdiction', 'territory'],
            'temporal': ['date', 'period', 'duration', 'effective', 'expire']
        }
        
        keyword_densities = []
        for category, words in keywords.items():
            density = sum(text_lower.count(word) for word in words) / max(word_count, 1)
            keyword_densities.append(min(density * 100, 1.0))  # Normalize to [0, 1]
        
        # Structural markers (15 features)
        structural_markers = {
            'articles': text_lower.count('article'),
            'sections': text_lower.count('section'),
            'clauses': text_lower.count('clause'),
            'paragraphs': text_lower.count('paragraph'),
            'subsections': text_lower.count('subsection'),
        }
        
        structural_features = [min(count / 10.0, 1.0) for count in structural_markers.values()]
        
        # Context-based features (15 features)
        context_features = []
        
        # Legal family encoding (one-hot-ish) - 5 features
        legal_families = ['civil_law', 'common_law', 'islamic_law', 'customary_law', 'socialist_law']
        current_family = self.context.get('legal_family', 'civil_law')
        for family in legal_families:
            context_features.append(1.0 if family == current_family else 0.0)
        
        # Cultural indices (if available) - 5 features
        cultural = self.context.get('cultural_indices', {})
        context_features.extend([
            cultural.get('power_distance', 50) / 100.0,
            cultural.get('individualism', 50) / 100.0,
            cultural.get('uncertainty_avoidance', 50) / 100.0,
            cultural.get('masculinity', 50) / 100.0,
            cultural.get('long_term_orientation', 50) / 100.0,
        ])
        
        # Domain and complexity features - 5 features
        context_features.extend([
            self.context.get('formality', 0.5),
            self.context.get('complexity', 0.5),
            1.0 if self.context.get('domain') == 'constitutional_law' else 0.0,
            1.0 if self.context.get('domain') == 'tax_law' else 0.0,
            1.0 if self.context.get('domain') == 'commercial_law' else 0.0,
        ])
        
        # Formalism indicators (10 features)
        formalism_markers = [
            'whereas', 'hereby', 'pursuant', 'notwithstanding', 'thereof',
            'herein', 'aforementioned', 'stipulated', 'promulgated', 'enacted'
        ]
        formalism_score = sum(text_lower.count(marker) for marker in formalism_markers) / max(word_count, 1)
        formalism_features = [min(formalism_score * 50, 1.0)] * 10
        
        # Combine all features to reach 89D
        all_features = (
            [text_length / 10000.0, avg_word_length / 10.0, avg_sentence_length / 50.0] +
            keyword_densities * 5 +  # Repeat to reach 30
            structural_features * 3 +  # 15 features
            context_features +  # 15 features
            formalism_features +  # 10 features
            [0.5] * 16  # Padding to reach exactly 89
        )
        
        # Ensure exactly 89 dimensions
        self.features = np.array(all_features[:89])
        
        # Normalize to [0, 1]
        self.features = np.clip(self.features, 0.0, 1.0)
    
    def to_vector(self) -> np.ndarray:
        """Return feature vector for distance calculations."""
        if self.features is None:
            raise ValueError("Features not extracted yet. Call _extract_features_fallback() first.")
        return self.features
    
    @classmethod
    def from_vector(cls, 
                    features: np.ndarray,
                    text_id: str,
                    context: Dict[str, Any],
                    **kwargs) -> 'LegalGenome':
        """
        Reconstruct genome from feature vector.
        
        Useful for mutation operations that work in feature space.
        """
        return cls(
            text="[Generated from feature vector]",
            context=context,
            text_id=text_id,
            features=features,
            **kwargs
        )
    
    def distance_to(self, other: 'LegalGenome') -> float:
        """
        Euclidean distance in 89D feature space.
        
        Measures "genetic distance" between two legal norms.
        Used to quantify mutation magnitude.
        
        Args:
            other: Another LegalGenome
        
        Returns:
            Float distance in [0, inf]
        """
        return float(np.linalg.norm(self.to_vector() - other.to_vector()))
    
    def cosine_similarity(self, other: 'LegalGenome') -> float:
        """
        Cosine similarity in feature space.
        
        Returns:
            Float in [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
        """
        v1 = self.to_vector()
        v2 = other.to_vector()
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        if norms == 0:
            return 0.0
        return float(dot_product / norms)
    
    @classmethod
    def from_empirical_case(cls, case_id: str, data_dir: str = "data") -> 'LegalGenome':
        """
        Initialize from Iusmorfos validated empirical case.
        
        Loads case from global_cases_database.json and extracts features.
        
        Args:
            case_id: Case identifier (e.g., "india_gst_2017")
            data_dir: Directory containing data files
        
        Returns:
            LegalGenome initialized with case data
        
        Example:
            genome = LegalGenome.from_empirical_case("india_gst_2017")
        """
        # Load global cases database
        db_path = Path(data_dir) / "global_cases_database.json"
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        with open(db_path, 'r') as f:
            database = json.load(f)
        
        # Find case
        case = None
        for c in database.get('cases', []):
            if c.get('case_id') == case_id:
                case = c
                break
        
        if case is None:
            raise ValueError(f"Case not found: {case_id}")
        
        # Extract context
        context = {
            'jurisdiction': case.get('country'),
            'legal_family': case.get('legal_family', 'civil_law'),
            'enactment_date': datetime.fromisoformat(case.get('enactment_date', '2020-01-01')),
            'cultural_indices': case.get('cultural_indices', {})
        }
        
        # Create genome
        return cls(
            text=case.get('description', f"Legal reform in {case.get('country')}"),
            context=context,
            text_id=case_id,
            fitness_score=case.get('implementation_success')
        )
    
    @classmethod
    def from_template(cls, template: str, text_id: str = None) -> 'LegalGenome':
        """
        Initialize from standard legal template.
        
        Templates:
        - "oecd_gst_model": OECD Goods & Services Tax harmonization
        - "world_bank_regulatory": World Bank regulatory best practices
        - "constitutional_amendment": Typical constitutional reform structure
        
        Args:
            template: Template name
            text_id: Optional custom ID (auto-generated if None)
        
        Returns:
            LegalGenome with template-specific features
        
        Example:
            genome = LegalGenome.from_template("oecd_gst_model")
        """
        templates = {
            "oecd_gst_model": {
                'text': """
Goods and Services Tax (GST) Framework

Article 1: Scope
This tax shall apply to all supplies of goods and services within the territory.

Article 2: Tax Rate
Standard rate of 18% shall apply, with reduced rates for essential goods.

Article 3: Registration
Businesses with turnover exceeding threshold must register.

Article 4: Input Tax Credit
Registered businesses may claim credit for tax paid on inputs.

Article 5: Compliance
Monthly returns required; penalties for non-compliance.

Article 6: Enforcement
Tax authorities may conduct audits and impose sanctions.

Article 7: Appeals
Taxpayers may appeal decisions within 30 days.

Article 8: Transitional Provisions
Existing tax credits transferred under new system.
                """,
                'context': {
                    'jurisdiction': 'OECD Model',
                    'legal_family': 'civil_law',
                    'enactment_date': datetime(2015, 1, 1),
                    'cultural_indices': {
                        'power_distance': 50,
                        'individualism': 70,
                        'uncertainty_avoidance': 60
                    }
                },
                'features_template': np.array([0.90, 0.95, 0.70, 0.85, 0.60] + [0.5]*84)  # 89D
            },
            
            "world_bank_regulatory": {
                'text': """
Regulatory Framework for Business Operations

Section 1: Registration Procedures
Simplified online registration within 5 business days.

Section 2: Licensing Requirements
Risk-based licensing approach; automatic approval for low-risk activities.

Section 3: Compliance Obligations
Annual reporting; digital submission platform.

Section 4: Enforcement
Proportionate penalties; appeals mechanism.

Section 5: Monitoring
Regular compliance reviews; support for small businesses.

Section 6: Dispute Resolution
Independent arbitration available for regulatory disputes.
                """,
                'context': {
                    'jurisdiction': 'World Bank Model',
                    'legal_family': 'common_law',
                    'enactment_date': datetime(2018, 1, 1),
                    'cultural_indices': {
                        'power_distance': 40,
                        'individualism': 80,
                        'uncertainty_avoidance': 50
                    }
                },
                'features_template': np.array([0.70, 0.85, 0.60, 0.75, 0.50] + [0.5]*84)
            },
            
            "constitutional_amendment": {
                'text': """
Constitutional Amendment Procedure

Article 1: Proposal
Amendments may be proposed by 2/3 of legislature or citizen petition.

Article 2: Deliberation
Public consultation period of 90 days minimum.

Article 3: Approval
Requires 2/3 supermajority in both chambers plus 50% of states.

Article 4: Promulgation
President shall promulgate within 30 days of approval.

Article 5: Judicial Review
Constitutional court may review amendments within 6 months.

Article 6: Entrenchment
Certain fundamental provisions cannot be amended.
                """,
                'context': {
                    'jurisdiction': 'Generic Constitutional',
                    'legal_family': 'civil_law',
                    'enactment_date': datetime(2000, 1, 1),
                    'cultural_indices': {
                        'power_distance': 60,
                        'individualism': 60,
                        'uncertainty_avoidance': 70
                    }
                },
                'features_template': np.array([0.95, 0.90, 0.80, 0.70, 0.80] + [0.5]*84)
            }
        }
        
        if template not in templates:
            raise ValueError(f"Unknown template: {template}. Available: {list(templates.keys())}")
        
        template_data = templates[template]
        
        if text_id is None:
            text_id = f"{template}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        genome = cls(
            text=template_data['text'],
            context=template_data['context'],
            text_id=text_id,
            features=template_data['features_template']
        )
        
        return genome
    
    def to_policy_genome(self) -> Dict[str, Any]:
        """
        Convert to RootFinder's PolicyGenealogy format.
        
        Returns dict that can be passed to RootFinder.add_policy()
        """
        return {
            'policy_id': self.text_id,
            'text': self.text,
            'features': self.features.tolist() if self.features is not None else None,
            'context': self.context,
            'generation': self.generation,
            'parent_id': self.parent_id,
            'metadata': {
                'fitness_score': self.fitness_score,
                'feature_dim': len(self.features) if self.features is not None else 0
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for saving."""
        # Convert datetime to ISO format if present
        context_copy = self.context.copy()
        if 'enactment_date' in context_copy and isinstance(context_copy['enactment_date'], datetime):
            context_copy['enactment_date'] = context_copy['enactment_date'].isoformat()
        
        return {
            'text': self.text,
            'context': context_copy,
            'text_id': self.text_id,
            'features': self.features.tolist() if self.features is not None else None,
            'fitness_score': self.fitness_score,
            'generation': self.generation,
            'parent_id': self.parent_id
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LegalGenome':
        """Deserialize from dictionary."""
        features = np.array(data['features']) if data.get('features') is not None else None
        
        # Convert ISO format back to datetime if present
        context = data['context'].copy()
        if 'enactment_date' in context and isinstance(context['enactment_date'], str):
            context['enactment_date'] = datetime.fromisoformat(context['enactment_date'])
        
        return cls(
            text=data['text'],
            context=context,
            text_id=data['text_id'],
            features=features,
            fitness_score=data.get('fitness_score'),
            generation=data.get('generation', 0),
            parent_id=data.get('parent_id')
        )
    
    def __repr__(self) -> str:
        fitness_str = f"{self.fitness_score:.3f}" if self.fitness_score is not None else "None"
        return (f"LegalGenome(id={self.text_id}, gen={self.generation}, "
                f"fitness={fitness_str}, parent={self.parent_id})")
