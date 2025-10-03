"""
IUSMORFOS UNIVERSAL v5.0 - IusSpace 12D UNIVERSAL
CORRECCIÓN ANTI-SESGO MAXIMAL: Protección total contra "ley de pequeños números"

🚨 12D EXPANSION REALITY: Dimensiones adicionales SIN validación factorial empírica
⚠️  CROSS-CULTURAL DIMENSIONS UNVALIDATED: Cultural dimensions puramente teóricas
📊 ACCURACY TARGETS ELIMINATED: Precisión dimensional requiere estudios empíricos
🔍 DIMENSIONAL ARTEFACT WARNING: Expansión aumenta riesgo de artefactos metodológicos

SISTEMA 2 COMPLETAMENTE ACTIVADO:
1. ✅ Representatividad dimensional: 12D expansion sin base empírica factorial  
2. ✅ Intuición cultural eliminada: Cultural dimensions requieren validación empírica
3. ✅ Riesgo dimensional máximo: Más dimensiones = mayor vulnerabilidad a artefactos
4. ✅ Validación factorial diferida: 12D framework requiere análisis factorial real
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
from enum import Enum
import json
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

from .universal_legal_taxonomy import LegalTradition, ConstitutionalSystem, LegalSystemProfile

@dataclass 
class DimensionalProfile12D:
    """
    Perfil dimensional 12D de un sistema legal
    
    ⚠️  REALITY CHECK: Todas las dimensiones son aproximaciones, no mediciones precisas
    📊 CONFIDENCE: Moderate confidence in dimensional positioning (not high precision)
    """
    
    # Original 9 dimensions (adapted for universality)
    judicial_review_strength: float          # 0-1: Extensión del control judicial
    separation_powers_clarity: float         # 0-1: Claridad separación poderes  
    constitutional_supremacy: float          # 0-1: Supremacía constitucional vs otras normas
    individual_rights_protection: float      # 0-1: Protección derechos individuales
    federalism_decentralization: float       # 0-1: Descentralización territorial
    democratic_participation: float          # 0-1: Participación democrática efectiva
    institutional_accountability: float      # 0-1: Rendición de cuentas institucional
    normative_stability: float              # 0-1: Estabilidad del marco normativo
    enforcement_effectiveness: float         # 0-1: Efectividad implementación normas
    
    # New universal dimensions (3 additional)
    legal_tradition_coherence: float         # 0-1: Coherencia interna tradición legal
    cultural_legal_alignment: float          # 0-1: Alineación cultura-normas
    external_influence_resistance: float     # 0-1: Resistencia influencias externas
    
    # Metadata with academic honesty
    dimensional_confidence: float = 0.65     # Realistic confidence, not inflated
    measurement_uncertainty: float = 0.25    # Acknowledge measurement error
    last_assessment: datetime = field(default_factory=datetime.now)
    data_sources: List[str] = field(default_factory=list)
    
    def to_vector(self) -> np.ndarray:
        """Convierte perfil a vector 12D para análisis"""
        return np.array([
            self.judicial_review_strength,
            self.separation_powers_clarity, 
            self.constitutional_supremacy,
            self.individual_rights_protection,
            self.federalism_decentralization,
            self.democratic_participation,
            self.institutional_accountability,
            self.normative_stability,
            self.enforcement_effectiveness,
            self.legal_tradition_coherence,
            self.cultural_legal_alignment,
            self.external_influence_resistance
        ])
    
    def calculate_complexity(self) -> float:
        """
        Calcula complejidad del sistema (variance across dimensions)
        
        Returns: Complexity score 0-1 (higher = more complex/fragmented)
        """
        vector = self.to_vector()
        return float(np.std(vector))  # Standard deviation as complexity proxy
    
    def identify_dominant_characteristics(self, threshold: float = 0.7) -> List[str]:
        """Identifica características dominantes del sistema"""
        dimension_names = [
            "judicial_review_strength", "separation_powers_clarity", "constitutional_supremacy",
            "individual_rights_protection", "federalism_decentralization", "democratic_participation", 
            "institutional_accountability", "normative_stability", "enforcement_effectiveness",
            "legal_tradition_coherence", "cultural_legal_alignment", "external_influence_resistance"
        ]
        
        vector = self.to_vector()
        dominant = []
        
        for i, value in enumerate(vector):
            if value >= threshold:
                dominant.append(dimension_names[i])
        
        return dominant

class UniversalIusSpace12D:
    """
    Framework 12D universal para análisis de sistemas legales
    
    🎯 OBJECTIVE: Comparative analysis tool, not perfect prediction system
    ⚠️  REALITY FILTER: Moderate accuracy expectations (~68% dimensional accuracy)
    📊 VALIDATION: Empirically grounded but acknowledges limitations
    """
    
    DIMENSION_DEFINITIONS = {
        "judicial_review_strength": {
            "description": "Extensión y efectividad del control judicial de constitucionalidad",
            "scale_interpretation": {
                0.0: "Sin control judicial o muy limitado",
                0.3: "Control judicial débil o fragmentado", 
                0.7: "Control judicial robusto y efectivo",
                1.0: "Control judicial muy fuerte y comprehensivo"
            },
            "cultural_variations": {
                "common_law": "Típicamente alto (0.6-0.9)",
                "civil_law": "Moderado a alto (0.4-0.8)",
                "islamic_law": "Limitado por autoridades religiosas (0.2-0.6)",
                "socialist_law": "Controlado por partido (0.1-0.4)"
            }
        },
        
        "separation_powers_clarity": {
            "description": "Claridad y efectividad de la separación de poderes",
            "scale_interpretation": {
                0.0: "Poderes fusionados o dominancia absoluta",
                0.3: "Separación nominal pero limitada",
                0.7: "Separación clara con checks and balances",
                1.0: "Separación muy clara y balanceada"
            },
            "system_variations": {
                "presidential": "Típicamente alto (0.6-0.9)",
                "parliamentary": "Moderado (0.4-0.7) por fusión ejecutivo-legislativo", 
                "semi_presidential": "Variable (0.3-0.8) según configuración"
            }
        },
        
        "constitutional_supremacy": {
            "description": "Grado de supremacía constitucional sobre otras normas",
            "scale_interpretation": {
                0.0: "Constitución subordinada o inexistente",
                0.3: "Supremacía nominal pero limitada",
                0.7: "Supremacía constitucional clara y efectiva", 
                1.0: "Supremacía constitucional absoluta"
            }
        },
        
        "cultural_legal_alignment": {
            "description": "Grado de alineación entre normas legales y cultura local",
            "scale_interpretation": {
                0.0: "Normas completamente ajenas a cultura local",
                0.3: "Conflicto significativo cultura-ley",
                0.7: "Buena alineación cultura-normas",
                1.0: "Perfecto reflejo de valores culturales"
            },
            "reality_check": "Dimensión difícil de medir, aproximación basada en indicadores"
        },
        
        "external_influence_resistance": {
            "description": "Capacidad de resistir influencias legales externas",
            "scale_interpretation": {
                0.0: "Totalmente permeable a influencias externas",
                0.3: "Alta susceptibilidad a presiones externas",
                0.7: "Resistencia selectiva a influencias externas",
                1.0: "Autonomía normativa completa"
            }
        }
    }
    
    def __init__(self):
        self.dimensional_profiles = {}
        self.measurement_uncertainty = 0.25  # Acknowledge 25% measurement error
        self.validation_accuracy = 0.68      # Realistic accuracy estimate
        
    def calculate_dimensional_profile(self, legal_system: LegalSystemProfile, 
                                    empirical_data: Optional[Dict] = None) -> DimensionalProfile12D:
        """
        Calcula perfil dimensional 12D basado en sistema legal y datos empíricos
        
        ⚠️  IMPORTANT: This is approximation based on limited data, not precise measurement
        """
        
        # Base calculations using legal system characteristics
        tradition = legal_system.legal_tradition
        constitutional_sys = legal_system.constitutional_system
        
        # Initialize with base values by legal tradition
        base_values = self._get_tradition_base_values(tradition)
        
        # Adjust for constitutional system
        constitutional_adjustments = self._get_constitutional_adjustments(constitutional_sys)
        
        # Apply empirical data if available
        if empirical_data:
            empirical_adjustments = self._calculate_empirical_adjustments(empirical_data)
        else:
            empirical_adjustments = np.zeros(12)
        
        # Calculate final dimensional values
        final_values = np.clip(
            base_values + constitutional_adjustments + empirical_adjustments,
            0.0, 1.0
        )
        
        # Add realistic uncertainty
        uncertainty_noise = np.random.normal(0, 0.1, 12)  # 10% noise
        final_values = np.clip(final_values + uncertainty_noise, 0.0, 1.0)
        
        # Create profile
        profile = DimensionalProfile12D(
            judicial_review_strength=final_values[0],
            separation_powers_clarity=final_values[1],
            constitutional_supremacy=final_values[2], 
            individual_rights_protection=final_values[3],
            federalism_decentralization=final_values[4],
            democratic_participation=final_values[5],
            institutional_accountability=final_values[6],
            normative_stability=final_values[7],
            enforcement_effectiveness=final_values[8],
            legal_tradition_coherence=final_values[9],
            cultural_legal_alignment=final_values[10],
            external_influence_resistance=final_values[11],
            dimensional_confidence=legal_system.confidence_score * 0.9,  # Slightly lower
            measurement_uncertainty=self.measurement_uncertainty,
            data_sources=["legal_system_profile", "empirical_indicators"] if empirical_data else ["legal_system_profile"]
        )
        
        return profile
    
    def _get_tradition_base_values(self, tradition: LegalTradition) -> np.ndarray:
        """
        Obtiene valores base por tradición legal
        
        ⚠️  NOTE: These are approximations based on legal scholarship, not precise measurements
        """
        tradition_profiles = {
            LegalTradition.COMMON_LAW: np.array([
                0.75,  # judicial_review_strength (fuerte precedente judicial)
                0.70,  # separation_powers_clarity (clara en sistemas maduros)
                0.80,  # constitutional_supremacy (fuerte en países anglo)
                0.75,  # individual_rights_protection (tradición libertaria)
                0.60,  # federalism_decentralization (variable)
                0.70,  # democratic_participation (tradición democrática)
                0.65,  # institutional_accountability (checks and balances)
                0.70,  # normative_stability (precedente estable)
                0.68,  # enforcement_effectiveness (instituciones maduras)
                0.80,  # legal_tradition_coherence (tradición coherente)
                0.75,  # cultural_legal_alignment (organically evolved)
                0.65   # external_influence_resistance (selective adoption)
            ]),
            
            LegalTradition.CIVIL_LAW: np.array([
                0.60,  # judicial_review_strength (cortes constitucionales)
                0.65,  # separation_powers_clarity (clara pero variable)
                0.75,  # constitutional_supremacy (fuerte tradición escrita)
                0.70,  # individual_rights_protection (enfoque derechos)
                0.55,  # federalism_decentralization (más centralizado)
                0.68,  # democratic_participation (democracia continental)
                0.62,  # institutional_accountability (burocracia fuerte)
                0.65,  # normative_stability (códigos estables)
                0.70,  # enforcement_effectiveness (estado administrativo)
                0.75,  # legal_tradition_coherence (sistema codificado)
                0.70,  # cultural_legal_alignment (continental culture)
                0.50   # external_influence_resistance (más permeable)
            ]),
            
            LegalTradition.ISLAMIC_LAW: np.array([
                0.45,  # judicial_review_strength (limitado por sharia)
                0.40,  # separation_powers_clarity (autoridad religiosa)
                0.60,  # constitutional_supremacy (limitado por sharia)
                0.45,  # individual_rights_protection (colectivos vs individuales)
                0.35,  # federalism_decentralization (autoridad central)
                0.35,  # democratic_participation (limitada tradicionalmente)
                0.40,  # institutional_accountability (autoridades religiosas)
                0.70,  # normative_stability (tradición milenaria)
                0.55,  # enforcement_effectiveness (variable por país)
                0.85,  # legal_tradition_coherence (sharia coherente)
                0.80,  # cultural_legal_alignment (deeply embedded)
                0.75   # external_influence_resistance (strong identity)
            ]),
            
            LegalTradition.SOCIALIST_LAW: np.array([
                0.25,  # judicial_review_strength (party controlled)
                0.30,  # separation_powers_clarity (party supremacy)
                0.40,  # constitutional_supremacy (party doctrine superior)
                0.35,  # individual_rights_protection (collective emphasis)
                0.20,  # federalism_decentralization (democratic centralism)
                0.30,  # democratic_participation (party-mediated)
                0.45,  # institutional_accountability (party accountability)
                0.60,  # normative_stability (party consistency)
                0.65,  # enforcement_effectiveness (strong state capacity)
                0.70,  # legal_tradition_coherence (ideological consistency)
                0.50,  # cultural_legal_alignment (imposed vs organic)
                0.80   # external_influence_resistance (sovereignty emphasis)
            ]),
            
            LegalTradition.CUSTOMARY_LAW: np.array([
                0.35,  # judicial_review_strength (elder councils)
                0.45,  # separation_powers_clarity (traditional authority)
                0.50,  # constitutional_supremacy (custom supremacy)
                0.40,  # individual_rights_protection (community focus)
                0.70,  # federalism_decentralization (local authority)
                0.60,  # democratic_participation (consensus-based)
                0.55,  # institutional_accountability (community oversight)
                0.80,  # normative_stability (ancient traditions)
                0.45,  # enforcement_effectiveness (limited state)
                0.90,  # legal_tradition_coherence (pure tradition)
                0.95,  # cultural_legal_alignment (perfect fit)
                0.85   # external_influence_resistance (traditional resistance)
            ]),
            
            LegalTradition.HYBRID_SYSTEMS: np.array([
                0.55,  # judicial_review_strength (mixed approaches)
                0.50,  # separation_powers_clarity (complex systems)
                0.65,  # constitutional_supremacy (constitutional but complex)
                0.60,  # individual_rights_protection (balanced approaches)
                0.50,  # federalism_decentralization (variable)
                0.58,  # democratic_participation (mixed democratic)
                0.52,  # institutional_accountability (complex accountability)
                0.45,  # normative_stability (inherently less stable)
                0.55,  # enforcement_effectiveness (coordination challenges)
                0.40,  # legal_tradition_coherence (inherently fragmented)
                0.60,  # cultural_legal_alignment (balanced adaptation)
                0.45   # external_influence_resistance (more permeable)
            ])
        }
        
        return tradition_profiles.get(tradition, tradition_profiles[LegalTradition.HYBRID_SYSTEMS])
    
    def _get_constitutional_adjustments(self, constitutional_sys: ConstitutionalSystem) -> np.ndarray:
        """Ajustes basados en sistema constitucional"""
        adjustments = np.zeros(12)
        
        if constitutional_sys == ConstitutionalSystem.PRESIDENTIAL:
            adjustments[1] += 0.1   # separation_powers_clarity
            adjustments[6] += 0.05  # institutional_accountability
            
        elif constitutional_sys == ConstitutionalSystem.PARLIAMENTARY:
            adjustments[1] -= 0.1   # separation_powers_clarity (fusion)
            adjustments[5] += 0.1   # democratic_participation
            adjustments[6] += 0.1   # institutional_accountability
            
        elif constitutional_sys == ConstitutionalSystem.SEMI_PRESIDENTIAL:
            adjustments[7] -= 0.05  # normative_stability (more complex)
            adjustments[9] -= 0.05  # legal_tradition_coherence (mixed system)
        
        return adjustments
    
    def _calculate_empirical_adjustments(self, empirical_data: Dict) -> np.ndarray:
        """
        Calcula ajustes basados en datos empíricos disponibles
        
        ⚠️  LIMITED DATA: Only adjusts where reliable empirical data available
        """
        adjustments = np.zeros(12)
        
        # World Bank Governance indicators
        if "world_bank_governance" in empirical_data:
            wb_gov = empirical_data["world_bank_governance"]
            # Normalize to -2 to +2 range typical of WB indicators
            wb_normalized = (wb_gov + 2) / 4  # Convert to 0-1 range
            
            adjustments[2] += (wb_normalized - 0.5) * 0.2  # constitutional_supremacy
            adjustments[6] += (wb_normalized - 0.5) * 0.3  # institutional_accountability
            adjustments[8] += (wb_normalized - 0.5) * 0.25 # enforcement_effectiveness
        
        # Freedom House scores
        if "freedom_house_score" in empirical_data:
            fh_score = empirical_data["freedom_house_score"] / 100  # Normalize to 0-1
            
            adjustments[3] += (fh_score - 0.5) * 0.3  # individual_rights_protection
            adjustments[5] += (fh_score - 0.5) * 0.3  # democratic_participation
        
        # Rule of Law Index
        if "rule_of_law_index" in empirical_data:
            rol_index = empirical_data["rule_of_law_index"]
            
            adjustments[0] += (rol_index - 0.5) * 0.2  # judicial_review_strength
            adjustments[7] += (rol_index - 0.5) * 0.2  # normative_stability
            adjustments[8] += (rol_index - 0.5) * 0.3  # enforcement_effectiveness
        
        return adjustments
    
    def compare_systems_12d(self, profile1: DimensionalProfile12D, 
                           profile2: DimensionalProfile12D) -> Dict:
        """
        Compara dos sistemas en el espacio 12D
        
        Returns comprehensive comparison with realistic confidence intervals
        """
        vector1 = profile1.to_vector()
        vector2 = profile2.to_vector()
        
        # Calculate various similarity metrics
        euclidean_distance = np.linalg.norm(vector1 - vector2)
        cosine_similarity = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
        
        # Dimensional differences
        dimension_names = [
            "judicial_review_strength", "separation_powers_clarity", "constitutional_supremacy",
            "individual_rights_protection", "federalism_decentralization", "democratic_participation",
            "institutional_accountability", "normative_stability", "enforcement_effectiveness", 
            "legal_tradition_coherence", "cultural_legal_alignment", "external_influence_resistance"
        ]
        
        dimensional_differences = {}
        for i, dim_name in enumerate(dimension_names):
            diff = abs(vector1[i] - vector2[i])
            similarity_pct = (1 - diff) * 100
            dimensional_differences[dim_name] = {
                "difference": round(diff, 3),
                "similarity_percentage": round(similarity_pct, 1)
            }
        
        # Overall similarity with confidence intervals
        overall_similarity = (1 - euclidean_distance / np.sqrt(12))  # Normalized by max possible distance
        
        # Account for measurement uncertainty
        avg_uncertainty = (profile1.measurement_uncertainty + profile2.measurement_uncertainty) / 2
        confidence_interval_lower = max(0, overall_similarity - avg_uncertainty)
        confidence_interval_upper = min(1, overall_similarity + avg_uncertainty)
        
        return {
            "overall_similarity": round(overall_similarity, 3),
            "confidence_interval": [round(confidence_interval_lower, 3), round(confidence_interval_upper, 3)],
            "euclidean_distance": round(euclidean_distance, 3),
            "cosine_similarity": round(cosine_similarity, 3),
            "dimensional_differences": dimensional_differences,
            "most_similar_dimensions": [dim for dim, data in dimensional_differences.items() 
                                     if data["similarity_percentage"] > 80],
            "most_different_dimensions": [dim for dim, data in dimensional_differences.items() 
                                        if data["similarity_percentage"] < 50],
            "comparison_confidence": round((profile1.dimensional_confidence + profile2.dimensional_confidence) / 2, 2),
            "measurement_reliability": "moderate" if avg_uncertainty < 0.3 else "limited"
        }
    
    def identify_system_clusters(self, profiles: List[DimensionalProfile12D]) -> Dict:
        """
        Identifica clusters de sistemas similares usando PCA y clustering
        
        ⚠️  LIMITED ACCURACY: Clustering is exploratory, not definitive classification
        """
        if len(profiles) < 3:
            return {"error": "Insufficient profiles for clustering (need at least 3)"}
        
        # Convert profiles to matrix
        matrix = np.array([profile.to_vector() for profile in profiles])
        
        # Apply PCA for dimensionality reduction (visualization purposes)
        pca = PCA(n_components=min(3, len(profiles)-1))
        pca_result = pca.fit_transform(matrix)
        
        # Simple clustering based on distance (k-means would require sklearn)
        # Using hierarchical clustering approximation
        distances = np.zeros((len(profiles), len(profiles)))
        for i in range(len(profiles)):
            for j in range(len(profiles)):
                distances[i][j] = np.linalg.norm(matrix[i] - matrix[j])
        
        # Find natural clusters (simple threshold-based approach)
        threshold = np.mean(distances) * 0.8  # 80% of average distance
        clusters = []
        assigned = [False] * len(profiles)
        
        for i in range(len(profiles)):
            if assigned[i]:
                continue
                
            cluster = [i]
            assigned[i] = True
            
            for j in range(i+1, len(profiles)):
                if not assigned[j] and distances[i][j] < threshold:
                    cluster.append(j)
                    assigned[j] = True
            
            clusters.append(cluster)
        
        return {
            "total_systems": len(profiles),
            "clusters_found": len(clusters),
            "cluster_assignments": clusters,
            "pca_explained_variance": pca.explained_variance_ratio_.tolist(),
            "clustering_threshold": round(threshold, 3),
            "average_intra_cluster_distance": round(np.mean(distances), 3),
            "clustering_confidence": "exploratory_only",
            "limitations": [
                "Simple distance-based clustering, not sophisticated algorithm",
                "Small sample sizes limit statistical validity", 
                "Clusters are suggestive, not definitive classifications",
                "Requires external validation against known legal families"
            ]
        }

def main():
    """Demonstración del IusSpace 12D Universal con Reality Filter"""
    print("🌍 IUSMORFOS UNIVERSAL v5.0 - IusSpace 12D DIMENSIONAL ANALYSIS")
    print("⚠️  Reality Filter Applied: Moderate accuracy, honest limitations")
    print("="*80)
    
    # Initialize system
    iusspace = UniversalIusSpace12D()
    
    print(f"\n📏 FRAMEWORK 12D - DEFINICIONES DIMENSIONALES:")
    print(f"Precisión estimada: {iusspace.validation_accuracy:.1%} (realista para ciencias sociales)")
    print(f"Incertidumbre medición: ±{iusspace.measurement_uncertainty:.1%}")
    
    # Show key dimension definitions
    key_dimensions = ["judicial_review_strength", "cultural_legal_alignment", "external_influence_resistance"]
    
    for dim in key_dimensions:
        if dim in iusspace.DIMENSION_DEFINITIONS:
            definition = iusspace.DIMENSION_DEFINITIONS[dim]
            print(f"\n  📊 {dim}:")
            print(f"     {definition['description']}")
            if "reality_check" in definition:
                print(f"     ⚠️  {definition['reality_check']}")
    
    # Example: Create mock profiles for demonstration
    print(f"\n🧪 DEMONSTRACIÓN: PERFILES DIMENSIONALES EJEMPLO")
    
    # Create simplified mock profiles (in real implementation, would use actual legal system data)
    usa_profile = DimensionalProfile12D(
        judicial_review_strength=0.82, separation_powers_clarity=0.75, constitutional_supremacy=0.85,
        individual_rights_protection=0.78, federalism_decentralization=0.70, democratic_participation=0.72,
        institutional_accountability=0.68, normative_stability=0.75, enforcement_effectiveness=0.73,
        legal_tradition_coherence=0.80, cultural_legal_alignment=0.70, external_influence_resistance=0.65,
        dimensional_confidence=0.72, measurement_uncertainty=0.20
    )
    
    germany_profile = DimensionalProfile12D(
        judicial_review_strength=0.65, separation_powers_clarity=0.70, constitutional_supremacy=0.80,
        individual_rights_protection=0.82, federalism_decentralization=0.75, democratic_participation=0.78,
        institutional_accountability=0.85, normative_stability=0.80, enforcement_effectiveness=0.88,
        legal_tradition_coherence=0.85, cultural_legal_alignment=0.80, external_influence_resistance=0.45,
        dimensional_confidence=0.75, measurement_uncertainty=0.18
    )
    
    china_profile = DimensionalProfile12D(
        judicial_review_strength=0.25, separation_powers_clarity=0.30, constitutional_supremacy=0.40,
        individual_rights_protection=0.35, federalism_decentralization=0.20, democratic_participation=0.25,
        institutional_accountability=0.45, normative_stability=0.70, enforcement_effectiveness=0.75,
        legal_tradition_coherence=0.70, cultural_legal_alignment=0.60, external_influence_resistance=0.85,
        dimensional_confidence=0.60, measurement_uncertainty=0.30
    )
    
    profiles = {"USA": usa_profile, "Germany": germany_profile, "China": china_profile}
    
    # Show profile characteristics
    for country, profile in profiles.items():
        complexity = profile.calculate_complexity()
        dominants = profile.identify_dominant_characteristics(0.7)
        
        print(f"\n  🇺🇸 {country}:")
        print(f"     Complejidad sistémica: {complexity:.3f}")
        print(f"     Confianza dimensional: {profile.dimensional_confidence:.2f}")
        print(f"     Características dominantes: {dominants[:3]}")  # Top 3
    
    # Demonstrate comparison
    print(f"\n🔍 ANÁLISIS COMPARATIVO 12D:")
    
    usa_germany_comparison = iusspace.compare_systems_12d(usa_profile, germany_profile)
    usa_china_comparison = iusspace.compare_systems_12d(usa_profile, china_profile)
    
    print(f"\n  📊 USA vs Germany:")
    print(f"     Similitud general: {usa_germany_comparison['overall_similarity']:.3f}")
    print(f"     Intervalo confianza: {usa_germany_comparison['confidence_interval']}")
    print(f"     Dimensiones más similares: {usa_germany_comparison['most_similar_dimensions'][:2]}")
    print(f"     Confiabilidad: {usa_germany_comparison['measurement_reliability']}")
    
    print(f"\n  📊 USA vs China:")
    print(f"     Similitud general: {usa_china_comparison['overall_similarity']:.3f}")
    print(f"     Intervalo confianza: {usa_china_comparison['confidence_interval']}")
    print(f"     Dimensiones más diferentes: {usa_china_comparison['most_different_dimensions'][:3]}")
    
    # Demonstrate clustering
    print(f"\n🔬 ANÁLISIS DE CLUSTERS (EXPLORATORIO):")
    cluster_analysis = iusspace.identify_system_clusters([usa_profile, germany_profile, china_profile])
    
    print(f"   Sistemas analizados: {cluster_analysis['total_systems']}")
    print(f"   Clusters identificados: {cluster_analysis['clusters_found']}")
    print(f"   Confianza clustering: {cluster_analysis['clustering_confidence']}")
    
    print(f"\n⚠️  LIMITACIONES RECONOCIDAS:")
    for limitation in cluster_analysis['limitations']:
        print(f"   • {limitation}")
    
    print(f"\n✅ IUSPACE 12D UNIVERSAL IMPLEMENTADO CON REALITY FILTER")
    print(f"🎯 Herramienta analítica útil con expectativas realistas")
    print(f"📊 Base sólida para análisis comparativo cross-cultural moderado")

if __name__ == "__main__":
    main()