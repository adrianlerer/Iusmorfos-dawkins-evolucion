# iusmorfo_evolution.py

import numpy as np
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
import json
from datetime import datetime

@dataclass
class Iusmorfo:
    """Representa un sistema legal con 9 genes"""
    genes: Dict[str, int]
    generation: int
    parent_id: str = None
    id: str = None
    
    def __post_init__(self):
        if self.id is None:
            self.id = f"gen{self.generation}_{random.randint(1000,9999)}"
    
    def to_legal_text(self) -> str:
        """Convierte genes en texto legal"""
        base = "Neminem laedere"
        
        # Modificadores basados en genes
        if self.genes['specificity'] > 7:
            base += " in contractibus specifice determinatis"
        if self.genes['procedure'] > 7:
            base += " nisi per debitum processum"
        if self.genes['exceptions'] > 7:
            base += " praeter casus necessitatis"
        if self.genes['severity'] > 7:
            base += " sub poena gravis"
        if self.genes['state_role'] > 7:
            base += " sub auctoritate publica"
        if self.genes['temporality'] > 7:
            base += " tempore determinato"
        if self.genes['burden_proof'] > 7:
            base += " onus probandi incumbit actori"
        if self.genes['remedy'] > 7:
            base += " cum remedio specifico"
        if self.genes['jurisdiction'] > 7:
            base += " in foro competenti"
            
        return base
    
    def calculate_complexity(self) -> float:
        """Calcula complejidad total del sistema"""
        return sum(self.genes.values()) / 90.0  # Normalizado 0-1
    
    def calculate_balance(self) -> float:
        """Calcula balance del sistema (qué tan equilibrados están los genes)"""
        values = list(self.genes.values())
        mean_val = np.mean(values)
        std_val = np.std(values)
        return 1.0 - (std_val / 10.0)  # Normalizado, mayor balance = menor desviación
    
    def calculate_diversity(self) -> float:
        """Calcula diversidad del sistema"""
        values = list(self.genes.values())
        unique_vals = len(set(values))
        return unique_vals / 9.0  # Normalizado 0-1

class IusmorfoEvolution:
    def __init__(self):
        # Gen 0: Máxima simple ("Neminem laedere" puro)
        self.initial_genes = {
            'specificity': 1,      # Especificidad/codificación
            'procedure': 1,        # Complejidad procesal
            'exceptions': 1,       # Excepciones y matices
            'severity': 1,         # Severidad punitiva
            'state_role': 1,       # Rol del estado
            'temporality': 1,      # Aspectos temporales
            'burden_proof': 1,     # Carga de la prueba
            'remedy': 1,           # Remedios disponibles
            'jurisdiction': 1      # Jurisdicción y competencia
        }
        self.current_generation = 0
        self.population = []
        self.history = []
        self.selection_history = []
        
    def generate_offspring(self, parent: Iusmorfo) -> List[Iusmorfo]:
        """Genera 9 descendientes, cada uno mutado en un gen diferente"""
        offspring = []
        gene_names = list(parent.genes.keys())
        
        for i, gene_name in enumerate(gene_names):
            child_genes = parent.genes.copy()
            # Mutación: +1 o -1 como en Dawkins
            mutation = random.choice([-1, 1])
            child_genes[gene_name] = max(1, min(10, child_genes[gene_name] + mutation))
            
            child = Iusmorfo(
                genes=child_genes,
                generation=self.current_generation + 1,
                parent_id=parent.id
            )
            offspring.append(child)
            
        return offspring
    
    def visualize_iusmorfo(self, iusmorfo: Iusmorfo, ax=None):
        """Visualiza un iusmorfo como Dawkins visualizaba biomorfos"""
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        
        # Limpiar eje
        ax.clear()
        ax.set_xlim(-10, 10)
        ax.set_ylim(-10, 10)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Dibujar estructura basada en genes
        genes = iusmorfo.genes
        
        # Centro (Estado) - tamaño basado en state_role
        state_size = genes['state_role'] / 1.5
        state_color = plt.cm.Blues(genes['state_role'] / 10.0)
        ax.add_patch(Rectangle((-state_size/2, -state_size/2), 
                               state_size, state_size, 
                               fill=True, alpha=0.7, color=state_color))
        
        # Ramas principales (procedimientos) - número basado en procedure
        n_branches = max(3, genes['procedure'])
        branch_colors = plt.cm.Reds(np.linspace(0.3, 0.9, n_branches))
        
        for i in range(n_branches):
            angle = 2 * np.pi * i / n_branches
            # Longitud basada en specificity
            length = 2 + genes['specificity'] * 0.8
            x_end = length * np.cos(angle)
            y_end = length * np.sin(angle)
            
            # Grosor basado en severity
            linewidth = 1 + genes['severity'] * 0.3
            ax.plot([0, x_end], [0, y_end], 
                   color=branch_colors[i % len(branch_colors)], 
                   linewidth=linewidth)
            
            # Nodos en extremos (remedios)
            remedy_size = genes['remedy'] * 0.3
            ax.add_patch(Circle((x_end, y_end), remedy_size, 
                               fill=True, alpha=0.6, 
                               color=plt.cm.Greens(genes['remedy'] / 10.0)))
            
            # Sub-ramas (excepciones) - número basado en exceptions
            n_exceptions = genes['exceptions']
            if n_exceptions > 3:
                for j in range(min(n_exceptions, 6)):
                    sub_angle = angle + (j - n_exceptions/2) * 0.15
                    sub_length = length * 0.4
                    sub_x = x_end + sub_length * np.cos(sub_angle)
                    sub_y = y_end + sub_length * np.sin(sub_angle)
                    ax.plot([x_end, sub_x], [y_end, sub_y], 
                           color='orange', linewidth=1, alpha=0.7)
        
        # Anillos concéntricos (temporality y jurisdiction)
        if genes['temporality'] > 5:
            temp_circles = genes['temporality'] - 4
            for t in range(temp_circles):
                radius = 8 + t * 0.5
                circle = plt.Circle((0, 0), radius, fill=False, 
                                  color='purple', alpha=0.3, linewidth=1)
                ax.add_patch(circle)
        
        if genes['jurisdiction'] > 5:
            juris_rect_size = 9 + genes['jurisdiction'] * 0.3
            ax.add_patch(Rectangle((-juris_rect_size/2, -juris_rect_size/2), 
                                 juris_rect_size, juris_rect_size,
                                 fill=False, color='brown', alpha=0.4, linewidth=2))
        
        # Información textual
        complexity = iusmorfo.calculate_complexity()
        balance = iusmorfo.calculate_balance()
        ax.text(0, -12, f"Complejidad: {complexity:.2f}", 
               ha='center', fontsize=10, weight='bold')
        ax.text(0, -13.5, f"Balance: {balance:.2f}", 
               ha='center', fontsize=8)
        
        return ax
    
    def display_generation(self, offspring: List[Iusmorfo]) -> plt.Figure:
        """Muestra los 9 descendientes en una grilla 3x3"""
        fig, axes = plt.subplots(3, 3, figsize=(18, 18))
        axes = axes.flatten()
        
        gene_names = list(offspring[0].genes.keys())
        
        for i, child in enumerate(offspring):
            self.visualize_iusmorfo(child, axes[i])
            
            # Información detallada
            complexity = child.calculate_complexity()
            balance = child.calculate_balance()
            family = self.identify_legal_family(child)
            
            # Mostrar qué gen cambió
            changed_gene = gene_names[i]
            gene_value = child.genes[changed_gene]
            
            title = f"[{i+1}] {changed_gene.replace('_', ' ').title()}: {gene_value}"
            axes[i].set_title(title, fontsize=12, weight='bold', pad=20)
            
            # Subtítulo con métricas
            subtitle = f"Complejidad: {complexity:.2f} | {family}"
            axes[i].text(0, 11, subtitle, ha='center', fontsize=9, 
                        style='italic', transform=axes[i].transData)
        
        plt.suptitle(f"Generación {self.current_generation + 1} - Elige tu sistema legal preferido", 
                    fontsize=16, weight='bold')
        plt.tight_layout()
        
        # Guardar figura
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f"generation_{self.current_generation + 1}_{timestamp}.png", 
                   dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return fig
    
    def identify_legal_family(self, iusmorfo: Iusmorfo) -> str:
        """Identifica a qué familia legal se parece más"""
        genes = iusmorfo.genes
        
        # Análisis basado en combinaciones de genes dominantes
        if genes['procedure'] > 7 and genes['state_role'] < 5:
            return "Common Law"
        elif genes['specificity'] > 7 and genes['state_role'] > 6:
            return "Civil Law"
        elif genes['state_role'] > 8 and genes['jurisdiction'] > 7:
            return "Socialist Law"
        elif genes['exceptions'] > 7 and genes['procedure'] > 6:
            return "Mixed System"
        elif genes['severity'] > 8:
            return "Authoritarian"
        elif sum(genes.values()) > 65:
            return "Hyper-Complex"
        elif max(genes.values()) - min(genes.values()) > 6:
            return "Specialized"
        else:
            return "Evolving"
    
    def calculate_fitness_score(self, iusmorfo: Iusmorfo) -> float:
        """Calcula puntuación de fitness automática"""
        complexity = iusmorfo.calculate_complexity()
        balance = iusmorfo.calculate_balance()
        diversity = iusmorfo.calculate_diversity()
        
        # Función de fitness balanceada
        fitness = 0.4 * complexity + 0.3 * diversity + 0.3 * balance
        return fitness
    
    def run_interactive_evolution(self, generations: int = 20, auto_mode: bool = False):
        """Ejecuta evolución interactiva con selección por usuario"""
        
        # Inicializar con máxima simple
        ancestor = Iusmorfo(
            genes=self.initial_genes.copy(),
            generation=0
        )
        
        print("="*80)
        print("🧬 EVOLUCIÓN LEGAL - EXPERIMENTO IUSMORFOS 🏛️")
        print("="*80)
        print(f"\n📜 Comenzando con: {ancestor.to_legal_text()}")
        print(f"📊 Complejidad inicial: {ancestor.calculate_complexity():.2f}")
        print(f"⚖️ Balance inicial: {ancestor.calculate_balance():.2f}")
        
        if auto_mode:
            print("\n🤖 Modo automático activado - selección por fitness")
        else:
            print("\n👤 Modo interactivo - tú eliges la evolución")
        
        current_parent = ancestor
        self.history.append(ancestor)
        
        for gen in range(generations):
            self.current_generation = gen
            print(f"\n{'='*80}")
            print(f"🔬 GENERACIÓN {gen + 1} de {generations}")
            print(f"{'='*80}")
            
            # Generar descendientes
            offspring = self.generate_offspring(current_parent)
            
            # Mostrar descendientes
            if not auto_mode:
                self.display_generation(offspring)
            
            # Selección
            if auto_mode:
                # Selección automática por fitness
                fitness_scores = [self.calculate_fitness_score(child) for child in offspring]
                choice_idx = np.argmax(fitness_scores)
                choice = choice_idx + 1
                print(f"\n🎯 Selección automática: Sistema {choice} (fitness: {max(fitness_scores):.3f})")
            else:
                # Selección por usuario
                while True:
                    try:
                        user_input = input("\n🎯 ¿Qué sistema legal prefieres? (1-9, 'auto' para automático, 'q' para salir): ")
                        if user_input.lower() == 'q':
                            print("👋 Evolución terminada por usuario")
                            return self.generate_final_report()
                        elif user_input.lower() == 'auto':
                            fitness_scores = [self.calculate_fitness_score(child) for child in offspring]
                            choice = np.argmax(fitness_scores) + 1
                            print(f"🤖 Selección automática: Sistema {choice}")
                            break
                        else:
                            choice = int(user_input)
                            if 1 <= choice <= 9:
                                break
                    except ValueError:
                        pass
                    print("❌ Por favor elige un número del 1 al 9, 'auto' o 'q'")
            
            # Evolucionar
            selected_child = offspring[choice - 1]
            current_parent = selected_child
            self.history.append(current_parent)
            self.selection_history.append({
                'generation': gen + 1,
                'choice': choice,
                'parent_id': current_parent.parent_id,
                'child_id': current_parent.id,
                'fitness': self.calculate_fitness_score(current_parent)
            })
            
            print(f"\n✅ Sistema elegido: {current_parent.to_legal_text()}")
            print(f"📈 Complejidad: {current_parent.calculate_complexity():.3f}")
            print(f"⚖️ Balance: {current_parent.calculate_balance():.3f}")
            print(f"🏷️ Familia legal: {self.identify_legal_family(current_parent)}")
            
            # Análisis cada 5 generaciones
            if (gen + 1) % 5 == 0:
                self.analyze_evolution()
        
        return self.generate_final_report()
    
    def analyze_evolution(self):
        """Analiza patrones evolutivos emergentes"""
        print("\n" + "="*60)
        print("📊 ANÁLISIS EVOLUTIVO INTERMEDIO")
        print("="*60)
        
        if len(self.history) < 2:
            return
        
        # Trayectoria de complejidad
        complexities = [h.calculate_complexity() for h in self.history]
        growth_rate = (complexities[-1] - complexities[0]) / len(complexities)
        print(f"📈 Complejidad: {complexities[0]:.3f} → {complexities[-1]:.3f} (crecimiento: {growth_rate:.3f}/gen)")
        
        # Identificar familia legal emergente
        latest = self.history[-1]
        family = self.identify_legal_family(latest)
        print(f"🏛️ Familia legal emergente: {family}")
        
        # Genes dominantes
        dominant_gene = max(latest.genes.items(), key=lambda x: x[1])
        recessive_gene = min(latest.genes.items(), key=lambda x: x[1])
        print(f"🔝 Gen dominante: {dominant_gene[0]} (valor: {dominant_gene[1]})")
        print(f"🔻 Gen recesivo: {recessive_gene[0]} (valor: {recessive_gene[1]})")
        
        # Diversidad genética
        diversity = latest.calculate_diversity()
        balance = latest.calculate_balance()
        print(f"🌈 Diversidad: {diversity:.3f} | ⚖️ Balance: {balance:.3f}")
    
    def generate_final_report(self) -> Dict:
        """Genera reporte final completo"""
        if len(self.history) < 2:
            return {}
        
        print("\n" + "="*80)
        print("📋 REPORTE FINAL DE EVOLUCIÓN")
        print("="*80)
        
        initial = self.history[0]
        final = self.history[-1]
        
        # Métricas evolutivas
        initial_complexity = initial.calculate_complexity()
        final_complexity = final.calculate_complexity()
        complexity_growth = ((final_complexity - initial_complexity) / initial_complexity) * 100
        
        # Análisis de trayectoria
        complexities = [h.calculate_complexity() for h in self.history]
        balances = [h.calculate_balance() for h in self.history]
        
        # Emergencia de familias
        families = [self.identify_legal_family(h) for h in self.history]
        family_changes = len(set(families))
        final_family = families[-1]
        
        report = {
            'evolution_summary': {
                'initial_complexity': initial_complexity,
                'final_complexity': final_complexity,
                'complexity_growth_percent': complexity_growth,
                'generations_evolved': len(self.history) - 1,
                'final_legal_family': final_family,
                'family_transitions': family_changes
            },
            'genetic_analysis': {
                'initial_genes': initial.genes,
                'final_genes': final.genes,
                'gene_changes': {k: final.genes[k] - initial.genes[k] for k in initial.genes},
                'dominant_genes': [k for k, v in final.genes.items() if v >= 8],
                'recessive_genes': [k for k, v in final.genes.items() if v <= 2]
            },
            'trajectory_metrics': {
                'complexity_trajectory': complexities,
                'balance_trajectory': balances,
                'family_evolution': families,
                'selection_history': self.selection_history
            }
        }
        
        # Imprimir resumen
        print(f"🎯 Evolución completada: {len(self.history)-1} generaciones")
        print(f"📈 Crecimiento de complejidad: {complexity_growth:.1f}%")
        print(f"🏛️ Familia legal final: {final_family}")
        print(f"🧬 Transiciones familiares: {family_changes}")
        
        print("\n📊 Genes finales:")
        for gene, value in final.genes.items():
            change = final.genes[gene] - initial.genes[gene]
            symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
            print(f"  {symbol} {gene.replace('_', ' ').title()}: {value} ({change:+d})")
        
        print(f"\n📜 Sistema legal final:")
        print(f"   {final.to_legal_text()}")
        
        # Guardar reporte
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"iusmorfo_evolution_report_{timestamp}.json"
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 Reporte guardado en: {filename}")
        
        return report

# Función de utilidad para ejecutar experimentos
def run_dawkins_experiment(generations=20, auto=False):
    """Función principal para ejecutar el experimento"""
    evolution = IusmorfoEvolution()
    return evolution.run_interactive_evolution(generations=generations, auto_mode=auto)

# Función para comparar múltiples evoluciones
def run_comparative_study(num_runs=5, generations=15):
    """Ejecuta múltiples evoluciones automáticas para análisis comparativo"""
    print("🔬 ESTUDIO COMPARATIVO DE EVOLUCIONES")
    print("="*60)
    
    results = []
    families_count = {}
    
    for run in range(num_runs):
        print(f"\n🧪 Ejecutando evolución {run + 1}/{num_runs}...")
        evolution = IusmorfoEvolution()
        report = evolution.run_interactive_evolution(generations=generations, auto_mode=True)
        
        if report:
            results.append(report)
            family = report['evolution_summary']['final_legal_family']
            families_count[family] = families_count.get(family, 0) + 1
    
    # Análisis comparativo
    print(f"\n📊 RESULTADOS COMPARATIVOS ({num_runs} evoluciones)")
    print("="*60)
    
    avg_complexity_growth = np.mean([r['evolution_summary']['complexity_growth_percent'] for r in results])
    print(f"📈 Crecimiento promedio de complejidad: {avg_complexity_growth:.1f}%")
    
    print(f"\n🏛️ Familias legales emergentes:")
    for family, count in sorted(families_count.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / num_runs) * 100
        print(f"   {family}: {count}/{num_runs} ({percentage:.1f}%)")
    
    return results

# Ejecutar experimento si se ejecuta directamente
if __name__ == "__main__":
    print("🧬 IUSMORFOS: Biomorphs de Dawkins aplicados a sistemas legales")
    print("="*60)
    print("1. Experimento interactivo (tú eliges)")
    print("2. Experimento automático (selección por fitness)")
    print("3. Estudio comparativo (múltiples evoluciones automáticas)")
    
    while True:
        try:
            choice = int(input("\n🎯 Elige opción (1-3): "))
            if choice in [1, 2, 3]:
                break
        except ValueError:
            pass
        print("❌ Por favor elige 1, 2 o 3")
    
    if choice == 1:
        run_dawkins_experiment(generations=20, auto=False)
    elif choice == 2:
        run_dawkins_experiment(generations=20, auto=True)
    elif choice == 3:
        run_comparative_study(num_runs=5, generations=15)