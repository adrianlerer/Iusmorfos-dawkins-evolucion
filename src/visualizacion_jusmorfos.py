#!/usr/bin/env python3
"""
VISUALIZACIÓN DE JUSMORFOS - Representación visual de sistemas legales

Módulo complementario para visualizar "jusmorfos" (sistemas legales) de forma
gráfica, similar a los biomorfos originales de Dawkins.

Author: AI Assistant (Genspark/Claude)
Date: 2025-09-21
Version: 1.0 - Visualización Jusmorfos
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle, Polygon
import seaborn as sns
from typing import List, Dict, Any
from biomorfos_legales_dawkins import Jusmorfo, GenLegal
import matplotlib.patches as mpatches

class VisualizadorJusmorfos:
    """Visualizador de sistemas legales como formas geométricas"""
    
    def __init__(self):
        self.colores_familias = {
            'Common Law': '#FF6B6B',
            'Civil Law': '#4ECDC4', 
            'Derecho Comunitario': '#45B7D1',
            'Derecho Autoritario': '#FFA07A',
            'Derecho Transnacional': '#98D8C8',
            'Derecho Económico': '#F7DC6F',
            'Sistema Híbrido': '#BB8FCE',
            'Primitivo': '#D5DBDB'
        }
        
    def generar_forma_jusmorfo(self, jusmorfo: Jusmorfo) -> Dict[str, Any]:
        """Genera la forma visual de un jusmorfo basada en sus genes"""
        
        genes = jusmorfo.gen.to_vector()
        
        # Características geométricas basadas en genes
        forma = {
            'tipo': self._determinar_tipo_forma(genes),
            'tamaño': genes[0] / 2,  # Formalismo determina tamaño base
            'complejidad_contorno': int(genes[5]),  # Complejidad procesal
            'simetría': genes[1] / 10,  # Centralización determina simetría
            'apertura': genes[7] / 10,  # Internacionalización determina apertura
            'densidad': genes[2] / 10,  # Codificación determina densidad
            'ramificaciones': int(genes[8]),  # Digitalización determina ramificaciones
            'color': self.colores_familias.get(jusmorfo.familia_legal, '#808080')
        }
        
        return forma
    
    def _determinar_tipo_forma(self, genes: np.ndarray) -> str:
        """Determina el tipo de forma geométrica básica"""
        
        # Algoritmo para determinar forma basado en combinaciones de genes
        formalismo = genes[0]
        centralización = genes[1]
        codificación = genes[2]
        
        if formalismo <= 3 and centralización <= 3:
            return 'orgánica'  # Sistemas flexibles y descentralizados
        elif formalismo >= 7 and centralización >= 7:
            return 'geométrica'  # Sistemas formales y centralizados
        elif codificación >= 7:
            return 'estructural'  # Sistemas altamente codificados
        else:
            return 'híbrida'  # Sistemas mixtos
    
    def dibujar_jusmorfo(self, jusmorfo: Jusmorfo, ax: plt.Axes, posición: tuple = (0, 0)):
        """Dibuja un jusmorfo individual en los ejes dados"""
        
        forma = self.generar_forma_jusmorfo(jusmorfo)
        x, y = posición
        
        # Dibujar forma base según tipo
        if forma['tipo'] == 'orgánica':
            self._dibujar_forma_orgánica(ax, forma, x, y)
        elif forma['tipo'] == 'geométrica':
            self._dibujar_forma_geométrica(ax, forma, x, y)
        elif forma['tipo'] == 'estructural':
            self._dibujar_forma_estructural(ax, forma, x, y)
        else:
            self._dibujar_forma_híbrida(ax, forma, x, y)
        
        # Agregar etiqueta
        ax.text(x, y - forma['tamaño'] - 0.5, jusmorfo.nombre, 
                ha='center', va='top', fontsize=8, weight='bold')
        
        # Mostrar complejidad como número
        ax.text(x, y + forma['tamaño'] + 0.3, f"C: {jusmorfo.complejidad:.1f}", 
                ha='center', va='bottom', fontsize=7)
    
    def _dibujar_forma_orgánica(self, ax: plt.Axes, forma: Dict, x: float, y: float):
        """Dibuja forma orgánica (sistemas flexibles)"""
        
        # Círculo base con perturbaciones
        ángulos = np.linspace(0, 2*np.pi, forma['complejidad_contorno'] + 3)
        radio_base = forma['tamaño']
        
        # Generar perturbaciones orgánicas
        radios = []
        for ángulo in ángulos:
            perturbación = 0.3 * np.sin(3 * ángulo) * (1 - forma['simetría'])
            radio = radio_base + perturbación
            radios.append(radio)
        
        # Coordenadas del contorno
        xs = [x + r * np.cos(a) for a, r in zip(ángulos, radios)]
        ys = [y + r * np.sin(a) for a, r in zip(ángulos, radios)]
        
        # Dibujar polígono
        poly = Polygon(list(zip(xs, ys)), facecolor=forma['color'], 
                      alpha=0.7, edgecolor='black', linewidth=1)
        ax.add_patch(poly)
        
        # Agregar ramificaciones si hay digitalización
        if forma['ramificaciones'] > 5:
            self._agregar_ramificaciones(ax, x, y, forma)
    
    def _dibujar_forma_geométrica(self, ax: plt.Axes, forma: Dict, x: float, y: float):
        """Dibuja forma geométrica (sistemas formales)"""
        
        lado = forma['tamaño'] * 2
        
        if forma['complejidad_contorno'] <= 4:
            # Cuadrado o rectángulo
            rect = Rectangle((x - lado/2, y - lado/2), lado, lado,
                           facecolor=forma['color'], alpha=0.7, 
                           edgecolor='black', linewidth=2)
            ax.add_patch(rect)
        else:
            # Polígono regular
            n_lados = min(forma['complejidad_contorno'], 8)
            ángulos = np.linspace(0, 2*np.pi, n_lados + 1)
            xs = [x + forma['tamaño'] * np.cos(a) for a in ángulos]
            ys = [y + forma['tamaño'] * np.sin(a) for a in ángulos]
            
            poly = Polygon(list(zip(xs, ys)), facecolor=forma['color'],
                          alpha=0.7, edgecolor='black', linewidth=2)
            ax.add_patch(poly)
        
        # Estructura interna para formalismo alto
        if forma['densidad'] > 0.7:
            self._agregar_estructura_interna(ax, x, y, forma)
    
    def _dibujar_forma_estructural(self, ax: plt.Axes, forma: Dict, x: float, y: float):
        """Dibuja forma estructural (sistemas codificados)"""
        
        # Estructura de árbol o red
        altura = forma['tamaño'] * 2
        niveles = min(forma['complejidad_contorno'], 4)
        
        # Tronco principal
        ax.plot([x, x], [y - altura/2, y + altura/2], 
                color='black', linewidth=3)
        
        # Ramas por niveles
        for nivel in range(1, niveles + 1):
            y_nivel = y + altura/2 - (nivel * altura / niveles)
            ramas = nivel + 1
            
            for i in range(ramas):
                x_rama = x + (i - ramas/2 + 0.5) * (forma['tamaño'] / ramas)
                ax.plot([x, x_rama], [y_nivel, y_nivel - altura/(niveles*2)],
                       color=forma['color'], linewidth=2, alpha=0.8)
                
                # Nodo terminal
                circle = Circle((x_rama, y_nivel - altura/(niveles*2)), 
                              0.1, facecolor=forma['color'], 
                              edgecolor='black')
                ax.add_patch(circle)
    
    def _dibujar_forma_híbrida(self, ax: plt.Axes, forma: Dict, x: float, y: float):
        """Dibuja forma híbrida (sistemas mixtos)"""
        
        # Combinación de elementos geométricos y orgánicos
        
        # Base geométrica
        rect = Rectangle((x - forma['tamaño']/2, y - forma['tamaño']/2), 
                        forma['tamaño'], forma['tamaño'],
                        facecolor=forma['color'], alpha=0.5,
                        edgecolor='black', linewidth=1)
        ax.add_patch(rect)
        
        # Elementos orgánicos superpuestos
        ángulos = np.linspace(0, 2*np.pi, 8)
        for i, ángulo in enumerate(ángulos):
            if i % 2 == 0:  # Solo algunos elementos
                x_elem = x + (forma['tamaño']/3) * np.cos(ángulo)
                y_elem = y + (forma['tamaño']/3) * np.sin(ángulo)
                
                circle = Circle((x_elem, y_elem), forma['tamaño']/6,
                              facecolor=forma['color'], alpha=0.7,
                              edgecolor='gray')
                ax.add_patch(circle)
    
    def _agregar_ramificaciones(self, ax: plt.Axes, x: float, y: float, forma: Dict):
        """Agrega ramificaciones digitales"""
        
        n_ramas = min(forma['ramificaciones'] - 5, 6)
        for i in range(n_ramas):
            ángulo = (2 * np.pi * i) / n_ramas
            x_rama = x + (forma['tamaño'] * 1.5) * np.cos(ángulo)
            y_rama = y + (forma['tamaño'] * 1.5) * np.sin(ángulo)
            
            # Línea de conexión
            ax.plot([x, x_rama], [y, y_rama], 
                   color='blue', linewidth=1, alpha=0.6, linestyle='--')
            
            # Nodo digital
            circle = Circle((x_rama, y_rama), 0.1, 
                          facecolor='blue', alpha=0.8)
            ax.add_patch(circle)
    
    def _agregar_estructura_interna(self, ax: plt.Axes, x: float, y: float, forma: Dict):
        """Agrega estructura interna para sistemas formales"""
        
        # Rejilla interna
        n_líneas = int(forma['densidad'] * 5)
        for i in range(1, n_líneas):
            # Líneas verticales
            x_línea = x - forma['tamaño'] + (2 * forma['tamaño'] * i / n_líneas)
            ax.plot([x_línea, x_línea], 
                   [y - forma['tamaño'], y + forma['tamaño']],
                   color='black', linewidth=0.5, alpha=0.5)
            
            # Líneas horizontales
            y_línea = y - forma['tamaño'] + (2 * forma['tamaño'] * i / n_líneas)
            ax.plot([x - forma['tamaño'], x + forma['tamaño']], 
                   [y_línea, y_línea],
                   color='black', linewidth=0.5, alpha=0.5)
    
    def visualizar_generación(self, jusmorfos: List[Jusmorfo], título: str = "Generación de Jusmorfos"):
        """Visualiza una generación completa de jusmorfos"""
        
        n_jusmorfos = len(jusmorfos)
        cols = min(3, n_jusmorfos)
        rows = (n_jusmorfos + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 4*rows))
        if rows == 1 and cols == 1:
            axes = [axes]
        elif rows == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for i, jusmorfo in enumerate(jusmorfos):
            ax = axes[i] if i < len(axes) else axes[-1]
            
            # Configurar ejes
            ax.set_xlim(-3, 3)
            ax.set_ylim(-3, 3)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Dibujar jusmorfo
            self.dibujar_jusmorfo(jusmorfo, ax)
            
            # Título del subplot
            ax.set_title(f"{jusmorfo.familia_legal}\nGenes: {jusmorfo.gen.to_vector()}", 
                        fontsize=10)
        
        # Ocultar ejes extra
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle(título, fontsize=16, weight='bold')
        plt.tight_layout()
        return fig
    
    def visualizar_árbol_evolutivo(self, árbol_evolutivo: Dict, historia: List[Dict]):
        """Visualiza el árbol evolutivo completo"""
        
        fig, ax = plt.subplots(1, 1, figsize=(15, 10))
        
        # Organizar por generaciones
        generaciones = {}
        for entrada in historia:
            gen = entrada['generación']
            if gen not in generaciones:
                generaciones[gen] = []
            generaciones[gen].append(entrada)
        
        # Dibujar árbol
        max_gen = max(generaciones.keys()) if generaciones else 0
        
        for gen, sistemas in generaciones.items():
            y_pos = max_gen - gen  # Invertir para mostrar evolución hacia arriba
            
            for i, sistema in enumerate(sistemas):
                x_pos = i - len(sistemas)/2 + 0.5
                
                # Dibujar nodo
                color = self.colores_familias.get(sistema['familia'], '#808080')
                circle = Circle((x_pos, y_pos), 0.3, 
                              facecolor=color, alpha=0.7,
                              edgecolor='black', linewidth=1)
                ax.add_patch(circle)
                
                # Etiqueta
                ax.text(x_pos, y_pos, f"G{gen}", ha='center', va='center',
                       fontsize=8, weight='bold')
                
                # Línea al padre (si no es la primera generación)
                if gen > 1:
                    ax.plot([x_pos, 0], [y_pos, y_pos + 1], 
                           'k-', alpha=0.5, linewidth=1)
        
        ax.set_xlim(-5, 5)
        ax.set_ylim(-1, max_gen + 1)
        ax.set_xlabel('Linaje Evolutivo')
        ax.set_ylabel('Generación')
        ax.set_title('Árbol Evolutivo de Sistemas Legales', fontsize=16, weight='bold')
        
        # Leyenda de familias
        handles = [mpatches.Patch(color=color, label=familia) 
                  for familia, color in self.colores_familias.items()]
        ax.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def comparar_con_sistemas_reales(self, jusmorfos_evolucionados: List[Jusmorfo]) -> Dict[str, Any]:
        """Compara sistemas evolucionados con sistemas legales reales conocidos"""
        
        # Sistemas legales reales de referencia
        sistemas_reales = {
            'Estados Unidos': GenLegal(3, 4, 3, 8, 7, 6, 8, 6, 7),
            'Reino Unido': GenLegal(4, 6, 2, 7, 5, 5, 7, 7, 6),
            'Alemania': GenLegal(8, 6, 9, 6, 4, 7, 8, 8, 8),
            'Francia': GenLegal(9, 8, 9, 5, 5, 6, 7, 8, 7),
            'Japón': GenLegal(7, 7, 8, 4, 3, 8, 9, 6, 9),
            'Brasil': GenLegal(7, 5, 8, 5, 6, 7, 6, 5, 5),
            'China': GenLegal(6, 9, 7, 3, 8, 5, 9, 4, 8),
            'Singapur': GenLegal(8, 8, 6, 6, 7, 8, 9, 9, 9)
        }
        
        # Comparar cada jusmorfo evolucionado con sistemas reales
        comparaciones = {}
        
        for jusmorfo in jusmorfos_evolucionados:
            distancias = {}
            
            for nombre_real, gen_real in sistemas_reales.items():
                distancia = jusmorfo.gen.distancia_euclidiana(gen_real)
                distancias[nombre_real] = distancia
            
            # Encontrar el sistema real más similar
            sistema_más_similar = min(distancias, key=distancias.get)
            distancia_mínima = distancias[sistema_más_similar]
            
            comparaciones[jusmorfo.nombre] = {
                'sistema_similar': sistema_más_similar,
                'distancia': distancia_mínima,
                'similitud_porcentaje': max(0, (1 - distancia_mínima/15) * 100),  # Normalizada
                'todas_distancias': distancias
            }
        
        return comparaciones

def demo_visualización():
    """Demostración de las capacidades de visualización"""
    
    print("🎨 DEMO: VISUALIZACIÓN DE JUSMORFOS")
    print("=" * 50)
    
    # Crear algunos jusmorfos de ejemplo
    from biomorfos_legales_dawkins import GenLegal, Jusmorfo
    
    # Sistema primitivo
    gen_primitivo = GenLegal(1, 1, 1, 1, 1, 1, 1, 1, 1)
    jusmorfo_primitivo = Jusmorfo(gen_primitivo, "Primitivo")
    
    # Common Law evolucionado
    gen_common = GenLegal(3, 4, 2, 8, 6, 5, 7, 6, 5)
    jusmorfo_common = Jusmorfo(gen_common, "Common Law")
    
    # Civil Law evolucionado
    gen_civil = GenLegal(8, 7, 9, 6, 4, 7, 7, 7, 6)
    jusmorfo_civil = Jusmorfo(gen_civil, "Civil Law")
    
    # Sistema híbrido
    gen_híbrido = GenLegal(5, 5, 5, 5, 5, 5, 5, 5, 5)
    jusmorfo_híbrido = Jusmorfo(gen_híbrido, "Híbrido")
    
    jusmorfos_demo = [jusmorfo_primitivo, jusmorfo_common, jusmorfo_civil, jusmorfo_híbrido]
    
    # Crear visualizador
    visualizador = VisualizadorJusmorfos()
    
    # Visualizar generación
    fig = visualizador.visualizar_generación(jusmorfos_demo, "Demo: Tipos de Sistemas Legales")
    plt.savefig('demo_jusmorfos.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Comparar con sistemas reales
    comparaciones = visualizador.comparar_con_sistemas_reales(jusmorfos_demo)
    
    print("\nComparación con sistemas legales reales:")
    for jusmorfo_nombre, comp in comparaciones.items():
        print(f"• {jusmorfo_nombre}: Similar a {comp['sistema_similar']} ({comp['similitud_porcentaje']:.1f}% similitud)")
    
    return comparaciones

if __name__ == "__main__":
    demo_visualización()