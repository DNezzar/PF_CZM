"""
Module de visualisation et post-traitement pour le modèle PF-CZM
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle
import os
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import glob


@dataclass
class PlotSettings:
    """Paramètres de visualisation"""
    figsize: Tuple[float, float] = (12, 8)
    dpi: int = 150
    cmap_damage: str = 'RdBu_r'
    cmap_stress: str = 'jet'
    cmap_displacement: str = 'jet'
    save_plots: bool = True
    display_plots: bool = True
    output_dir: str = 'results/plots'


class PlotManager:
    """Gestionnaire principal des visualisations"""
    
    def __init__(self, mesh_manager, material_manager, settings: PlotSettings = None):
        self.mesh = mesh_manager
        self.materials = material_manager
        self.settings = settings or PlotSettings()
        
        # Créer le répertoire de sortie
        if self.settings.save_plots:
            os.makedirs(self.settings.output_dir, exist_ok=True)
    
    def plot_current_state(self, u: np.ndarray, d: np.ndarray, 
                          interface_damage: List, step: int, time: float,
                          energies: Dict = None) -> None:
        """
        Trace l'état actuel de la simulation avec 3 sous-graphiques
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))
        
        # 1. Magnitude du déplacement
        self._plot_displacement_magnitude(ax1, u)
        
        # 2. Champ d'endommagement
        self._plot_damage_field(ax2, d)
        
        # 3. Endommagement de l'interface
        self._plot_interface_damage(ax3, interface_damage)
        
        # Ajouter les informations d'énergie si disponibles
        if energies:
            energy_text = self._format_energy_text(energies)
            fig.text(0.5, 0.01, energy_text, ha='center', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
        
        # Titre principal
        load_factor = min(time / 1.0, 1.0)  # Supposant ramp_time = 1.0
        max_damage = np.max(d)
        max_interface_damage = self._get_max_interface_damage(interface_damage)
        
        plt.suptitle(
            f'Pas {step}: Temps = {time:.4f}, Facteur de charge = {load_factor:.2f}, '
            f'Endommagement max = {max_damage:.4f}, '
            f'Endommagement interface max = {max_interface_damage:.4f}',
            fontsize=14
        )
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Sauvegarder et/ou afficher
        self._save_and_show(fig, f'step_{step:04d}_time_{time:.4f}')
    
    def plot_stress_profiles(self, stress_data: Dict, step: int, time: float) -> None:
        """
        Trace les profils de contraintes dans le volume et à l'interface
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
        
        # 1. Contrainte normale dans la glace
        self._plot_bulk_stress_profiles(ax1, stress_data['bulk'])
        
        # 2. Contrainte de cisaillement à l'interface
        self._plot_interface_stress_profile(ax2, stress_data['interface'])
        
        # Titre principal
        load_factor = min(time / 1.0, 1.0)
        fig.suptitle(
            f'Profils de contraintes - Pas {step}: Temps = {time:.4f}, '
            f'Facteur de charge = {load_factor:.2f}',
            fontsize=14
        )
        
        plt.tight_layout()
        self._save_and_show(fig, f'stress_profiles_step_{step:04d}_time_{time:.4f}')
    
    def plot_field_at_gauss_points(self, field_data: Dict, field_type: str,
                                  step: int, time: float) -> None:
        """
        Trace un champ aux points de Gauss
        """
        fig, ax = plt.subplots(figsize=self.settings.figsize)
        
        # Données du champ
        x = field_data['x']
        y = field_data['y']
        values = field_data[field_type]
        
        # Configuration selon le type de champ
        if field_type == 'damage':
            title = 'Endommagement du champ de phase'
            cmap = self.settings.cmap_damage
            vmin, vmax = 0, 1
        elif field_type == 'von_mises':
            title = 'Contrainte de von Mises'
            cmap = self.settings.cmap_stress
            vmin, vmax = None, None
        else:
            title = f'Contrainte {field_type}'
            cmap = self.settings.cmap_stress
            vmin, vmax = None, None
        
        # Scatter plot
        scatter = ax.scatter(x, y, c=values, cmap=cmap, vmin=vmin, vmax=vmax,
                           s=20, alpha=0.8)
        plt.colorbar(scatter, ax=ax, label=title)
        
        # Ajouter la ligne d'interface
        interface_y = self.mesh.params.interface_y
        ax.axhline(y=interface_y, color='r', linestyle='--', linewidth=1)
        ax.text(self.mesh.params.length/2, interface_y + 0.2, 'Interface',
               ha='center', color='r')
        
        ax.set_title(f'{title} - Temps: {time:.4f}')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        self._save_and_show(fig, f'{field_type}_step_{step}_time_{time:.4f}')
    
    def plot_final_results(self, history: Dict) -> None:
        """
        Trace les résultats finaux de la simulation
        """
        # Figure principale avec 4 sous-graphiques
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. Endommagement maximum vs temps
        self._plot_damage_evolution(axs[0, 0], history)
        
        # 2. Évolution des énergies
        self._plot_energy_evolution(axs[0, 1], history)
        
        # 3. État final - déplacement
        self._plot_final_displacement(axs[1, 0], history['final_u'])
        
        # 4. État final - endommagement
        self._plot_final_damage(axs[1, 1], history['final_d'])
        
        plt.suptitle('Résumé des résultats de simulation', fontsize=16)
        plt.tight_layout()
        
        self._save_and_show(fig, 'final_results_summary')
        
        # Figure supplémentaire pour le modèle et les lois cohésives
        self._plot_model_setup()
    
    def create_animation(self, input_dir: str = None, output_file: str = 'simulation.mp4',
                        fps: int = 10) -> None:
        """
        Crée une animation à partir des images sauvegardées
        """
        if input_dir is None:
            input_dir = self.settings.output_dir
        
        print(f"Création de l'animation à partir de {input_dir}...")
        
        # Récupérer tous les fichiers d'images
        pattern = os.path.join(input_dir, 'step_*.png')
        image_files = sorted(glob.glob(pattern))
        
        if not image_files:
            print(f"Aucun fichier trouvé avec le motif {pattern}")
            return
        
        print(f"Trouvé {len(image_files)} images")
        
        # Charger toutes les images
        images = []
        for file in image_files:
            img = plt.imread(file)
            images.append(img)
        
        # Créer la figure pour l'animation
        height, width = images[0].shape[:2]
        dpi = 100
        fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
        ax.set_axis_off()
        plt.tight_layout(pad=0)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1)
        
        # Fonction de mise à jour
        def update(frame):
            ax.clear()
            ax.set_axis_off()
            im = ax.imshow(images[frame])
            return [im]
        
        # Créer l'animation
        ani = FuncAnimation(fig, update, frames=len(images), blit=True)
        
        # Sauvegarder avec FFmpeg
        writer = FFMpegWriter(
            fps=fps,
            metadata=dict(artist='PF-CZM Simulation'),
            bitrate=3600
        )
        writer.codec = 'libx264'
        writer.extra_args = ['-pix_fmt', 'yuv420p']
        
        output_path = os.path.join(self.settings.output_dir, output_file)
        ani.save(output_path, writer=writer)
        
        print(f"Animation sauvegardée dans {output_path}")
        plt.close(fig)
    
    def _plot_displacement_magnitude(self, ax, u: np.ndarray) -> None:
        """Trace la magnitude du déplacement"""
        # Calculer la magnitude
        u_mag = np.zeros(self.mesh.num_nodes)
        for node in range(self.mesh.num_nodes):
            ux = u[self.mesh.dof_map_u[node, 0]]
            uy = u[self.mesh.dof_map_u[node, 1]]
            u_mag[node] = np.sqrt(ux**2 + uy**2)
        
        # Scatter plot
        scatter = ax.scatter(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1],
                           c=u_mag, cmap=self.settings.cmap_displacement, s=3)
        plt.colorbar(scatter, ax=ax)
        
        ax.set_title('Magnitude du déplacement (mm)')
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        
        # Ligne d'interface
        self._add_interface_line(ax)
    
    def _plot_damage_field(self, ax, d: np.ndarray) -> None:
        """Trace le champ d'endommagement"""
        scatter = ax.scatter(self.mesh.nodes[:, 0], self.mesh.nodes[:, 1],
                           c=d, cmap=self.settings.cmap_damage, vmin=0, vmax=1, s=3)
        plt.colorbar(scatter, ax=ax)
        
        ax.set_title('Champ d\'endommagement')
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        
        # Ligne d'interface
        self._add_interface_line(ax)
    
    def _plot_interface_damage(self, ax, interface_damage: List) -> None:
        """Trace l'endommagement de l'interface"""
        # Extraire les données d'endommagement
        x_coords = []
        damage_values = []
        
        for elem in interface_damage:
            # Coordonnées des points de Gauss
            nodes = elem['nodes']
            gauss_points = elem['gauss_points']
            damage = elem['damage']
            
            for i, xi in enumerate(gauss_points):
                # Interpolation des coordonnées x
                N1 = (1.0 - xi) / 2.0
                N2 = (1.0 + xi) / 2.0
                
                x1 = self.mesh.nodes[nodes[0], 0]
                x2 = self.mesh.nodes[nodes[1], 0]
                x_gauss = N1 * x1 + N2 * x2
                
                x_coords.append(x_gauss)
                damage_values.append(damage[i])
        
        # Trier par coordonnée x
        sorted_indices = np.argsort(x_coords)
        x_coords = np.array(x_coords)[sorted_indices]
        damage_values = np.array(damage_values)[sorted_indices]
        
        # Tracer
        ax.plot(x_coords, damage_values, 'r-', linewidth=2)
        ax.fill_between(x_coords, 0, damage_values, alpha=0.3, color='red')
        
        ax.set_title('Endommagement de l\'interface')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Endommagement')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
    
    def _plot_bulk_stress_profiles(self, ax, stress_data: Dict) -> None:
        """Trace les profils de contrainte dans le volume"""
        # Différents niveaux y dans la glace
        for level_name, data in stress_data.items():
            if len(data['x']) > 0:
                ax.plot(data['x'], data['sigma_xx'], 'o-', label=level_name,
                       markersize=3)
        
        ax.set_title('Contrainte normale (σ_xx) dans la couche de glace')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('σ_xx (MPa)')
        ax.grid(True, alpha=0.3)
        ax.legend()
    
    def _plot_interface_stress_profile(self, ax, stress_data: Dict) -> None:
        """Trace le profil de contrainte de cisaillement à l'interface"""
        if len(stress_data['x']) > 0:
            ax.plot(stress_data['x'], stress_data['shear'], 'ro-', markersize=3)
        
        ax.set_title('Contrainte de cisaillement à l\'interface')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Contrainte de cisaillement (MPa)')
        ax.grid(True, alpha=0.3)
    
    def _plot_damage_evolution(self, ax, history: Dict) -> None:
        """Trace l'évolution de l'endommagement maximum"""
        time = np.array(history['time'])
        max_damage = np.array(history['max_damage'])
        max_interface_damage = np.array(history['max_interface_damage'])
        
        ax.plot(time, max_damage, 'b-', linewidth=2, label='Volume')
        ax.plot(time, max_interface_damage, 'r-', linewidth=2, label='Interface')
        
        ax.set_xlabel('Temps')
        ax.set_ylabel('Endommagement maximum')
        ax.set_title('Évolution de l\'endommagement')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_ylim(0, 1.05)
    
    def _plot_energy_evolution(self, ax, history: Dict) -> None:
        """Trace l'évolution des énergies"""
        time = np.array(history['time'])
        energies = history['energies']
        
        for energy_type, values in energies.items():
            ax.plot(time, values, label=energy_type.capitalize(), linewidth=2)
        
        ax.set_xlabel('Temps')
        ax.set_ylabel('Énergie')
        ax.set_title('Évolution des énergies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Échelle log si nécessaire
        if np.max(list(energies.values())) / np.min(list(energies.values())) > 1000:
            ax.set_yscale('log')
    
    def _plot_final_displacement(self, ax, u: np.ndarray) -> None:
        """Trace le déplacement final"""
        # Configuration déformée
        scale = 10.0  # Facteur d'amplification
        
        for e in range(self.mesh.num_elements):
            nodes = self.mesh.elements[e]
            x = self.mesh.nodes[nodes, 0]
            y = self.mesh.nodes[nodes, 1]
            
            # Ajouter les déplacements
            ux = np.array([u[self.mesh.dof_map_u[n, 0]] for n in nodes])
            uy = np.array([u[self.mesh.dof_map_u[n, 1]] for n in nodes])
            
            x_def = x + scale * ux
            y_def = y + scale * uy
            
            # Fermer le polygone
            x_def = np.append(x_def, x_def[0])
            y_def = np.append(y_def, y_def[0])
            
            # Couleur selon le matériau
            color = 'lightblue' if self.mesh.material_id[e] == 1 else 'lightgray'
            ax.fill(x_def, y_def, facecolor=color, edgecolor='k', 
                   linewidth=0.5, alpha=0.7)
        
        ax.set_title(f'Configuration déformée (×{scale})')
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
    
    def _plot_final_damage(self, ax, d: np.ndarray) -> None:
        """Trace l'endommagement final avec contours"""
        # Créer une grille pour l'interpolation
        xi = np.linspace(0, self.mesh.params.length, 200)
        yi = np.linspace(0, self.mesh.params.total_height, 100)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpolation (simplifiée - utiliser scipy.interpolate pour une meilleure qualité)
        from scipy.interpolate import griddata
        
        points = self.mesh.nodes
        values = d
        
        Zi = griddata(points, values, (Xi, Yi), method='linear', fill_value=0)
        
        # Contour plot
        contour = ax.contourf(Xi, Yi, Zi, levels=20, cmap=self.settings.cmap_damage,
                            vmin=0, vmax=1)
        plt.colorbar(contour, ax=ax)
        
        # Contours de niveau
        contour_lines = ax.contour(Xi, Yi, Zi, levels=[0.1, 0.5, 0.9], 
                                  colors='black', linewidths=0.5)
        ax.clabel(contour_lines, inline=True, fontsize=8)
        
        ax.set_title('Endommagement final')
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        
        # Ligne d'interface
        self._add_interface_line(ax)
    
    def _plot_model_setup(self) -> None:
        """Trace la configuration du modèle et les lois cohésives"""
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Géométrie et maillage
        self._plot_mesh_geometry(ax1)
        
        # 2. Loi cohésive Mode I
        self._plot_cohesive_law_mode_i(ax2)
        
        # 3. Loi cohésive Mode II
        self._plot_cohesive_law_mode_ii(ax3)
        
        plt.suptitle('Configuration du modèle et lois cohésives', fontsize=16)
        plt.tight_layout()
        
        self._save_and_show(fig, 'model_setup_and_cohesive_laws')
    
    def _plot_mesh_geometry(self, ax) -> None:
        """Trace la géométrie et le maillage"""
        # Tracer les éléments
        for e in range(self.mesh.num_elements):
            nodes = self.mesh.elements[e]
            x = self.mesh.nodes[nodes, 0]
            y = self.mesh.nodes[nodes, 1]
            
            color = 'lightblue' if self.mesh.material_id[e] == 1 else 'lightgray'
            ax.fill(x, y, facecolor=color, edgecolor='k', linewidth=0.2, alpha=0.5)
        
        # Ligne d'interface
        interface_y = self.mesh.params.interface_y
        ax.axhline(y=interface_y, color='r', linestyle='--', linewidth=2)
        ax.text(self.mesh.params.length/2, interface_y + 0.2, 'Interface',
               ha='center', color='r', fontsize=12)
        
        # Conditions aux limites
        bc = self.mesh.get_boundary_conditions()
        for node in bc['fully_fixed']:
            ax.plot(self.mesh.nodes[node, 0], self.mesh.nodes[node, 1],
                   'ks', markersize=4)
        
        # Labels
        ax.text(self.mesh.params.length/2, interface_y + self.mesh.params.ice_height/2,
               'Glace', ha='center', va='center', fontsize=14)
        ax.text(self.mesh.params.length/2, self.mesh.params.substrate_height/2,
               'Substrat', ha='center', va='center', fontsize=14)
        
        # Flèches pour la force centrifuge
        for x in np.linspace(self.mesh.params.length/4, self.mesh.params.length, 4):
            y_ice = interface_y + self.mesh.params.ice_height/2
            y_sub = self.mesh.params.substrate_height/2
            
            ax.arrow(x, y_ice, 5, 0, head_width=0.3, head_length=0.5,
                    fc='blue', ec='blue')
            ax.arrow(x, y_sub, 5, 0, head_width=0.3, head_length=0.5,
                    fc='blue', ec='blue')
        
        ax.set_title('Système glace-substrat avec chargement centrifuge')
        ax.set_aspect('equal')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_xlim(-5, self.mesh.params.length + 10)
        ax.set_ylim(-1, self.mesh.params.total_height + 1)
    
    def _plot_cohesive_law_mode_i(self, ax) -> None:
        """Trace la loi cohésive Mode I"""
        props = self.materials.cohesive
        
        # Plage de déplacement
        delta_n = np.linspace(-0.15, props.normal_deltac * 1.2, 1000)
        traction_n = np.zeros_like(delta_n)
        
        # Calcul des tractions
        for i, dn in enumerate(delta_n):
            if dn >= 0.0:  # Traction
                if dn <= props.normal_delta0:
                    traction_n[i] = props.normal_stiffness * dn
                elif dn <= props.normal_deltac:
                    traction_n[i] = props.normal_strength * \
                                   (props.normal_deltac - dn) / \
                                   (props.normal_deltac - props.normal_delta0)
                else:
                    traction_n[i] = 0.0
            else:  # Compression
                traction_n[i] = props.compression_factor * props.normal_stiffness * dn
        
        # Tracer
        ax.plot(delta_n, traction_n, 'b-', linewidth=2)
        ax.fill_between(delta_n, traction_n, 0, 
                       where=(delta_n >= 0) & (delta_n <= props.normal_deltac),
                       alpha=0.3, color='lightblue')
        
        # Points clés
        ax.plot([0, props.normal_delta0, props.normal_deltac],
               [0, props.normal_strength, 0], 'ro', markersize=6)
        
        # Annotations
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.axvline(x=props.normal_delta0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=props.normal_deltac, color='k', linestyle='--', linewidth=0.5)
        
        ax.text(props.normal_delta0, -0.02, 'δⁿ⁰', ha='center')
        ax.text(props.normal_deltac, -0.02, 'δⁿᶜ', ha='center')
        ax.text(-0.02, props.normal_strength, 'σⁿᶜ', va='center')
        
        ax.set_xlabel('Saut normal δn (mm)')
        ax.set_ylabel('Traction normale σn (MPa)')
        ax.set_title('Loi cohésive Mode I (Normal)')
        ax.grid(True, alpha=0.3)
    
    def _plot_cohesive_law_mode_ii(self, ax) -> None:
        """Trace la loi cohésive Mode II"""
        props = self.materials.cohesive
        
        # Plage de déplacement
        delta_t = np.linspace(-props.shear_deltac * 1.2, props.shear_deltac * 1.2, 1000)
        traction_t = np.zeros_like(delta_t)
        
        # Calcul des tractions
        for i, dt in enumerate(delta_t):
            dt_abs = abs(dt)
            sign = np.sign(dt)
            
            if dt_abs <= props.shear_delta0:
                traction_t[i] = sign * props.shear_stiffness * dt_abs
            elif dt_abs <= props.shear_deltac:
                traction_t[i] = sign * props.shear_strength * \
                               (props.shear_deltac - dt_abs) / \
                               (props.shear_deltac - props.shear_delta0)
            else:
                traction_t[i] = 0.0
        
        # Tracer
        ax.plot(delta_t, traction_t, 'b-', linewidth=2)
        
        # Remplissage
        ax.fill_between(delta_t, traction_t, 0,
                       where=(delta_t >= 0) & (delta_t <= props.shear_deltac),
                       alpha=0.3, color='lightblue')
        ax.fill_between(delta_t, traction_t, 0,
                       where=(delta_t <= 0) & (delta_t >= -props.shear_deltac),
                       alpha=0.3, color='lightblue')
        
        # Points clés
        ax.plot([0, props.shear_delta0, props.shear_deltac,
                -props.shear_delta0, -props.shear_deltac],
               [0, props.shear_strength, 0, -props.shear_strength, 0],
               'ro', markersize=6)
        
        # Annotations
        ax.axhline(y=0, color='k', linewidth=0.5)
        ax.axvline(x=0, color='k', linewidth=0.5)
        ax.axvline(x=props.shear_delta0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=props.shear_deltac, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=-props.shear_delta0, color='k', linestyle='--', linewidth=0.5)
        ax.axvline(x=-props.shear_deltac, color='k', linestyle='--', linewidth=0.5)
        
        ax.text(props.shear_delta0, -0.02, 'δᵗ⁰', ha='center')
        ax.text(props.shear_deltac, -0.02, 'δᵗᶜ', ha='center')
        ax.text(-0.02, props.shear_strength, 'τᶜ', va='center')
        
        ax.set_xlabel('Saut tangentiel δt (mm)')
        ax.set_ylabel('Traction tangentielle τ (MPa)')
        ax.set_title('Loi cohésive Mode II (Cisaillement)')
        ax.grid(True, alpha=0.3)
    
    def _add_interface_line(self, ax) -> None:
        """Ajoute la ligne d'interface à un graphique"""
        interface_y = self.mesh.params.interface_y
        ax.axhline(y=interface_y, color='r', linestyle='--', linewidth=1)
    
    def _format_energy_text(self, energies: Dict) -> str:
        """Formate le texte des énergies"""
        lines = []
        for key, value in energies.items():
            lines.append(f'{key.capitalize()}: {value:.6f}')
        return ' | '.join(lines)
    
    def _get_max_interface_damage(self, interface_damage: List) -> float:
        """Obtient l'endommagement maximal de l'interface"""
        max_damage = 0.0
        for elem in interface_damage:
            elem_max = np.max(elem['damage'])
            max_damage = max(max_damage, elem_max)
        return max_damage
    
    def _save_and_show(self, fig, filename: str) -> None:
        """Sauvegarde et/ou affiche une figure"""
        if self.settings.save_plots:
            filepath = os.path.join(self.settings.output_dir, f'{filename}.png')
            fig.savefig(filepath, dpi=self.settings.dpi, bbox_inches='tight')
        
        if self.settings.display_plots:
            plt.show()
        else:
            plt.close(fig)


class StressFieldCalculator:
    """Calculateur de champs de contraintes"""
    
    def __init__(self, mesh_manager, material_manager, cohesive_manager=None):
        self.mesh = mesh_manager
        self.materials = material_manager
        self.cohesive_manager = cohesive_manager
    
    def calculate_stress_field(self, u: np.ndarray, d: np.ndarray,
                             use_decomposition: bool = False) -> Dict:
        """
        Calcule le champ de contraintes aux points de Gauss
        
        Returns:
            Dictionnaire contenant les coordonnées et composantes de contrainte
        """
        # Initialiser les arrays
        gauss_x = []
        gauss_y = []
        stress_xx = []
        stress_yy = []
        stress_xy = []
        damage_at_gauss = []
        
        # Boucle sur les éléments
        for e in range(self.mesh.num_elements):
            # Calculer les contraintes aux points de Gauss
            elem_data = self._calculate_element_stresses(
                e, u, d, use_decomposition
            )
            
            # Ajouter aux listes
            gauss_x.extend(elem_data['x'])
            gauss_y.extend(elem_data['y'])
            stress_xx.extend(elem_data['sigma_xx'])
            stress_yy.extend(elem_data['sigma_yy'])
            stress_xy.extend(elem_data['sigma_xy'])
            damage_at_gauss.extend(elem_data['damage'])
        
        return {
            'x': np.array(gauss_x),
            'y': np.array(gauss_y),
            'stress_xx': np.array(stress_xx),
            'stress_yy': np.array(stress_yy),
            'stress_xy': np.array(stress_xy),
            'damage': np.array(damage_at_gauss),
            'von_mises': self._calculate_von_mises(
                np.array(stress_xx), np.array(stress_yy), np.array(stress_xy)
            )
        }
    
    def extract_stress_profiles(self, u: np.ndarray, d: np.ndarray) -> Dict:
        """Version améliorée avec extraction robuste"""
        # Calculer le champ de contraintes
        field_data = self.calculate_stress_field(u, d)

        # Profils dans la glace à différentes hauteurs
        interface_y = self.mesh.params.interface_y
        ice_height = self.mesh.params.ice_height

        # Calculer la tolérance basée sur le maillage
        dy_ice = ice_height / self.mesh.params.ny_ice
        tolerance = dy_ice * 0.6  # 60% de la taille d'élément

        y_levels = {
            'Près de l\'interface': interface_y + 0.1 * ice_height,
            'Milieu de la glace': interface_y + 0.5 * ice_height,
            'Près de la surface': interface_y + 0.9 * ice_height
        }

        bulk_profiles = {}
        for name, y_level in y_levels.items():
            profile = self._extract_profile_at_y_improved(
                field_data, y_level, tolerance
            )
            if len(profile['x']) > 0:
                bulk_profiles[name] = profile
            else:
                print(f"  Attention: Aucun point trouvé pour '{name}'")

        # Profil de cisaillement à l'interface
        interface_profile = self._extract_interface_shear_profile(u)

        return {
            'bulk': bulk_profiles,
            'interface': interface_profile
        }
    
    def _calculate_element_stresses(self, elem_id: int, u: np.ndarray, 
                                   d: np.ndarray, use_decomposition: bool) -> Dict:
        """Calcule les contraintes dans un élément"""
        # Récupérer les données de l'élément
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        # Déplacements nodaux
        u_elem = np.zeros(8)
        for i in range(4):
            node = element_nodes[i]
            u_elem[2*i] = u[self.mesh.dof_map_u[node, 0]]
            u_elem[2*i+1] = u[self.mesh.dof_map_u[node, 1]]
        
        # Propriétés matériaux
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        D = self.materials.get_constitutive_matrix(self.mesh.material_id[elem_id])
        
        # Points de Gauss
        gauss_points = [
            (-1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)),
            (-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0))
        ]
        
        elem_data = {
            'x': [], 'y': [],
            'sigma_xx': [], 'sigma_yy': [], 'sigma_xy': [],
            'damage': []
        }
        
        for gp_idx, (xi, eta) in enumerate(gauss_points):
            # Fonctions de forme et dérivées
            N, dN_dx, dN_dy, detJ = self._shape_functions(
                xi, eta, x_coords, y_coords
            )
            
            # Endommagement au point de Gauss
            damage_gauss = np.dot(N, d[element_nodes])
            damage_factor = self.materials.degradation_function(damage_gauss)
            
            # Coordonnées du point de Gauss
            x_gauss = np.dot(N, x_coords)
            y_gauss = np.dot(N, y_coords)
            
            # Matrice B
            B = self._compute_B_matrix(dN_dx, dN_dy)
            
            # Déformation
            strain = B @ u_elem
            
            # Contrainte
            if use_decomposition:
                # Utiliser la décomposition spectrale
                # (À implémenter selon le module materials)
                stress = damage_factor * D @ strain
            else:
                stress = damage_factor * D @ strain
            
            # Stocker les résultats
            elem_data['x'].append(x_gauss)
            elem_data['y'].append(y_gauss)
            elem_data['sigma_xx'].append(stress[0])
            elem_data['sigma_yy'].append(stress[1])
            elem_data['sigma_xy'].append(stress[2])
            elem_data['damage'].append(damage_gauss)
        
        return elem_data
    
    def _shape_functions(self, xi: float, eta: float, x_coords: np.ndarray,
                        y_coords: np.ndarray) -> Tuple:
        """Calcule les fonctions de forme et leurs dérivées"""
        # Fonctions de forme
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        N = np.array([N1, N2, N3, N4])
        
        # Dérivées par rapport à xi et eta
        dN_dxi = np.array([
            -0.25 * (1.0 - eta),
            0.25 * (1.0 - eta),
            0.25 * (1.0 + eta),
            -0.25 * (1.0 + eta)
        ])
        
        dN_deta = np.array([
            -0.25 * (1.0 - xi),
            -0.25 * (1.0 + xi),
            0.25 * (1.0 + xi),
            0.25 * (1.0 - xi)
        ])
        
        # Jacobien
        J = np.zeros((2, 2))
        J[0, 0] = np.dot(dN_dxi, x_coords)
        J[0, 1] = np.dot(dN_dxi, y_coords)
        J[1, 0] = np.dot(dN_deta, x_coords)
        J[1, 1] = np.dot(dN_deta, y_coords)
        
        detJ = np.linalg.det(J)
        invJ = np.linalg.inv(J)
        
        # Dérivées par rapport à x et y
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        
        return N, dN_dx, dN_dy, detJ
    
    def _compute_B_matrix(self, dN_dx: np.ndarray, dN_dy: np.ndarray) -> np.ndarray:
        """Calcule la matrice B pour le calcul des déformations"""
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dN_dx[i]      # du/dx
            B[1, 2*i+1] = dN_dy[i]    # dv/dy
            B[2, 2*i] = dN_dy[i]      # du/dy
            B[2, 2*i+1] = dN_dx[i]    # dv/dx
        return B
    
    def _calculate_von_mises(self, sigma_xx: np.ndarray, sigma_yy: np.ndarray,
                           sigma_xy: np.ndarray) -> np.ndarray:
        """Calcule la contrainte de von Mises"""
        return np.sqrt(sigma_xx**2 + sigma_yy**2 - sigma_xx*sigma_yy + 3*sigma_xy**2)
    
    def _extract_profile_at_y_improved(self, field_data: Dict, y_level: float,
                                      tolerance: float = None) -> Dict:
        """
        Extrait un profil de contrainte à une hauteur donnée
        avec recherche des points les plus proches
        
        Parameters:
            field_data: Données du champ de contraintes
            y_level: Hauteur cible
            tolerance: Tolérance de recherche (auto si None)
        """
        # Déterminer la tolérance automatiquement si non fournie
        if tolerance is None:
            # Estimer l'espacement moyen entre points de Gauss en y
            y_unique = np.unique(field_data['y'])
            if len(y_unique) > 1:
                dy_mean = np.mean(np.diff(np.sort(y_unique)))
                tolerance = dy_mean * 0.75  # 75% de l'espacement moyen
            else:
                tolerance = self.mesh.params.ice_height * 0.05
        
        # Obtenir les coordonnées x uniques
        x_unique = np.unique(field_data['x'])
        
        # Résultats
        x_profile = []
        sigma_xx_profile = []
        
        # Pour chaque position x, trouver le point de Gauss le plus proche en y
        for x_target in x_unique:
            # Filtrer les points à cette coordonnée x (avec petite tolérance)
            x_tol = 1e-6
            mask_x = np.abs(field_data['x'] - x_target) < x_tol
            
            if not np.any(mask_x):
                continue
            
            # Points à cette position x
            y_at_x = field_data['y'][mask_x]
            sigma_at_x = field_data['stress_xx'][mask_x]
            
            # Trouver le point le plus proche de y_level
            distances = np.abs(y_at_x - y_level)
            min_dist = np.min(distances)
            
            # Vérifier si dans la tolérance
            if min_dist <= tolerance:
                idx_closest = np.argmin(distances)
                
                # Interpolation linéaire si on a plusieurs points proches
                mask_near = distances <= tolerance
                if np.sum(mask_near) > 1:
                    # Interpolation pondérée par distance inverse
                    y_near = y_at_x[mask_near]
                    sigma_near = sigma_at_x[mask_near]
                    weights = 1.0 / (distances[mask_near] + 1e-10)
                    weights /= np.sum(weights)
                    
                    sigma_interp = np.sum(weights * sigma_near)
                    x_profile.append(x_target)
                    sigma_xx_profile.append(sigma_interp)
                else:
                    # Un seul point, pas d'interpolation
                    x_profile.append(x_target)
                    sigma_xx_profile.append(sigma_at_x[idx_closest])
        
        # Convertir en arrays et trier
        if x_profile:
            x_profile = np.array(x_profile)
            sigma_xx_profile = np.array(sigma_xx_profile)
            
            # Trier par x
            sort_idx = np.argsort(x_profile)
            x_profile = x_profile[sort_idx]
            sigma_xx_profile = sigma_xx_profile[sort_idx]
            
            # Lisser légèrement pour réduire les oscillations
            if len(sigma_xx_profile) > 5:
                from scipy.ndimage import uniform_filter1d
                sigma_xx_profile = uniform_filter1d(sigma_xx_profile, size=3, mode='nearest')
        
        print(f"Profil extrait à y={y_level:.3f} avec {len(x_profile)} points")
        
        return {
            'x': x_profile,
            'sigma_xx': sigma_xx_profile,
            'y_actual': np.full_like(x_profile, y_level),
            'tolerance_used': tolerance
        }
    
    def _extract_interface_shear_profile(self, u: np.ndarray) -> Dict:
        """Extrait le profil de cisaillement à l'interface"""
        from cohesive_zone import CohesiveZoneManager
        cohesive = CohesiveZoneManager(self.materials, self.mesh)
        
        x_coords = []
        shear_stress = []
        
        for elem in self.mesh.cohesive_elements:
            for j, xi in enumerate(elem.gauss_points):
                # Position x
                N1 = (1.0 - xi) / 2.0
                N2 = (1.0 + xi) / 2.0
                x_gauss = N1 * self.mesh.nodes[elem.nodes[0], 0] + N2 * self.mesh.nodes[elem.nodes[1], 0]
                
                # Calculer vraiment les contraintes
                delta_n, delta_t, _ = cohesive._compute_local_kinematics(elem.nodes, xi, u)
                traction = cohesive._compute_tractions(delta_n, delta_t, elem.damage[j])
                
                x_coords.append(x_gauss)
                shear_stress.append(traction.tangential)  # Au lieu de 0.0 !
        
        if x_coords:
            sort_idx = np.argsort(x_coords)
            return {'x': np.array(x_coords)[sort_idx], 'shear': np.array(shear_stress)[sort_idx]}
        return {'x': [], 'shear': []}