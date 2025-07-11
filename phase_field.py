"""
Module de gestion du champ de phase pour la rupture
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class PhaseFieldParameters:
    """Paramètres du modèle de champ de phase"""
    l0: float = 1.0                    # Longueur caractéristique
    k_res: float = 1.0e-10            # Rigidité résiduelle
    irreversibility: bool = True       # Appliquer l'irréversibilité
    threshold: float = 1e-6             # Seuil minimal d'endommagement
    max_damage: float = 1.0            # Endommagement maximal
    
    def degradation_function(self, d: float) -> float:
        """Fonction de dégradation g(d) = (1-d)² + k_res"""
        return (1.0 - d)**2 + self.k_res


class HistoryField:
    """Gestion du champ d'histoire pour l'irréversibilité"""
    
    def __init__(self, num_elements: int, num_gauss_points: int = 4):
        self.num_elements = num_elements
        self.num_gauss_points = num_gauss_points
        
        # Stockage aux points de Gauss
        self.H_gauss = np.zeros((num_elements, num_gauss_points), dtype=np.float64)
        
        # Stockage nodal pour visualisation
        self.H_nodal = None
        
    def update(self, element_id: int, gauss_point: int, psi_plus: float):
        """Met à jour l'histoire à un point de Gauss"""
        self.H_gauss[element_id, gauss_point] = max(
            self.H_gauss[element_id, gauss_point], 
            psi_plus
        )
    
    def update_element(self, element_id: int, psi_plus_element: np.ndarray):
        """Met à jour l'histoire pour un élément spécifique"""
        self.H_gauss[element_id] = np.maximum(
            self.H_gauss[element_id], 
            psi_plus_element
        )
    
    def update_all(self, psi_plus_all: np.ndarray):
        """Met à jour l'histoire pour tous les points de Gauss"""
        self.H_gauss = np.maximum(self.H_gauss, psi_plus_all)
    
    def get_element_history(self, element_id: int) -> np.ndarray:
        """Retourne l'histoire pour un élément"""
        return self.H_gauss[element_id]
    
    def project_to_nodes(self, mesh, elements):
        """Projette l'histoire des points de Gauss vers les nœuds"""
        num_nodes = mesh.num_nodes
        self.H_nodal = np.zeros(num_nodes, dtype=np.float64)
        node_count = np.zeros(num_nodes, dtype=np.float64)
        
        # Moyenner les valeurs des points de Gauss aux nœuds
        for e in range(self.num_elements):
            element_nodes = elements[e]
            avg_history = np.mean(self.H_gauss[e])
            
            for node in element_nodes:
                self.H_nodal[node] += avg_history
                node_count[node] += 1
        
        # Moyenner
        mask = node_count > 0
        self.H_nodal[mask] /= node_count[mask]
        
        return self.H_nodal


class PhaseFieldSolver:
    """Solveur pour le problème de champ de phase"""
    
    def __init__(self, mesh_manager, material_manager, energy_calculator,
                 params: PhaseFieldParameters = None):
        self.mesh = mesh_manager
        self.materials = material_manager
        self.energy_calc = energy_calculator
        self.params = params or PhaseFieldParameters()
        
        # Champ d'histoire
        self.history = HistoryField(self.mesh.num_elements)
        
        # Cache pour les matrices
        self._matrix_cache = {}
        
    def solve(self, u: np.ndarray, d_prev: np.ndarray) -> np.ndarray:
        """
        Résout le problème de champ de phase
        
        Parameters:
            u: Déplacements actuels
            d_prev: Endommagement précédent
            
        Returns:
            d: Nouvel endommagement
        """
        # Mettre à jour le champ d'histoire
        self._update_history_field(u)
        
        # Assembler les matrices
        A, b = self.assemble_system()
        
        # Résoudre le système linéaire
        d = spsolve(A, b)
        
        # Appliquer les contraintes
        d = self._apply_constraints(d, d_prev)
        
        return d
    
    def assemble_system(self) -> Tuple[csr_matrix, np.ndarray]:
        """
        Assemble le système linéaire pour le champ de phase
        
        Returns:
            A: Matrice du système
            b: Vecteur second membre
        """
        # Initialiser les matrices
        A = lil_matrix((self.mesh.num_dofs_d, self.mesh.num_dofs_d), dtype=np.float64)
        b = np.zeros(self.mesh.num_dofs_d, dtype=np.float64)
        
        # Boucle sur les éléments
        for e in range(self.mesh.num_elements):
            # Matrices élémentaires
            A_elem, b_elem = self._get_element_matrices(e)
            
            # Assemblage
            element_nodes = self.mesh.elements[e]
            dofs = self.mesh.dof_map_d[element_nodes]
            
            for i in range(4):
                for j in range(4):
                    A[dofs[i], dofs[j]] += A_elem[i, j]
                b[dofs[i]] += b_elem[i]
        
        # Convertir en format CSR pour la résolution
        A = A.tocsr()
        
        return A, b
    
    def _get_element_matrices(self, elem_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Calcule les matrices élémentaires pour le champ de phase"""
        # Récupérer les nœuds et coordonnées
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        # Propriétés matériaux
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        Gc = mat_props.Gc
        l0 = self.params.l0
        
        # Initialiser les matrices élémentaires
        A_elem = np.zeros((4, 4), dtype=np.float64)
        b_elem = np.zeros(4, dtype=np.float64)
        
        # Points de Gauss
        gauss_points = [
            (-1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)),
            (-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0))
        ]
        
        # Intégration sur les points de Gauss
        for gp_idx, (xi, eta) in enumerate(gauss_points):
            # Fonctions de forme et dérivées
            N, dN_dx, dN_dy, detJ = self._shape_functions_and_derivatives(
                xi, eta, x_coords, y_coords
            )
            
            # Matrice gradient
            B_d = np.zeros((2, 4))
            B_d[0, :] = dN_dx
            B_d[1, :] = dN_dy
            
            # Récupérer l'histoire à ce point de Gauss
            H_gauss = self.history.H_gauss[elem_id, gp_idx]
            
            # Contributions aux matrices
            # Terme de masse : (2H + Gc/l0) * N^T * N
            mass_term = (2.0 * H_gauss + Gc / l0)
            A_elem += np.outer(N, N) * mass_term * detJ
            
            # Terme de rigidité : Gc * l0 * B^T * B
            A_elem += B_d.T @ B_d * Gc * l0 * detJ
            
            # Second membre : 2H * N
            b_elem += N * 2.0 * H_gauss * detJ
        
        return A_elem, b_elem
    
    def _update_history_field(self, u: np.ndarray):
        """Met à jour le champ d'histoire basé sur l'énergie de déformation"""
        # CORRECTION : Mettre à jour élément par élément
        for e in range(self.mesh.num_elements):
            # Calculer les densités d'énergie aux points de Gauss
            psi_plus_gauss = self.energy_calc.calculate_strain_energy_density_at_gauss_points(
                e, u, use_decomposition=True
            )
            
            # Mettre à jour l'histoire pour CET élément spécifique
            self.history.update_element(e, psi_plus_gauss)
    
    def _apply_constraints(self, d: np.ndarray, d_prev: np.ndarray) -> np.ndarray:
        """Applique les contraintes sur l'endommagement"""
        # Bornes [0, 1]
        d = np.clip(d, 0.0, self.params.max_damage)
        
        # Irréversibilité
        if self.params.irreversibility:
            d = np.maximum(d, d_prev)
        
        # Seuil minimal
        if self.params.threshold > 0:
            d = np.where(d < self.params.threshold, 0.0, d)
        
        return d.astype(np.float64)
    
    def _shape_functions_and_derivatives(self, xi: float, eta: float,
                                       x_coords: np.ndarray, 
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
        
        # Protection contre les jacobiens singuliers
        if abs(detJ) < 1e-12:
            detJ = 1e-12 if detJ >= 0 else -1e-12
        
        invJ = np.linalg.inv(J)
        
        # Dérivées par rapport à x et y
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        
        return N, dN_dx, dN_dy, detJ
    
    def get_crack_path(self, d: np.ndarray, threshold: float = 0.95) -> Dict:
        """
        Extrait le chemin de fissure basé sur l'endommagement
        
        Parameters:
            d: Champ d'endommagement
            threshold: Seuil pour considérer un point comme fissuré
            
        Returns:
            Dictionnaire avec les coordonnées du chemin de fissure
        """
        crack_nodes = []
        crack_coords = []
        
        # Trouver les nœuds avec endommagement élevé
        for i in range(self.mesh.num_nodes):
            if d[i] >= threshold:
                crack_nodes.append(i)
                crack_coords.append(self.mesh.nodes[i])
        
        crack_coords = np.array(crack_coords) if crack_coords else np.empty((0, 2))
        
        return {
            'nodes': crack_nodes,
            'coordinates': crack_coords,
            'num_crack_nodes': len(crack_nodes)
        }
    
    def estimate_crack_length(self, d: np.ndarray, threshold: float = 0.5) -> float:
        """
        Estime la longueur de fissure basée sur le champ d'endommagement
        
        Returns:
            Longueur estimée de la fissure
        """
        crack_length = 0.0
        
        # Méthode simple : intégrer l'endommagement
        for e in range(self.mesh.num_elements):
            element_nodes = self.mesh.elements[e]
            elem_damage = np.mean(d[element_nodes])
            
            if elem_damage > threshold:
                # Ajouter la contribution de cet élément
                elem_size = self.mesh.get_element_size(e)
                crack_length += elem_damage * np.sqrt(elem_size)
        
        return crack_length


class DamageEvolutionTracker:
    """Suivi de l'évolution de l'endommagement"""
    
    def __init__(self):
        self.history = []
        self.crack_tips = []
        
    def add_state(self, time: float, d: np.ndarray, crack_info: Dict):
        """Ajoute un état d'endommagement"""
        state = {
            'time': time,
            'max_damage': np.max(d),
            'mean_damage': np.mean(d),
            'damaged_nodes': np.sum(d > 0.01),
            'crack_length': crack_info.get('length', 0.0),
            'crack_nodes': crack_info.get('num_crack_nodes', 0)
        }
        
        self.history.append(state)
        
        # Détecter les pointes de fissure
        if crack_info.get('coordinates') is not None and len(crack_info['coordinates']) > 0:
            self._detect_crack_tips(crack_info['coordinates'])
    
    def _detect_crack_tips(self, crack_coords: np.ndarray):
        """Détecte les pointes de fissure"""
        if len(crack_coords) < 2:
            return
        
        # Méthode simple : points extrêmes en x
        x_coords = crack_coords[:, 0]
        left_tip = crack_coords[np.argmin(x_coords)]
        right_tip = crack_coords[np.argmax(x_coords)]
        
        self.crack_tips.append({
            'left': left_tip,
            'right': right_tip,
            'length': np.linalg.norm(right_tip - left_tip)
        })
    
    def get_damage_rate(self) -> np.ndarray:
        """Calcule le taux d'évolution de l'endommagement"""
        if len(self.history) < 2:
            return np.array([])
        
        times = np.array([s['time'] for s in self.history])
        max_damages = np.array([s['max_damage'] for s in self.history])
        
        dt = np.diff(times)
        dd = np.diff(max_damages)
        
        # Éviter la division par zéro
        dt = np.where(dt > 0, dt, 1e-10)
        
        return dd / dt
    
    def get_crack_velocity(self) -> np.ndarray:
        """Calcule la vitesse de propagation de la fissure"""
        if len(self.history) < 2:
            return np.array([])
        
        times = np.array([s['time'] for s in self.history])
        lengths = np.array([s['crack_length'] for s in self.history])
        
        dt = np.diff(times)
        dl = np.diff(lengths)
        
        # Éviter la division par zéro
        dt = np.where(dt > 0, dt, 1e-10)
        
        return dl / dt


class PhaseFieldPostProcessor:
    """Post-traitement des résultats du champ de phase"""
    
    def __init__(self, mesh_manager, phase_field_solver):
        self.mesh = mesh_manager
        self.pf_solver = phase_field_solver
    
    def compute_damage_contours(self, d: np.ndarray, levels: list = None) -> Dict:
        """
        Calcule les contours d'endommagement
        
        Parameters:
            d: Champ d'endommagement
            levels: Niveaux de contour (par défaut [0.1, 0.5, 0.9])
            
        Returns:
            Dictionnaire avec les contours
        """
        if levels is None:
            levels = [0.1, 0.5, 0.9]
        
        from scipy.interpolate import griddata
        
        # Créer une grille régulière
        nx, ny = 200, 100
        xi = np.linspace(0, self.mesh.params.length, nx)
        yi = np.linspace(0, self.mesh.params.total_height, ny)
        Xi, Yi = np.meshgrid(xi, yi)
        
        # Interpoler l'endommagement sur la grille
        points = self.mesh.nodes
        values = d
        Zi = griddata(points, values, (Xi, Yi), method='linear', fill_value=0)
        
        # Calculer les contours
        import matplotlib.pyplot as plt
        contours = plt.contour(Xi, Yi, Zi, levels=levels)
        
        # Extraire les coordonnées des contours
        contour_data = {}
        for i, level in enumerate(levels):
            paths = contours.collections[i].get_paths()
            contour_data[f'level_{level}'] = [path.vertices for path in paths]
        
        plt.close()  # Fermer la figure temporaire
        
        return contour_data
    
    def compute_damaged_area(self, d: np.ndarray, threshold: float = 0.5) -> float:
        """
        Calcule l'aire endommagée
        
        Parameters:
            d: Champ d'endommagement
            threshold: Seuil d'endommagement
            
        Returns:
            Aire totale endommagée
        """
        damaged_area = 0.0
        
        for e in range(self.mesh.num_elements):
            element_nodes = self.mesh.elements[e]
            elem_damage = np.mean(d[element_nodes])
            
            if elem_damage > threshold:
                elem_area = self.mesh.get_element_size(e)
                damaged_area += elem_area * elem_damage
        
        return damaged_area
    
    def compute_energy_release_rate(self, d: np.ndarray, dd_dt: np.ndarray) -> float:
        """
        Calcule le taux de restitution d'énergie
        
        Parameters:
            d: Champ d'endommagement
            dd_dt: Taux d'évolution de l'endommagement
            
        Returns:
            Taux de restitution d'énergie global
        """
        G_total = 0.0
        
        for e in range(self.mesh.num_elements):
            element_nodes = self.mesh.elements[e]
            mat_props = self.mesh.materials.get_properties(self.mesh.material_id[e])
            Gc = mat_props.Gc
            
            # Taux moyen sur l'élément
            elem_dd_dt = np.mean(dd_dt[element_nodes])
            
            if elem_dd_dt > 0:
                elem_area = self.mesh.get_element_size(e)
                G_total += Gc * elem_dd_dt * elem_area
        
        return G_total