"""
Module de gestion des éléments cohésifs et de l'endommagement d'interface
Version corrigée avec ordre des nœuds cohérent
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from materials import MaterialManager, CohesiveProperties


@dataclass
class CohesiveTraction:
    """Tractions cohésives à un point d'intégration"""
    normal: float
    tangential: float
    damage: float
    
    @property
    def magnitude(self) -> float:
        """Magnitude totale de la traction"""
        return np.sqrt(self.normal**2 + self.tangential**2)


class CohesiveZoneManager:
    """Gestionnaire des éléments cohésifs et de l'endommagement d'interface"""
    
    def __init__(self, material_manager: MaterialManager, mesh_manager):
        self.materials = material_manager
        self.mesh = mesh_manager
        self.cohesive_props = material_manager.cohesive
        
        # Cache pour les matrices de rigidité cohésives
        self._stiffness_cache: Dict[int, np.ndarray] = {}
        
        # Historique de l'endommagement pour l'irréversibilité
        self._damage_history: Dict[int, np.ndarray] = {}
        
        # Historique des vitesses pour la régularisation visqueuse
        self._velocity_history: Dict[int, Dict[str, np.ndarray]] = {}
        
        # Constantes numériques
        self.ZERO_TOL = 1.0e-12
        self.FORCE_TOL = 1.0e-12
        
        # Flag de debug
        self.debug = True
        
    def get_cohesive_stiffness(self, elem_idx: int, u: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de rigidité pour un élément cohésif
        
        ORDRE DES NŒUDS: [sub_gauche, sub_droite, ice_gauche, ice_droite]
        ORDRE DES DOFs: [u_sub1, v_sub1, u_sub2, v_sub2, u_ice1, v_ice1, u_ice2, v_ice2]
        """
        cohesive_elem = self.mesh.cohesive_elements[elem_idx]
        
        # Initialiser la matrice de rigidité
        K_coh = np.zeros((8, 8), dtype=np.float64)
        
        # Récupérer les nœuds et la longueur
        nodes = cohesive_elem.nodes  # [sub1, sub2, ice1, ice2]
        element_length = cohesive_elem.length
        
        # Points d'intégration
        gauss_points = cohesive_elem.gauss_points
        gauss_weights = cohesive_elem.gauss_weights
        
        # Boucle sur les points de Gauss
        for i, (xi, weight) in enumerate(zip(gauss_points, gauss_weights)):
            # Calculer les déplacements et tractions au point de Gauss
            delta_n, delta_t, T = self._compute_local_kinematics(
                nodes, xi, u
            )
            
            # Calculer la matrice de rigidité tangente locale
            D_local = self._compute_tangent_stiffness(
                delta_n, delta_t, cohesive_elem.damage[i]
            )
            
            # Assembler dans la matrice élémentaire
            K_coh += self._assemble_local_stiffness(
                D_local, T, xi, weight, element_length
            )
        
        return K_coh
    
    def update_damage(self, u: np.ndarray, dt: float = None) -> None:
        """
        Met à jour l'endommagement dans tous les éléments cohésifs
        """
        if self.debug:
            print("\n=== MISE À JOUR ENDOMMAGEMENT INTERFACE ===")
        
        for elem_idx, cohesive_elem in enumerate(self.mesh.cohesive_elements):
            # Sauvegarder l'endommagement précédent
            cohesive_elem.damage_prev = cohesive_elem.damage.copy()
            
            # Initialiser l'historique des vitesses si nécessaire
            if elem_idx not in self._velocity_history:
                self._velocity_history[elem_idx] = {
                    'delta_n': np.zeros(len(cohesive_elem.gauss_points)),
                    'delta_t': np.zeros(len(cohesive_elem.gauss_points))
                }
            
            # Points d'intégration
            gauss_points = cohesive_elem.gauss_points
            
            # Calculer le nouvel endommagement à chaque point de Gauss
            for i, xi in enumerate(gauss_points):
                # Cinématique locale
                delta_n, delta_t, _ = self._compute_local_kinematics(
                    cohesive_elem.nodes, xi, u
                )
                
                # Debug pour le premier élément
                #if self.debug and elem_idx == 0 and i == 0:
                #    print(f"  Élément 0, Point Gauss 0:")
                #    print(f"    delta_n = {delta_n:.6e} mm")
                #    print(f"    delta_t = {delta_t:.6e} mm")
                #    delta_eq = np.sqrt(max(0, delta_n)**2 + delta_t**2)
                #    print(f"    delta_eq = {delta_eq:.6e} mm")
                
                # Calculer les vitesses si dt fourni
                delta_n_rate = 0.0
                delta_t_rate = 0.0
                if dt is not None and dt > 0:
                    delta_n_prev = self._velocity_history[elem_idx]['delta_n'][i]
                    delta_t_prev = self._velocity_history[elem_idx]['delta_t'][i]
                    delta_n_rate = (delta_n - delta_n_prev) / dt
                    delta_t_rate = (delta_t - delta_t_prev) / dt
                    
                    # Mettre à jour l'historique
                    self._velocity_history[elem_idx]['delta_n'][i] = delta_n
                    self._velocity_history[elem_idx]['delta_t'][i] = delta_t
                
                # Calculer l'endommagement
                new_damage = self._compute_damage(delta_n, delta_t)
                
                #if self.debug and elem_idx == 0 and i == 0:
                #    print(f"    damage calculé = {new_damage:.6f}")
                #    print(f"    damage précédent = {cohesive_elem.damage_prev[i]:.6f}")
                
                # Appliquer l'irréversibilité
                cohesive_elem.damage[i] = max(
                    cohesive_elem.damage_prev[i], 
                    new_damage
                )
    
    def calculate_interface_forces(self, u: np.ndarray) -> np.ndarray:
        """
        Calcule les forces cohésives à l'interface
        """
        f_coh = np.zeros(self.mesh.num_dofs_u, dtype=np.float64)
        
        for cohesive_elem in self.mesh.cohesive_elements:
            # Calculer les forces élémentaires
            f_elem = self._compute_element_forces(cohesive_elem, u)
            
            # Assembler dans le vecteur global
            self._assemble_element_forces(f_coh, f_elem, cohesive_elem.nodes)
        
        # Nettoyer les forces très petites
        f_coh = np.where(np.abs(f_coh) < self.FORCE_TOL, 0.0, f_coh)
        
        return f_coh
    
    def calculate_interface_energy(self, u: np.ndarray) -> float:
        """
        Calcule l'énergie de rupture de l'interface
        """
        interface_energy = 0.0
        
        for cohesive_elem in self.mesh.cohesive_elements:
            # Points d'intégration
            gauss_points = cohesive_elem.gauss_points
            gauss_weights = cohesive_elem.gauss_weights
            element_length = cohesive_elem.length
            
            for i, (xi, weight) in enumerate(zip(gauss_points, gauss_weights)):
                # Cinématique locale
                delta_n, delta_t, _ = self._compute_local_kinematics(
                    cohesive_elem.nodes, xi, u
                )
                
                # Propriétés effectives
                eff_props = self.materials.calculate_effective_properties(
                    delta_n, delta_t
                )
                
                # Contribution énergétique
                damage = cohesive_elem.damage[i]
                energy_density = eff_props['Gc_eff'] * damage**2
                
                # Jacobien pour l'élément ligne
                detJ = element_length / 2.0
                
                # Ajouter à l'énergie totale
                interface_energy += energy_density * weight * detJ
        
        return interface_energy
    
    def get_max_interface_damage(self) -> float:
        """Retourne l'endommagement maximal dans l'interface"""
        max_damage = 0.0
        
        for cohesive_elem in self.mesh.cohesive_elements:
            elem_max = np.max(cohesive_elem.damage)
            max_damage = max(max_damage, elem_max)
        
        return max_damage
    
    def _compute_local_kinematics(self, nodes: List[int], xi: float, 
                                 u: np.ndarray) -> Tuple[float, float, np.ndarray]:
        """
        Calcule les sauts de déplacement et la matrice de transformation
        
        ORDRE DES NŒUDS: [sub_gauche, sub_droite, ice_gauche, ice_droite]
        """
        # Récupérer les nœuds
        sub_node1, sub_node2, ice_node1, ice_node2 = nodes
        
        # Fonctions de forme pour élément 1D
        # xi = -1 correspond au nœud gauche (1)
        # xi = +1 correspond au nœud droite (2)
        N1 = (1.0 - xi) / 2.0  # Fonction de forme pour nœud gauche
        N2 = (1.0 + xi) / 2.0  # Fonction de forme pour nœud droite
        
        # Récupérer les déplacements nodaux
        # Ordre: [sub1, sub2, ice1, ice2]
        u_nodes = self._extract_nodal_displacements(
            [sub_node1, sub_node2, ice_node1, ice_node2], u
        )
        
        # Calculer les coordonnées courantes (référence + déplacement)
        coords_current = self._compute_current_coordinates(
            [sub_node1, sub_node2, ice_node1, ice_node2], 
            u_nodes
        )
        
        # Interpoler au point de Gauss
        # Substrat: interpolation entre sub1 (gauche) et sub2 (droite)
        x_sub = N1 * coords_current[0, 0] + N2 * coords_current[1, 0]
        y_sub = N1 * coords_current[0, 1] + N2 * coords_current[1, 1]
        
        # Glace: interpolation entre ice1 (gauche) et ice2 (droite)
        x_ice = N1 * coords_current[2, 0] + N2 * coords_current[3, 0]
        y_ice = N1 * coords_current[2, 1] + N2 * coords_current[3, 1]
        
        # Sauts de déplacement (glace - substrat)
        delta_x = x_ice - x_sub
        delta_y = y_ice - y_sub
        
        # Calculer le repère local (tangent à l'interface)
        # Utilise les nœuds du substrat pour définir la direction tangente
        T = self._compute_local_basis(coords_current[0], coords_current[1])
        
        # Transformer en coordonnées locales
        delta_local = T @ np.array([delta_x, delta_y])
        delta_n = delta_local[0]  # Normal (perpendiculaire à l'interface)
        delta_t = delta_local[1]  # Tangentiel (le long de l'interface)
        
        return delta_n, delta_t, T
    
    def _compute_tangent_stiffness(self, delta_n: float, delta_t: float, 
                                  damage: float) -> np.ndarray:
        """
        Calcule la matrice de rigidité tangente en coordonnées locales
        """
        damage_factor = 1.0 - damage
        
        # Rigidité normale
        k_nn = self._compute_normal_stiffness(delta_n, damage_factor)
        
        # Rigidité tangentielle
        k_tt = self._compute_tangential_stiffness(delta_t, damage_factor)
        
        # Matrice de rigidité locale
        D_local = np.array([
            [k_nn, 0.0],
            [0.0, k_tt]
        ], dtype=np.float64)
        
        return D_local
    
    def _compute_normal_stiffness(self, delta_n: float, damage_factor: float) -> float:
        """Calcule la rigidité normale"""
        if delta_n >= 0.0:  # Traction
            if delta_n <= self.cohesive_props.normal_delta0:
                # Phase élastique
                return damage_factor * self.cohesive_props.normal_stiffness
            elif delta_n <= self.cohesive_props.normal_deltac:
                # Phase d'adoucissement
                slope = -self.cohesive_props.normal_strength / \
                        (self.cohesive_props.normal_deltac - self.cohesive_props.normal_delta0)
                return damage_factor * slope
            else:
                # Rupture complète
                return self.ZERO_TOL
        else:  # Compression
            # Pénalité pour éviter l'interpénétration
            return self.cohesive_props.compression_factor * self.cohesive_props.normal_stiffness
    
    def _compute_tangential_stiffness(self, delta_t: float, damage_factor: float) -> float:
        """Calcule la rigidité tangentielle"""
        delta_t_abs = abs(delta_t)
        
        if delta_t_abs <= self.cohesive_props.shear_delta0:
            # Phase élastique
            return damage_factor * self.cohesive_props.shear_stiffness
        elif delta_t_abs <= self.cohesive_props.shear_deltac:
            # Phase d'adoucissement
            slope = -self.cohesive_props.shear_strength / \
                    (self.cohesive_props.shear_deltac - self.cohesive_props.shear_delta0)
            return damage_factor * abs(slope)
        else:
            # Rupture complète
            return self.ZERO_TOL
    
    def _compute_damage(self, delta_n: float, delta_t: float) -> float:
        """
        Calcule l'endommagement basé sur les sauts de déplacement
        """
        # Saut équivalent en mode mixte
        delta_n_pos = max(0.0, delta_n)  # Seulement la partie positive
        delta_eq = np.sqrt(delta_n_pos**2 + delta_t**2)
        
        # Propriétés effectives en mode mixte
        eff_props = self.materials.calculate_effective_properties(delta_n, delta_t)
        
        #if self.debug and delta_eq > 1e-10:
         #   print(f"      delta_eq = {delta_eq:.6e} mm")
         #   print(f"      delta0_eff = {eff_props['delta0_eff']:.6e} mm")
         #   print(f"      deltac_eff = {eff_props['deltac_eff']:.6e} mm")
        
        # Calcul de l'endommagement
        if delta_eq <= eff_props['delta0_eff']:
            return 0.0
        elif delta_eq <= eff_props['deltac_eff']:
            # Évolution linéaire de l'endommagement
            damage = (delta_eq - eff_props['delta0_eff']) / \
                    (eff_props['deltac_eff'] - eff_props['delta0_eff'])
            
            # Pas de seuil minimal pour permettre l'initiation
            return damage
        else:
            return 1.0
    
    def _compute_tractions(self, delta_n: float, delta_t: float, 
                          damage: float, delta_n_rate: float = 0.0,
                          delta_t_rate: float = 0.0) -> CohesiveTraction:
        """Calcule les tractions cohésives avec régularisation visqueuse optionnelle"""
        damage_factor = 1.0 - damage
        
        # Paramètre de viscosité
        eta = self.cohesive_props.viscosity if hasattr(self.cohesive_props, 'viscosity') else 0.0
        
        # Traction normale
        if delta_n >= 0.0:  # Traction
            if delta_n <= self.cohesive_props.normal_delta0:
                traction_n = damage_factor * self.cohesive_props.normal_stiffness * delta_n
            elif delta_n <= self.cohesive_props.normal_deltac:
                traction_n = damage_factor * self.cohesive_props.normal_strength * \
                            (self.cohesive_props.normal_deltac - delta_n) / \
                            (self.cohesive_props.normal_deltac - self.cohesive_props.normal_delta0)
            else:
                traction_n = 0.0
            
            # Ajouter terme visqueux en traction seulement
            traction_n += eta * delta_n_rate
        else:  # Compression
            traction_n = self.cohesive_props.compression_factor * \
                        self.cohesive_props.normal_stiffness * delta_n
        
        # Traction tangentielle
        delta_t_abs = abs(delta_t)
        if delta_t_abs <= self.cohesive_props.shear_delta0:
            traction_t_mag = damage_factor * self.cohesive_props.shear_stiffness * delta_t_abs
        elif delta_t_abs <= self.cohesive_props.shear_deltac:
            traction_t_mag = damage_factor * self.cohesive_props.shear_strength * \
                            (self.cohesive_props.shear_deltac - delta_t_abs) / \
                            (self.cohesive_props.shear_deltac - self.cohesive_props.shear_delta0)
        else:
            traction_t_mag = 0.0
        
        traction_t = np.sign(delta_t) * traction_t_mag
        
        # Ajouter terme visqueux tangentiel
        traction_t += eta * delta_t_rate
        
        return CohesiveTraction(
            normal=traction_n,
            tangential=traction_t,
            damage=damage
        )
    
    def _compute_local_basis(self, node1_coords: np.ndarray, 
                           node2_coords: np.ndarray) -> np.ndarray:
        """
        Calcule la matrice de transformation vers le repère local
        
        Returns:
            Matrice de transformation 2x2 [n; t]
        """
        # Vecteur tangent (de gauche vers droite)
        tangent = node2_coords - node1_coords
        tangent_length = np.linalg.norm(tangent)
        
        if tangent_length > self.ZERO_TOL:
            tangent = tangent / tangent_length
        else:
            tangent = np.array([1.0, 0.0])
        
        # Vecteur normal (rotation de 90° dans le sens anti-horaire)
        # Pour une interface horizontale, normale pointe vers le haut
        normal = np.array([-tangent[1], tangent[0]])
        
        # Matrice de transformation
        T = np.array([normal, tangent])
        
        return T
    
    def _extract_nodal_displacements(self, node_list: List[int], 
                                   u: np.ndarray) -> np.ndarray:
        """Extrait les déplacements nodaux"""
        u_nodes = np.zeros((len(node_list), 2), dtype=np.float64)
        
        for i, node in enumerate(node_list):
            u_nodes[i, 0] = u[self.mesh.dof_map_u[node, 0]]
            u_nodes[i, 1] = u[self.mesh.dof_map_u[node, 1]]
        
        return u_nodes
    
    def _compute_current_coordinates(self, node_list: List[int], 
                                   u_nodes: np.ndarray) -> np.ndarray:
        """Calcule les coordonnées courantes (référence + déplacement)"""
        coords = np.zeros((len(node_list), 2), dtype=np.float64)
        
        for i, node in enumerate(node_list):
            coords[i, 0] = self.mesh.nodes[node, 0] + u_nodes[i, 0]
            coords[i, 1] = self.mesh.nodes[node, 1] + u_nodes[i, 1]
        
        return coords
    
    def _assemble_local_stiffness(self, D_local: np.ndarray, T: np.ndarray,
                                 xi: float, weight: float, 
                                 element_length: float) -> np.ndarray:
        """
        Assemble la contribution locale dans la matrice élémentaire
        
        ORDRE DES NŒUDS: [sub_gauche, sub_droite, ice_gauche, ice_droite]
        ORDRE DES DOFs: [u_sub1, v_sub1, u_sub2, v_sub2, u_ice1, v_ice1, u_ice2, v_ice2]
        """
        # Matrice de rigidité globale
        D_global = T.T @ D_local @ T
        
        # Fonctions de forme
        N1 = (1.0 - xi) / 2.0  # Nœud gauche
        N2 = (1.0 + xi) / 2.0  # Nœud droite
        
        # Matrice B pour le calcul des sauts
        # Saut = déplacement_glace - déplacement_substrat
        B = np.zeros((2, 8), dtype=np.float64)
        
        # Contributions du substrat (négatif car on fait glace - substrat)
        # Nœud 1 du substrat (gauche)
        B[0, 0] = -N1  # u_sub1
        B[1, 1] = -N1  # v_sub1
        
        # Nœud 2 du substrat (droite)
        B[0, 2] = -N2  # u_sub2
        B[1, 3] = -N2  # v_sub2
        
        # Contributions de la glace (positif)
        # Nœud 1 de la glace (gauche)
        B[0, 4] = N1   # u_ice1
        B[1, 5] = N1   # v_ice1
        
        # Nœud 2 de la glace (droite)
        B[0, 6] = N2   # u_ice2
        B[1, 7] = N2   # v_ice2
        
        # Jacobien pour élément ligne
        detJ = element_length / 2.0
        
        # Contribution à la matrice de rigidité
        K_local = B.T @ D_global @ B * weight * detJ
        
        return K_local
    
    def _compute_element_forces(self, cohesive_elem, u: np.ndarray) -> np.ndarray:
        """Calcule les forces pour un élément cohésif"""
        f_elem = np.zeros(8, dtype=np.float64)
        
        # Points d'intégration
        gauss_points = cohesive_elem.gauss_points
        gauss_weights = cohesive_elem.gauss_weights
        element_length = cohesive_elem.length
        
        for i, (xi, weight) in enumerate(zip(gauss_points, gauss_weights)):
            # Cinématique locale
            delta_n, delta_t, T = self._compute_local_kinematics(
                cohesive_elem.nodes, xi, u
            )
            
            # Calculer les tractions
            traction = self._compute_tractions(
                delta_n, delta_t, cohesive_elem.damage[i]
            )
            
            # Transformer en coordonnées globales
            t_global = T.T @ np.array([traction.normal, traction.tangential])
            
            # Assembler les forces
            f_elem += self._assemble_local_forces(
                t_global, xi, weight, element_length
            )
        
        return f_elem
    
    def _assemble_local_forces(self, t_global: np.ndarray, xi: float,
                             weight: float, element_length: float) -> np.ndarray:
        """
        Assemble les forces locales dans le vecteur élémentaire
        
        ORDRE DES NŒUDS: [sub_gauche, sub_droite, ice_gauche, ice_droite]
        """
        # Fonctions de forme
        N1 = (1.0 - xi) / 2.0  # Nœud gauche
        N2 = (1.0 + xi) / 2.0  # Nœud droite
        
        # Jacobien
        detJ = element_length / 2.0
        
        # Facteur de force
        force_factor = weight * detJ
        fx = t_global[0] * force_factor
        fy = t_global[1] * force_factor
        
        # Forces élémentaires
        # Les tractions s'opposent au saut de déplacement
        f_elem = np.zeros(8, dtype=np.float64)
        
        # Forces sur les nœuds du substrat (positives - les tractions tirent le substrat)
        f_elem[0] = fx * N1   # sub_node1 x
        f_elem[1] = fy * N1   # sub_node1 y
        f_elem[2] = fx * N2   # sub_node2 x
        f_elem[3] = fy * N2   # sub_node2 y
        
        # Forces sur les nœuds de glace (négatives - réaction opposée)
        f_elem[4] = -fx * N1  # ice_node1 x
        f_elem[5] = -fy * N1  # ice_node1 y
        f_elem[6] = -fx * N2  # ice_node2 x
        f_elem[7] = -fy * N2  # ice_node2 y
        
        return f_elem
    
    def _assemble_element_forces(self, f_global: np.ndarray, f_elem: np.ndarray,
                               nodes: List[int]) -> None:
        """
        Assemble les forces élémentaires dans le vecteur global
        
        ORDRE DES NŒUDS: [sub_gauche, sub_droite, ice_gauche, ice_droite]
        """
        sub_node1, sub_node2, ice_node1, ice_node2 = nodes
        
        # Assembler dans le vecteur global
        f_global[self.mesh.dof_map_u[sub_node1, 0]] += f_elem[0]
        f_global[self.mesh.dof_map_u[sub_node1, 1]] += f_elem[1]
        f_global[self.mesh.dof_map_u[sub_node2, 0]] += f_elem[2]
        f_global[self.mesh.dof_map_u[sub_node2, 1]] += f_elem[3]
        f_global[self.mesh.dof_map_u[ice_node1, 0]] += f_elem[4]
        f_global[self.mesh.dof_map_u[ice_node1, 1]] += f_elem[5]
        f_global[self.mesh.dof_map_u[ice_node2, 0]] += f_elem[6]
        f_global[self.mesh.dof_map_u[ice_node2, 1]] += f_elem[7]