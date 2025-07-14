"""
Module de calculs énergétiques pour le modèle PF-CZM
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class EnergyComponents:
    """Composantes énergétiques du système"""
    strain: float = 0.0
    fracture: float = 0.0
    interface: float = 0.0
    kinetic: float = 0.0
    external_work: float = 0.0
    
    @property
    def total(self) -> float:
        """Énergie totale"""
        return self.strain + self.fracture + self.interface + self.kinetic
    
    @property
    def potential(self) -> float:
        """Énergie potentielle totale"""
        return self.strain + self.fracture + self.interface
    
    def to_dict(self) -> Dict[str, float]:
        """Convertit en dictionnaire"""
        return {
            'strain': self.strain,
            'fracture': self.fracture,
            'interface': self.interface,
            'kinetic': self.kinetic,
            'total': self.total,
            'potential': self.potential,
            'external_work': self.external_work
        }


class EnergyCalculator:
    """Calculateur d'énergies pour le système PF-CZM"""
    
    def __init__(self, mesh_manager, material_manager, cohesive_manager=None):
        self.mesh = mesh_manager
        self.materials = material_manager
        self.cohesive = cohesive_manager
        
        # Cache pour les calculs intermédiaires
        self._element_volumes = {}
        self._gauss_weights = None
        self._initialize_cache()
    
    def _initialize_cache(self):
        """Initialise le cache des volumes d'éléments"""
        # Points de Gauss pour l'intégration 2x2
        self.gauss_points = [
            (-1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)),
            (-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0))
        ]
        self._gauss_weights = np.ones(4, dtype=np.float64)
        
        # Pré-calculer les volumes des éléments
        for e in range(self.mesh.num_elements):
            self._element_volumes[e] = self._calculate_element_volume(e)
    
    def calculate_all_energies(self, u: np.ndarray, v: np.ndarray, 
                             d: np.ndarray, H_gauss: np.ndarray = None,
                             use_decomposition: bool = False) -> EnergyComponents:
        """
        Calcule toutes les composantes énergétiques
        
        Parameters:
            u: Vecteur de déplacements
            v: Vecteur de vitesses
            d: Vecteur d'endommagement nodal
            H_gauss: Champ d'histoire aux points de Gauss (optionnel)
            use_decomposition: Utiliser la décomposition spectrale
            
        Returns:
            EnergyComponents avec toutes les énergies
        """
        energies = EnergyComponents()
        
        # Énergie de déformation élastique
        energies.strain = self.calculate_strain_energy(u, d, use_decomposition)
        
        # Énergie de rupture
        energies.fracture = self.calculate_fracture_energy(d)
        
        # Énergie d'interface
        if self.cohesive is not None:
            energies.interface = self.cohesive.calculate_interface_energy(u)
        
        # Énergie cinétique
        energies.kinetic = self.calculate_kinetic_energy(v)
        
        return energies
    
    def calculate_strain_energy(self, u: np.ndarray, d: np.ndarray,
                              use_decomposition: bool = False) -> float:
        """
        Calcule l'énergie de déformation élastique totale
        
        Parameters:
            u: Vecteur de déplacements
            d: Vecteur d'endommagement
            use_decomposition: Si True, utilise la décomposition spectrale
            
        Returns:
            Énergie de déformation totale
        """
        strain_energy = 0.0
        
        # Boucle sur les éléments
        for e in range(self.mesh.num_elements):
            # Énergie de l'élément
            elem_energy = self._calculate_element_strain_energy(
                e, u, d, use_decomposition
            )
            
            # Ajouter à l'énergie totale
            strain_energy += elem_energy
        
        return strain_energy
    
    def calculate_fracture_energy(self, d: np.ndarray) -> float:
        """
        Calcule l'énergie de rupture totale
        
        Parameters:
            d: Vecteur d'endommagement nodal
            
        Returns:
            Énergie de rupture totale
        """
        fracture_energy = 0.0
        
        # Boucle sur les éléments
        for e in range(self.mesh.num_elements):
            # Propriétés du matériau
            mat_props = self.materials.get_properties(self.mesh.material_id[e])
            Gc = mat_props.Gc
            l0 = self.materials.l0 if hasattr(self.materials, 'l0') else 1.0
            
            # Coordonnées nodales
            element_nodes = self.mesh.elements[e]
            x_coords = self.mesh.nodes[element_nodes, 0]
            y_coords = self.mesh.nodes[element_nodes, 1]
            
            # Intégration sur les points de Gauss
            for gp_idx, (xi, eta) in enumerate(self.gauss_points):
                # Fonctions de forme et dérivées
                N, dN_dx, dN_dy, detJ = self._shape_functions_and_derivatives(
                    xi, eta, x_coords, y_coords
                )
                
                # Endommagement et gradient au point de Gauss
                d_gauss = np.dot(N, d[element_nodes])
                grad_d = np.array([
                    np.dot(dN_dx, d[element_nodes]),
                    np.dot(dN_dy, d[element_nodes])
                ])
                
                # Densité d'énergie de rupture
                fracture_density = 0.5 * Gc * (
                    d_gauss**2 / l0 + l0 * np.dot(grad_d, grad_d)
                )
                
                # Contribution à l'énergie totale
                fracture_energy += fracture_density * detJ * self._gauss_weights[gp_idx]
        
        return fracture_energy
    
    def calculate_kinetic_energy(self, v: np.ndarray) -> float:
        """
        Calcule l'énergie cinétique totale
        
        Parameters:
            v: Vecteur de vitesses
            
        Returns:
            Énergie cinétique totale
        """
        kinetic_energy = 0.0
        
        # Boucle sur les éléments
        for e in range(self.mesh.num_elements):
            # Propriétés du matériau
            mat_props = self.materials.get_properties(self.mesh.material_id[e])
            rho = mat_props.rho
            
            # Vitesses nodales
            element_nodes = self.mesh.elements[e]
            v_elem = np.zeros(8)
            for i in range(4):
                node = element_nodes[i]
                v_elem[2*i] = v[self.mesh.dof_map_u[node, 0]]
                v_elem[2*i+1] = v[self.mesh.dof_map_u[node, 1]]
            
            # Coordonnées nodales
            x_coords = self.mesh.nodes[element_nodes, 0]
            y_coords = self.mesh.nodes[element_nodes, 1]
            
            # Intégration sur les points de Gauss
            for gp_idx, (xi, eta) in enumerate(self.gauss_points):
                # Fonctions de forme
                N = self._shape_functions(xi, eta)
                
                # Jacobien
                detJ = self._calculate_jacobian(xi, eta, x_coords, y_coords)
                
                # Matrice N pour les vitesses
                N_matrix = np.zeros((2, 8))
                for i in range(4):
                    N_matrix[0, 2*i] = N[i]
                    N_matrix[1, 2*i+1] = N[i]
                
                # Vitesse au point de Gauss
                v_gauss = N_matrix @ v_elem
                
                # Densité d'énergie cinétique
                kinetic_density = 0.5 * rho * np.dot(v_gauss, v_gauss)
                
                # Contribution à l'énergie totale
                kinetic_energy += kinetic_density * detJ * self._gauss_weights[gp_idx]
        
        return kinetic_energy
    
    def calculate_external_work(self, u: np.ndarray, f_ext: np.ndarray) -> float:
        """
        Calcule le travail des forces externes
        
        Parameters:
            u: Vecteur de déplacements
            f_ext: Vecteur de forces externes
            
        Returns:
            Travail des forces externes
        """
        return np.dot(f_ext, u)
    
    def calculate_strain_energy_density_at_gauss_points(self, elem_id: int, 
                                                      u: np.ndarray,
                                                      use_decomposition: bool = False) -> np.ndarray:
        """
        Calcule la densité d'énergie de déformation aux points de Gauss d'un élément

        Returns:
            Array de 4 valeurs (une par point de Gauss)
        """
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

        # Densités aux points de Gauss
        psi_gauss = np.zeros(4)

        # Boucle sur les points de Gauss
        for gp_idx, (xi, eta) in enumerate(self.gauss_points):
            # Matrice B
            B = self._compute_B_matrix(xi, eta, x_coords, y_coords)

            # Déformation
            strain = B @ u_elem

            # Vérifier si la déformation est très petite
            strain_norm = np.linalg.norm(strain)

            if strain_norm < 1e-12:
                # Déformation négligeable : énergie nulle
                psi_gauss[gp_idx] = 0.0
                continue

            # Calculer l'énergie selon le type de matériau et les options
            if self.mesh.material_id[elem_id] == 1:  # Glace
                if use_decomposition and hasattr(self.materials, 'spectral_decomp'):
                    # Décomposition spectrale pour la glace
                    try:
                        strain_pos, strain_neg, P_pos, P_neg = self.materials.spectral_decomp.decompose(
                            strain, mat_props.E, mat_props.nu
                        )

                        # Énergie positive uniquement : ψ⁺ = ½ ε⁺:C:ε⁺
                        stress_pos = D @ strain_pos
                        psi_plus = 0.5 * np.dot(strain_pos, stress_pos)
                        psi_gauss[gp_idx] = max(0.0, psi_plus)

                    except Exception as e:
                        print(f"Erreur dans la décomposition spectrale: {e}")
                        # Fallback : énergie totale
                        stress = D @ strain
                        psi_gauss[gp_idx] = 0.5 * np.dot(strain, stress)
                else:
                    # Sans décomposition : énergie totale pour la glace
                    stress = D @ strain
                    psi_gauss[gp_idx] = max(0.0, 0.5 * np.dot(strain, stress))
            else:
                # Substrat : énergie totale (pas d'endommagement)
                stress = D @ strain
                psi_gauss[gp_idx] = max(0.0, 0.5 * np.dot(strain, stress))

        return psi_gauss
    
    def _calculate_element_strain_energy(self, elem_id: int, u: np.ndarray,
                                       d: np.ndarray, use_decomposition: bool) -> float:
        """Calcule l'énergie de déformation d'un élément"""
        # Densité d'énergie aux points de Gauss
        psi_gauss = self.calculate_strain_energy_density_at_gauss_points(
            elem_id, u, use_decomposition
        )
        
        # Volume de l'élément
        elem_volume = self._element_volumes[elem_id]
        
        # Pour le substrat : pas de dégradation
        if self.mesh.material_id[elem_id] == 0:  # Substrat
            elem_energy = np.sum(psi_gauss) * elem_volume / 4.0
            return elem_energy
        
        # Pour la glace : appliquer la dégradation
        element_nodes = self.mesh.elements[elem_id]
        
        # Calculer l'énergie dégradée aux points de Gauss
        elem_energy = 0.0
        
        for gp_idx, (xi, eta) in enumerate(self.gauss_points):
            # Interpoler l'endommagement au point de Gauss
            N = self._shape_functions(xi, eta)
            damage_gauss = np.dot(N, d[element_nodes])
            g_d = self.materials.degradation_function(damage_gauss)
            
            # Énergie dégradée au point de Gauss
            # Pour la décomposition spectrale : seulement ψ⁺ est dégradée
            # Sans décomposition : toute l'énergie est dégradée
            elem_energy += g_d * psi_gauss[gp_idx] * elem_volume / 4.0
        
        return elem_energy
    
    def _calculate_element_volume(self, elem_id: int) -> float:
        """Calcule le volume (aire en 2D) d'un élément"""
        if self.mesh.material_id[elem_id] == 0:  # Substrat
            return self.mesh.hx * self.mesh.hy_sub
        else:  # Glace
            return self.mesh.hx * self.mesh.hy_ice
    
    def _shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Calcule les fonctions de forme"""
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        return np.array([N1, N2, N3, N4])
    
    def _shape_functions_and_derivatives(self, xi: float, eta: float,
                                       x_coords: np.ndarray, 
                                       y_coords: np.ndarray) -> Tuple:
        """Calcule les fonctions de forme et leurs dérivées"""
        # Fonctions de forme
        N = self._shape_functions(xi, eta)
        
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
    
    def _calculate_jacobian(self, xi: float, eta: float, 
                          x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Calcule le déterminant du jacobien"""
        # Dérivées des fonctions de forme
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
        
        return np.linalg.det(J)
    
    def _compute_B_matrix(self, xi: float, eta: float,
                         x_coords: np.ndarray, y_coords: np.ndarray) -> np.ndarray:
        """Calcule la matrice B pour le calcul des déformations"""
        # Dérivées des fonctions de forme
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
        
        invJ = np.linalg.inv(J)
        
        # Dérivées par rapport à x et y
        dN_dx = invJ[0, 0] * dN_dxi + invJ[0, 1] * dN_deta
        dN_dy = invJ[1, 0] * dN_dxi + invJ[1, 1] * dN_deta
        
        # Matrice B
        B = np.zeros((3, 8))
        for i in range(4):
            B[0, 2*i] = dN_dx[i]      # du/dx
            B[1, 2*i+1] = dN_dy[i]    # dv/dy
            B[2, 2*i] = dN_dy[i]      # du/dy
            B[2, 2*i+1] = dN_dx[i]    # dv/dx
        
        return B


class EnergyMonitor:
    """Moniteur d'énergie pour vérifier la conservation et la dissipation"""
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
        self.history = []
        self.initial_total = None
    
    def add_record(self, time: float, energies: EnergyComponents, 
                  external_work: float = 0.0):
        """Ajoute un enregistrement d'énergie"""
        record = {
            'time': time,
            'energies': energies.to_dict(),
            'external_work': external_work,
            'dissipated': self._calculate_dissipation(energies, external_work)
        }
        
        self.history.append(record)
        
        # Stocker l'énergie initiale
        if self.initial_total is None:
            self.initial_total = energies.total
    
    def check_energy_balance(self) -> Tuple[bool, Dict]:
        """
        Vérifie le bilan énergétique
        
        Returns:
            balanced: True si le bilan est respecté
            info: Informations sur le bilan
        """
        if len(self.history) < 2:
            return True, {'message': 'Pas assez de données'}
        
        current = self.history[-1]
        
        # Bilan: E_total = E_initial + W_ext - D_dissipated
        expected_total = (self.initial_total + 
                         current['external_work'] - 
                         current['dissipated'])
        
        actual_total = current['energies']['total']
        
        error = abs(actual_total - expected_total)
        relative_error = error / max(abs(expected_total), 1e-10)
        
        balanced = relative_error < self.tolerance
        
        info = {
            'balanced': balanced,
            'expected_total': expected_total,
            'actual_total': actual_total,
            'absolute_error': error,
            'relative_error': relative_error,
            'dissipated_energy': current['dissipated']
        }
        
        return balanced, info
    
    def _calculate_dissipation(self, energies: EnergyComponents, 
                             external_work: float) -> float:
        """Calcule l'énergie dissipée"""
        if self.initial_total is None:
            return 0.0
        
        # Dissipation = W_ext + E_initial - E_current
        return external_work + self.initial_total - energies.total
    
    def plot_energy_history(self, ax=None):
        """Trace l'historique des énergies"""
        if ax is None:
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
        
        times = [r['time'] for r in self.history]
        
        # Tracer chaque composante
        for key in ['strain', 'fracture', 'interface', 'kinetic', 'total']:
            values = [r['energies'][key] for r in self.history]
            ax.plot(times, values, label=key.capitalize())
        
        # Tracer l'énergie dissipée
        dissipated = [r['dissipated'] for r in self.history]
        ax.plot(times, dissipated, '--', label='Dissipée')
        
        ax.set_xlabel('Temps')
        ax.set_ylabel('Énergie')
        ax.set_title('Évolution des énergies')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        return ax
    
    def get_dissipation_rate(self) -> np.ndarray:
        """Calcule le taux de dissipation d'énergie"""
        if len(self.history) < 2:
            return np.array([])
        
        times = np.array([r['time'] for r in self.history])
        dissipated = np.array([r['dissipated'] for r in self.history])
        
        # Dérivée numérique
        dt = np.diff(times)
        d_dissipated = np.diff(dissipated)
        
        # Éviter la division par zéro
        dt = np.where(dt > 0, dt, 1e-10)
        
        return d_dissipated / dt