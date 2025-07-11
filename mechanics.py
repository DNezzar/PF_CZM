"""
Module de mécanique pour l'assemblage des matrices et le calcul des forces
"""

import numpy as np
from scipy.sparse import lil_matrix, csr_matrix
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass


@dataclass
class LoadingParameters:
    """Paramètres de chargement"""
    omega: float = 830.1135          # Vitesse angulaire (rad/s)
    ramp_time: float = 1.0           # Temps de montée en charge
    load_type: str = 'centrifugal'  # Type de chargement
    
    def get_load_factor(self, time: float) -> float:
        """Calcule le facteur de charge à un instant donné"""
        if self.ramp_time <= 0:
            return 1.0
        return min(time / self.ramp_time, 1.0)


class ElementMatrices:
    """Calcul des matrices élémentaires"""
    
    def __init__(self, mesh_manager, material_manager):
        self.mesh = mesh_manager
        self.materials = material_manager
        
        # Points de Gauss pour intégration 2x2
        self.gauss_points = [
            (-1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), -1.0/np.sqrt(3.0)),
            (1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0)),
            (-1.0/np.sqrt(3.0), 1.0/np.sqrt(3.0))
        ]
        self.gauss_weights = np.ones(4)
        
        # Cache pour les calculs répétitifs
        self._B_cache = {}
        self._detJ_cache = {}
    
    def get_stiffness_matrix(self, elem_id: int, u: np.ndarray, d: np.ndarray,
                           P_pos_prev: List[np.ndarray] = None,
                           P_neg_prev: List[np.ndarray] = None,
                           use_decomposition: bool = False) -> np.ndarray:
        """
        Calcule la matrice de rigidité élémentaire
        
        Parameters:
            elem_id: Identifiant de l'élément
            u: Déplacements nodaux
            d: Endommagement nodal
            P_pos_prev: Projecteurs positifs du pas précédent
            P_neg_prev: Projecteurs négatifs du pas précédent
            use_decomposition: Utiliser la décomposition spectrale
            
        Returns:
            K_elem: Matrice de rigidité 8x8
        """
        # Récupérer les données de l'élément
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        # Déplacements élémentaires
        u_elem = self._extract_element_displacements(element_nodes, u)
        
        # Propriétés matériaux
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        D = self.materials.get_constitutive_matrix(self.mesh.material_id[elem_id])
        
        # Initialiser la matrice de rigidité
        K_elem = np.zeros((8, 8), dtype=np.float64)
        
        # Nouvelles listes de projecteurs pour le pas suivant
        P_pos_new = []
        P_neg_new = []
        
        # Intégration sur les points de Gauss
        for gp_idx, (xi, eta) in enumerate(self.gauss_points):
            # Interpoler l'endommagement
            N = self._shape_functions(xi, eta)
            damage_gauss = np.dot(N, d[element_nodes])
            g_d = self.materials.degradation_function(damage_gauss)
            
            # Matrice B et jacobien
            B, detJ = self._get_B_matrix_and_jacobian(elem_id, xi, eta, x_coords, y_coords)
            
            if use_decomposition and P_pos_prev is not None:
                # Décomposition spectrale avec approximation temporelle
                
                # Utiliser les projecteurs du pas précédent pour la rigidité
                P_pos = P_pos_prev[gp_idx]
                P_neg = P_neg_prev[gp_idx]
                
                # Matrice constitutive dégradée (Eq. 26)
                D_degraded = g_d * (P_pos.T @ D @ P_pos) + (P_neg.T @ D @ P_neg)
                
                # Calculer la déformation actuelle pour mettre à jour les projecteurs
                strain = B @ u_elem
                
                # Nouveaux projecteurs pour le pas suivant
                if hasattr(self.materials, 'spectral_decomp'):
                    _, _, P_pos_new_gp, P_neg_new_gp = self.materials.spectral_decomp.decompose(
                        strain, mat_props.E, mat_props.nu
                    )
                    P_pos_new.append(P_pos_new_gp)
                    P_neg_new.append(P_neg_new_gp)
                else:
                    # Sans décomposition disponible
                    P_pos_new.append(np.eye(3))
                    P_neg_new.append(np.eye(3))
                
            else:
                # Sans décomposition spectrale
                D_degraded = g_d * D
                P_pos_new.append(np.eye(3))
                P_neg_new.append(np.eye(3))
            
            # Contribution à la matrice de rigidité
            K_elem += B.T @ D_degraded @ B * detJ * self.gauss_weights[gp_idx]
        
        return K_elem, P_pos_new, P_neg_new
    
    def get_mass_matrix(self, elem_id: int) -> np.ndarray:
        """
        Calcule la matrice de masse élémentaire (cohérente)
        
        Returns:
            M_elem: Matrice de masse 8x8
        """
        # Récupérer les données de l'élément
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        # Propriétés matériaux
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        rho = mat_props.rho
        
        # Initialiser la matrice de masse
        M_elem = np.zeros((8, 8), dtype=np.float64)
        
        # Intégration sur les points de Gauss
        for xi, eta in self.gauss_points:
            # Fonctions de forme
            N = self._shape_functions(xi, eta)
            
            # Matrice N étendue
            N_matrix = self._build_N_matrix(N)
            
            # Jacobien
            detJ = self._calculate_jacobian(xi, eta, x_coords, y_coords)
            
            # Contribution à la matrice de masse
            M_elem += rho * N_matrix.T @ N_matrix * detJ
        
        return M_elem
    
    def get_centrifugal_force(self, elem_id: int, omega: float, 
                            load_factor: float = 1.0) -> np.ndarray:
        """
        Calcule le vecteur de force centrifuge élémentaire
        
        Returns:
            f_elem: Vecteur de force 8x1
        """
        # Récupérer les données de l'élément
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        # Propriétés matériaux
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        rho = mat_props.rho
        
        # Initialiser le vecteur de force
        f_elem = np.zeros(8, dtype=np.float64)
        
        # Intégration sur les points de Gauss
        for xi, eta in self.gauss_points:
            # Fonctions de forme
            N = self._shape_functions(xi, eta)
            
            # Position x au point de Gauss
            x_gauss = np.dot(N, x_coords)
            
            # Force centrifuge (direction radiale = x)
            f_centrifugal = load_factor * rho * omega**2 * x_gauss
            
            # Jacobien
            detJ = self._calculate_jacobian(xi, eta, x_coords, y_coords)
            
            # Contribution au vecteur de force (seulement composante x)
            for i in range(4):
                f_elem[2*i] += N[i] * f_centrifugal * detJ
        
        return f_elem
    
    def _shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Calcule les fonctions de forme"""
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        return np.array([N1, N2, N3, N4])
    
    def _get_B_matrix_and_jacobian(self, elem_id: int, xi: float, eta: float,
                                  x_coords: np.ndarray, y_coords: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calcule la matrice B et le jacobien (avec cache)"""
        cache_key = (elem_id, xi, eta)
        
        if cache_key in self._B_cache:
            return self._B_cache[cache_key], self._detJ_cache[cache_key]
        
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
        
        detJ = np.linalg.det(J)
        
        # Protection contre les jacobiens singuliers
        if abs(detJ) < 1e-12:
            detJ = 1e-12 if detJ >= 0 else -1e-12
        
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
        
        # Stocker dans le cache
        self._B_cache[cache_key] = B
        self._detJ_cache[cache_key] = detJ
        
        return B, detJ
    
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
        J11 = np.dot(dN_dxi, x_coords)
        J12 = np.dot(dN_dxi, y_coords)
        J21 = np.dot(dN_deta, x_coords)
        J22 = np.dot(dN_deta, y_coords)
        
        return J11 * J22 - J12 * J21
    
    def _build_N_matrix(self, N: np.ndarray) -> np.ndarray:
        """Construit la matrice N étendue pour les déplacements"""
        N_matrix = np.zeros((2, 8))
        for i in range(4):
            N_matrix[0, 2*i] = N[i]
            N_matrix[1, 2*i+1] = N[i]
        return N_matrix
    
    def _extract_element_displacements(self, element_nodes: np.ndarray, 
                                     u: np.ndarray) -> np.ndarray:
        """Extrait les déplacements élémentaires"""
        u_elem = np.zeros(8)
        for i in range(4):
            node = element_nodes[i]
            u_elem[2*i] = u[self.mesh.dof_map_u[node, 0]]
            u_elem[2*i+1] = u[self.mesh.dof_map_u[node, 1]]
        return u_elem


class SystemAssembler:
    """Assemblage des matrices et vecteurs globaux"""
    
    def __init__(self, mesh_manager, material_manager, cohesive_manager,
                 element_matrices: ElementMatrices):
        self.mesh = mesh_manager
        self.materials = material_manager
        self.cohesive = cohesive_manager
        self.elem_matrices = element_matrices
        
        # Stockage des projecteurs pour la décomposition spectrale
        self.P_pos_prev = {}
        self.P_neg_prev = {}
        self._initialize_projection_tensors()
        
        # Cache pour les forces internes
        self._internal_forces_cache = None
        self._cache_u = None
        self._cache_d = None
    
    def _initialize_projection_tensors(self):
        """Initialise les tenseurs de projection"""
        for e in range(self.mesh.num_elements):
            self.P_pos_prev[e] = [np.eye(3, dtype=np.float64) for _ in range(4)]
            self.P_neg_prev[e] = [np.eye(3, dtype=np.float64) for _ in range(4)]
    
    def assemble_system(self, u: np.ndarray, d: np.ndarray, time: float,
                       loading_params: LoadingParameters,
                       use_decomposition: bool = False) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
        """
        Assemble les matrices et vecteurs du système global
        
        Returns:
            K: Matrice de rigidité globale
            M: Matrice de masse globale
            f_ext: Vecteur de forces externes
        """
        # Initialiser les matrices creuses
        K = lil_matrix((self.mesh.num_dofs_u, self.mesh.num_dofs_u), dtype=np.float64)
        M = lil_matrix((self.mesh.num_dofs_u, self.mesh.num_dofs_u), dtype=np.float64)
        f_ext = np.zeros(self.mesh.num_dofs_u, dtype=np.float64)
        
        # Facteur de charge
        load_factor = loading_params.get_load_factor(time)
        
        # Nouveaux projecteurs pour le pas suivant
        P_pos_new = {}
        P_neg_new = {}
        
        # Assembler les éléments quadrilatéraux
        for e in range(self.mesh.num_elements):
            # DOFs globaux
            element_nodes = self.mesh.elements[e]
            dofs = self._get_element_dofs(element_nodes)
            
            # Matrices élémentaires
            K_elem, P_pos_elem, P_neg_elem = self.elem_matrices.get_stiffness_matrix(
                e, u, d, 
                self.P_pos_prev.get(e), 
                self.P_neg_prev.get(e),
                use_decomposition
            )
            M_elem = self.elem_matrices.get_mass_matrix(e)
            
            # Forces centrifuges
            if loading_params.load_type == 'centrifugal':
                f_elem = self.elem_matrices.get_centrifugal_force(
                    e, loading_params.omega, load_factor
                )
            else:
                f_elem = np.zeros(8)
            
            # Assembler dans les matrices globales
            for i in range(8):
                for j in range(8):
                    K[dofs[i], dofs[j]] += K_elem[i, j]
                    M[dofs[i], dofs[j]] += M_elem[i, j]
                f_ext[dofs[i]] += f_elem[i]
            
            # Stocker les nouveaux projecteurs
            P_pos_new[e] = P_pos_elem
            P_neg_new[e] = P_neg_elem
        
        # Assembler les éléments cohésifs
        if self.cohesive is not None:
            for i, cohesive_elem in enumerate(self.mesh.cohesive_elements):
                # Matrice de rigidité cohésive
                K_coh = self.cohesive.get_cohesive_stiffness(i, u)
                
                # DOFs cohésifs
                dofs_coh = self._get_cohesive_dofs(cohesive_elem.nodes)
                
                # Assembler
                for i in range(8):
                    for j in range(8):
                        K[dofs_coh[i], dofs_coh[j]] += K_coh[i, j]
        
        # Mettre à jour les projecteurs pour le pas suivant
        self.P_pos_prev = P_pos_new
        self.P_neg_prev = P_neg_new
        
        # Appliquer les conditions aux limites
        bc_dict = self.mesh.get_boundary_conditions()
        self._apply_boundary_conditions(K, M, f_ext, bc_dict)
        
        # Convertir en format CSR pour la résolution
        K = K.tocsr()
        M = M.tocsr()
        
        # Invalider le cache des forces internes
        self._internal_forces_cache = None
        
        return K, M, f_ext
    
    def _get_element_dofs(self, element_nodes: np.ndarray) -> np.ndarray:
        """Récupère les DOFs globaux d'un élément"""
        dofs = np.zeros(8, dtype=int)
        for i in range(4):
            node = element_nodes[i]
            dofs[2*i] = self.mesh.dof_map_u[node, 0]
            dofs[2*i+1] = self.mesh.dof_map_u[node, 1]
        return dofs
    
    def _get_cohesive_dofs(self, cohesive_nodes: List[int]) -> np.ndarray:
        """Récupère les DOFs globaux d'un élément cohésif"""
        sub_node1, sub_node2, ice_node2, ice_node1 = cohesive_nodes
        
        dofs = np.zeros(8, dtype=int)
        dofs[0] = self.mesh.dof_map_u[sub_node1, 0]
        dofs[1] = self.mesh.dof_map_u[sub_node1, 1]
        dofs[2] = self.mesh.dof_map_u[sub_node2, 0]
        dofs[3] = self.mesh.dof_map_u[sub_node2, 1]
        dofs[4] = self.mesh.dof_map_u[ice_node2, 0]
        dofs[5] = self.mesh.dof_map_u[ice_node2, 1]
        dofs[6] = self.mesh.dof_map_u[ice_node1, 0]
        dofs[7] = self.mesh.dof_map_u[ice_node1, 1]
        
        return dofs
    
    def _apply_boundary_conditions(self, K, M, f, bc_dict: Dict):
        """Applique les conditions aux limites"""
        # Nœuds complètement fixés
        for node in bc_dict.get('fully_fixed', []):
            for dof in range(2):
                dof_idx = self.mesh.dof_map_u[node, dof]
                K[dof_idx, :] = 0.0
                K[:, dof_idx] = 0.0
                K[dof_idx, dof_idx] = 1.0
                
                M[dof_idx, :] = 0.0
                M[:, dof_idx] = 0.0
                M[dof_idx, dof_idx] = 1.0
                
                f[dof_idx] = 0.0
        
        # Nœuds fixés en x seulement
        for node in bc_dict.get('fixed_x', []):
            dof_idx = self.mesh.dof_map_u[node, 0]
            K[dof_idx, :] = 0.0
            K[:, dof_idx] = 0.0
            K[dof_idx, dof_idx] = 1.0
            
            M[dof_idx, :] = 0.0
            M[:, dof_idx] = 0.0
            M[dof_idx, dof_idx] = 1.0
            
            f[dof_idx] = 0.0
        
        # Nœuds fixés en y seulement
        for node in bc_dict.get('fixed_y', []):
            dof_idx = self.mesh.dof_map_u[node, 1]
            K[dof_idx, :] = 0.0
            K[:, dof_idx] = 0.0
            K[dof_idx, dof_idx] = 1.0
            
            M[dof_idx, :] = 0.0
            M[:, dof_idx] = 0.0
            M[dof_idx, dof_idx] = 1.0
            
            f[dof_idx] = 0.0
    
    def get_internal_forces(self, u: np.ndarray, d: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """
        Calcule les forces internes incluant les contributions volumiques et cohésives
        
        Parameters:
            u: Déplacements
            d: Endommagement
            use_cache: Utiliser le cache si disponible
            
        Returns:
            Forces internes totales
        """
        # Vérifier le cache
        if use_cache and self._internal_forces_cache is not None:
            if np.array_equal(u, self._cache_u) and np.array_equal(d, self._cache_d):
                return self._internal_forces_cache.copy()
        
        f_int = np.zeros(self.mesh.num_dofs_u)
        
        # Forces des éléments volumiques
        for e in range(self.mesh.num_elements):
            # Récupérer la matrice de rigidité élémentaire
            K_elem, _, _ = self.elem_matrices.get_stiffness_matrix(e, u, d)
            
            # Déplacements élémentaires
            element_nodes = self.mesh.elements[e]
            u_elem = np.zeros(8)
            for i in range(4):
                node = element_nodes[i]
                u_elem[2*i] = u[self.mesh.dof_map_u[node, 0]]
                u_elem[2*i+1] = u[self.mesh.dof_map_u[node, 1]]
            
            # Forces internes élémentaires
            f_elem = K_elem @ u_elem
            
            # Assembler
            dofs = self._get_element_dofs(element_nodes)
            for i in range(8):
                f_int[dofs[i]] += f_elem[i]
        
        # Forces cohésives
        if self.cohesive is not None:
            f_coh = self.cohesive.calculate_interface_forces(u)
            f_int += f_coh
            #f_int -= f_coh
        
        # Mettre en cache si demandé
        if use_cache:
            self._internal_forces_cache = f_int.copy()
            self._cache_u = u.copy()
            self._cache_d = d.copy()
        
        return f_int
    
    def reset_projection_tensors(self):
        """Réinitialise les tenseurs de projection"""
        self._initialize_projection_tensors()


class ResidualCalculator:
    """Calcul des résidus pour les solveurs non-linéaires"""
    
    def __init__(self, system_assembler: SystemAssembler):
        self.assembler = system_assembler
    
    def compute_mechanical_residual(self, u: np.ndarray, v: np.ndarray, a: np.ndarray,
                                  u_prev: np.ndarray, v_prev: np.ndarray, a_prev: np.ndarray,
                                  d: np.ndarray, time: float, dt: float,
                                  params: Dict) -> np.ndarray:
        """
        Calcule le résidu mécanique pour le schéma HHT-α
        
        R = M*a_{n+1} + (1+α)*f_int(u_{n+1}) - α*f_int(u_n) - (1+α)*f_ext(t_{n+1}) + α*f_ext(t_n)
        """
        alpha = params['alpha_HHT']
        loading_params = params['loading_params']
        
        # Assembler les matrices au temps n+1
        K_curr, M, f_ext_curr = self.assembler.assemble_system(
            u, d, time, loading_params, params.get('use_decomposition', False)
        )
        
        # Forces internes au temps n+1
        f_int_curr = self.assembler.get_internal_forces(u, d)
        
        # Forces au temps n si nécessaire
        if abs(alpha) > 1e-10:
            # Forces externes au temps n
            _, _, f_ext_prev = self.assembler.assemble_system(
                u_prev, d, time - dt, loading_params, params.get('use_decomposition', False)
            )
            # Forces internes au temps n
            f_int_prev = self.assembler.get_internal_forces(u_prev, d)
        else:
            f_int_prev = np.zeros_like(f_int_curr)
            f_ext_prev = np.zeros_like(f_ext_curr)
        
        # Résidu HHT-α
        residual = (M @ a + 
                   (1.0 + alpha) * f_int_curr - 
                   alpha * f_int_prev - 
                   (1.0 + alpha) * f_ext_curr + 
                   alpha * f_ext_prev)
        
        # Appliquer les conditions aux limites au résidu
        bc_dict = self.assembler.mesh.get_boundary_conditions()
        self._apply_bc_to_residual(residual, bc_dict)
        
        return residual
    
    def compute_staggered_residual(self, u_old: np.ndarray, u_new: np.ndarray,
                                 d_old: np.ndarray, d_new: np.ndarray) -> Dict[str, float]:
        """
        Calcule les résidus pour la convergence du schéma décalé
        
        Returns:
            Dictionnaire avec les normes des résidus
        """
        # Résidu en déplacement
        u_diff = np.linalg.norm(u_new - u_old)
        u_norm = np.linalg.norm(u_new) + 1e-10
        u_residual = u_diff / u_norm
        
        # Résidu en endommagement
        d_diff = np.linalg.norm(d_new - d_old)
        d_norm = np.linalg.norm(d_new) + 1e-10
        d_residual = d_diff / d_norm
        
        # Résidu combiné
        combined_residual = np.sqrt(u_residual**2 + d_residual**2)
        
        return {
            'displacement': u_residual,
            'damage': d_residual,
            'combined': combined_residual,
            'u_diff': u_diff,
            'd_diff': d_diff
        }
    
    def _apply_bc_to_residual(self, residual: np.ndarray, bc_dict: Dict):
        """Applique les conditions aux limites au résidu"""
        # Nœuds complètement fixés
        for node in bc_dict.get('fully_fixed', []):
            for dof in range(2):
                dof_idx = self.assembler.mesh.dof_map_u[node, dof]
                residual[dof_idx] = 0.0
        
        # Nœuds fixés en x
        for node in bc_dict.get('fixed_x', []):
            dof_idx = self.assembler.mesh.dof_map_u[node, 0]
            residual[dof_idx] = 0.0
        
        # Nœuds fixés en y
        for node in bc_dict.get('fixed_y', []):
            dof_idx = self.assembler.mesh.dof_map_u[node, 1]
            residual[dof_idx] = 0.0