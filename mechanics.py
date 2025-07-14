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
        
        # Cache pour les calculs
        self._B_cache = {}
        self._detJ_cache = {}
    
    def get_stiffness_matrix(self, elem_id: int, u: np.ndarray, 
                           d: np.ndarray, P_pos_elem: List[np.ndarray], 
                           P_neg_elem: List[np.ndarray]) -> np.ndarray:
        """
        Calcule la matrice de rigidité élémentaire selon SD3
        """
        # Récupérer les données de l'élément
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        # Déplacements nodaux
        u_elem = self._extract_element_displacements(element_nodes, u)
        
        # Propriétés matériaux
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        D = self.materials.get_constitutive_matrix(self.mesh.material_id[elem_id])
        
        # Initialiser la matrice
        K_elem = np.zeros((8, 8), dtype=np.float64)
        
        # Intégration sur les points de Gauss
        for gp_idx, (xi, eta) in enumerate(self.gauss_points):
            # Matrice B et jacobien
            B, detJ = self._get_B_matrix_and_jacobian(elem_id, xi, eta, x_coords, y_coords)
            
            # Matrice constitutive effective
            if self.mesh.material_id[elem_id] == 1:  # Glace
                # Interpoler l'endommagement
                N = self._shape_functions(xi, eta)
                damage_gauss = np.dot(N, d[element_nodes])
                g_d = self.materials.degradation_function(damage_gauss)
                
                if self.materials.use_decomposition and gp_idx < len(P_pos_elem):
                    # Utiliser les projecteurs fournis
                    P_pos = P_pos_elem[gp_idx]
                    P_neg = P_neg_elem[gp_idx]
                    
                    # Matrice constitutive dégradée selon SD3
                    D_effective = g_d * (P_pos.T @ D @ P_pos) + (P_neg.T @ D @ P_neg)
                else:
                    # Sans décomposition: dégradation isotrope
                    D_effective = g_d * D
            else:
                # Substrat: pas de dégradation
                D_effective = D
            
            # Contribution à la matrice de rigidité
            K_elem += B.T @ D_effective @ B * detJ * self.gauss_weights[gp_idx]
        
        return K_elem
    
    def get_mass_matrix(self, elem_id: int) -> np.ndarray:
        """Calcule la matrice de masse élémentaire"""
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        rho = mat_props.rho
        
        M_elem = np.zeros((8, 8), dtype=np.float64)
        
        for xi, eta in self.gauss_points:
            N = self._shape_functions(xi, eta)
            N_matrix = self._build_N_matrix(N)
            detJ = self._calculate_jacobian(xi, eta, x_coords, y_coords)
            M_elem += rho * N_matrix.T @ N_matrix * detJ
        
        return M_elem
    
    def get_centrifugal_force(self, elem_id: int, omega: float, 
                            load_factor: float = 1.0) -> np.ndarray:
        """Calcule le vecteur de force centrifuge élémentaire"""
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        rho = mat_props.rho
        
        f_elem = np.zeros(8, dtype=np.float64)
        
        for xi, eta in self.gauss_points:
            N = self._shape_functions(xi, eta)
            x_gauss = np.dot(N, x_coords)
            f_centrifugal = load_factor * rho * omega**2 * x_gauss
            detJ = self._calculate_jacobian(xi, eta, x_coords, y_coords)
            
            for i in range(4):
                f_elem[2*i] += N[i] * f_centrifugal * detJ
        
        return f_elem
    
    def compute_strain_and_projectors(self, elem_id: int, u: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """
        Calcule les projecteurs P+ et P- pour chaque point de Gauss selon SD3
        """
        element_nodes = self.mesh.elements[elem_id]
        x_coords = self.mesh.nodes[element_nodes, 0]
        y_coords = self.mesh.nodes[element_nodes, 1]
        
        u_elem = self._extract_element_displacements(element_nodes, u)
        
        mat_props = self.materials.get_properties(self.mesh.material_id[elem_id])
        
        P_pos_list = []
        P_neg_list = []
        
        # Si pas de décomposition ou substrat, utiliser l'identité
        if not self.materials.use_decomposition or self.mesh.material_id[elem_id] == 0:
            for _ in self.gauss_points:
                P_pos_list.append(np.eye(3))
                P_neg_list.append(np.zeros((3, 3)))
            return P_pos_list, P_neg_list
        
        # Pour la glace avec décomposition
        if hasattr(self.materials, 'spectral_decomp'):
            for xi, eta in self.gauss_points:
                B, _ = self._get_B_matrix_and_jacobian(elem_id, xi, eta, x_coords, y_coords)
                strain = B @ u_elem
                
                # Décomposition spectrale SD3
                _, _, P_pos, P_neg = self.materials.spectral_decomp.decompose(
                    strain, mat_props.E, mat_props.nu
                )
                
                P_pos_list.append(P_pos)
                P_neg_list.append(P_neg)
        else:
            # Fallback si pas de décomposition spectrale
            for _ in self.gauss_points:
                P_pos_list.append(np.eye(3))
                P_neg_list.append(np.zeros((3, 3)))
        
        return P_pos_list, P_neg_list
    
    def _shape_functions(self, xi: float, eta: float) -> np.ndarray:
        """Fonctions de forme"""
        N1 = 0.25 * (1.0 - xi) * (1.0 - eta)
        N2 = 0.25 * (1.0 + xi) * (1.0 - eta)
        N3 = 0.25 * (1.0 + xi) * (1.0 + eta)
        N4 = 0.25 * (1.0 - xi) * (1.0 + eta)
        return np.array([N1, N2, N3, N4])
    
    def _get_B_matrix_and_jacobian(self, elem_id: int, xi: float, eta: float,
                                  x_coords: np.ndarray, y_coords: np.ndarray) -> Tuple[np.ndarray, float]:
        """Calcule la matrice B et le jacobien"""
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
        
        self._B_cache[cache_key] = B
        self._detJ_cache[cache_key] = detJ
        
        return B, detJ
    
    def _calculate_jacobian(self, xi: float, eta: float,
                          x_coords: np.ndarray, y_coords: np.ndarray) -> float:
        """Calcule le déterminant du jacobien"""
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
        
        J11 = np.dot(dN_dxi, x_coords)
        J12 = np.dot(dN_dxi, y_coords)
        J21 = np.dot(dN_deta, x_coords)
        J22 = np.dot(dN_deta, y_coords)
        
        return J11 * J22 - J12 * J21
    
    def _build_N_matrix(self, N: np.ndarray) -> np.ndarray:
        """Construit la matrice N étendue"""
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
        
        # Stockage des projecteurs
        self.P_pos_stored = {}
        self.P_neg_stored = {}
        
        # Cache pour les forces internes
        self._internal_forces_cache = None
        self._cache_u = None
        self._cache_d = None
    
    def assemble_system(self, u: np.ndarray, u_prev: np.ndarray, d: np.ndarray, 
                       time: float, loading_params: LoadingParameters,
                       use_decomposition: bool = False) -> Tuple[csr_matrix, csr_matrix, np.ndarray]:
        """
        Assemble les matrices et vecteurs du système global selon SD3
        """
        # Initialiser les matrices creuses
        K = lil_matrix((self.mesh.num_dofs_u, self.mesh.num_dofs_u), dtype=np.float64)
        M = lil_matrix((self.mesh.num_dofs_u, self.mesh.num_dofs_u), dtype=np.float64)
        f_ext = np.zeros(self.mesh.num_dofs_u, dtype=np.float64)
        
        # Facteur de charge
        load_factor = loading_params.get_load_factor(time)
        
        # Calculer d'abord tous les projecteurs si décomposition activée
        if use_decomposition:
            for e in range(self.mesh.num_elements):
                if self.mesh.material_id[e] == 1:  # Glace seulement
                    # Utiliser u_prev pour calculer les projecteurs
                    P_pos_elem, P_neg_elem = self.elem_matrices.compute_strain_and_projectors(e, u_prev)
                    self.P_pos_stored[e] = P_pos_elem
                    self.P_neg_stored[e] = P_neg_elem
        
        # Assembler les éléments
        for e in range(self.mesh.num_elements):
            element_nodes = self.mesh.elements[e]
            dofs = self._get_element_dofs(element_nodes)
            
            # Récupérer les projecteurs stockés
            P_pos_elem = self.P_pos_stored.get(e, [np.eye(3) for _ in range(4)])
            P_neg_elem = self.P_neg_stored.get(e, [np.zeros((3, 3)) for _ in range(4)])
            
            # Matrices élémentaires
            K_elem = self.elem_matrices.get_stiffness_matrix(e, u, d, P_pos_elem, P_neg_elem)
            M_elem = self.elem_matrices.get_mass_matrix(e)
            
            # Forces centrifuges
            if loading_params.load_type == 'centrifugal':
                f_elem = self.elem_matrices.get_centrifugal_force(e, loading_params.omega, load_factor)
            else:
                f_elem = np.zeros(8)
            
            # Assembler
            for i in range(8):
                for j in range(8):
                    K[dofs[i], dofs[j]] += K_elem[i, j]
                    M[dofs[i], dofs[j]] += M_elem[i, j]
                f_ext[dofs[i]] += f_elem[i]
        
        # Éléments cohésifs
        if self.cohesive is not None:
            for i, cohesive_elem in enumerate(self.mesh.cohesive_elements):
                K_coh = self.cohesive.get_cohesive_stiffness(i, u)
                dofs_coh = self._get_cohesive_dofs(cohesive_elem.nodes)
                
                for i in range(8):
                    for j in range(8):
                        K[dofs_coh[i], dofs_coh[j]] += K_coh[i, j]
        
        # Appliquer les conditions aux limites
        bc_dict = self.mesh.get_boundary_conditions()
        self._apply_boundary_conditions(K, M, f_ext, bc_dict)
        
        # Convertir en CSR
        K = K.tocsr()
        M = M.tocsr()
        
        # Invalider le cache
        self._internal_forces_cache = None
        
        return K, M, f_ext
    
    def get_internal_forces(self, u: np.ndarray, d: np.ndarray, use_cache: bool = False) -> np.ndarray:
        """
        Calcule les forces internes selon SD3
        """
        if use_cache and self._internal_forces_cache is not None:
            if np.array_equal(u, self._cache_u) and np.array_equal(d, self._cache_d):
                return self._internal_forces_cache.copy()
        
        f_int = np.zeros(self.mesh.num_dofs_u)
        
        for e in range(self.mesh.num_elements):
            element_nodes = self.mesh.elements[e]
            x_coords = self.mesh.nodes[element_nodes, 0]
            y_coords = self.mesh.nodes[element_nodes, 1]
            
            mat_props = self.materials.get_properties(self.mesh.material_id[e])
            D = self.materials.get_constitutive_matrix(self.mesh.material_id[e])
            
            u_elem = np.zeros(8)
            for i in range(4):
                node = element_nodes[i]
                u_elem[2*i] = u[self.mesh.dof_map_u[node, 0]]
                u_elem[2*i+1] = u[self.mesh.dof_map_u[node, 1]]
            
            f_elem = np.zeros(8)
            
            # Récupérer les projecteurs stockés
            P_pos_elem = self.P_pos_stored.get(e, [np.eye(3) for _ in range(4)])
            P_neg_elem = self.P_neg_stored.get(e, [np.zeros((3, 3)) for _ in range(4)])
            
            for gp_idx, (xi, eta) in enumerate(self.elem_matrices.gauss_points):
                B, detJ = self.elem_matrices._get_B_matrix_and_jacobian(
                    e, xi, eta, x_coords, y_coords
                )
                
                strain = B @ u_elem
                
                if self.mesh.material_id[e] == 1:  # Glace
                    N = self.elem_matrices._shape_functions(xi, eta)
                    damage_gauss = np.dot(N, d[element_nodes])
                    g_d = self.materials.degradation_function(damage_gauss)
                    
                    if self.materials.use_decomposition and gp_idx < len(P_pos_elem):
                        P_pos = P_pos_elem[gp_idx]
                        P_neg = P_neg_elem[gp_idx]
                        
                        # Contrainte selon SD3
                        stress_pos = P_pos.T @ D @ strain
                        stress_neg = P_neg.T @ D @ strain
                        stress = g_d * stress_pos + stress_neg
                    else:
                        stress = g_d * D @ strain
                else:
                    # Substrat
                    stress = D @ strain
                
                f_elem += B.T @ stress * detJ * self.elem_matrices.gauss_weights[gp_idx]
            
            dofs = self._get_element_dofs(element_nodes)
            for i in range(8):
                f_int[dofs[i]] += f_elem[i]
        
        # Forces cohésives
        if self.cohesive is not None:
            f_coh = self.cohesive.calculate_interface_forces(u)
            f_int += f_coh
        
        if use_cache:
            self._internal_forces_cache = f_int.copy()
            self._cache_u = u.copy()
            self._cache_d = d.copy()
        
        return f_int
    
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
        
        for node in bc_dict.get('fixed_x', []):
            dof_idx = self.mesh.dof_map_u[node, 0]
            K[dof_idx, :] = 0.0
            K[:, dof_idx] = 0.0
            K[dof_idx, dof_idx] = 1.0
            
            M[dof_idx, :] = 0.0
            M[:, dof_idx] = 0.0
            M[dof_idx, dof_idx] = 1.0
            
            f[dof_idx] = 0.0
        
        for node in bc_dict.get('fixed_y', []):
            dof_idx = self.mesh.dof_map_u[node, 1]
            K[dof_idx, :] = 0.0
            K[:, dof_idx] = 0.0
            K[dof_idx, dof_idx] = 1.0
            
            M[dof_idx, :] = 0.0
            M[:, dof_idx] = 0.0
            M[dof_idx, dof_idx] = 1.0
            
            f[dof_idx] = 0.0