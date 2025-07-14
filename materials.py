"""
Module de gestion des propriétés matériaux et décomposition spectrale
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Dict, Optional
from enum import Enum


class MaterialType(Enum):
    SUBSTRATE = 0
    ICE = 1


@dataclass
class MaterialProperties:
    """Propriétés d'un matériau"""
    E: float        # Module de Young (MPa)
    nu: float       # Coefficient de Poisson
    rho: float      # Densité (ton/mm³)
    Gc: float       # Énergie de rupture (N/mm)
    name: str       # Nom du matériau


@dataclass
class CohesiveProperties:
    """Propriétés cohésives pour l'interface"""
    # Mode I (normal)
    normal_stiffness: float = 1.0e+6      # Rigidité normale (MPa/mm)
    normal_strength: float = 0.3          # Résistance normale (MPa)
    normal_Gc: float = 0.00056           # Énergie de rupture normale (N/mm)
    compression_factor: float = 50.0      # Facteur de pénalité en compression
    
    # Mode II (cisaillement)  
    shear_stiffness: float = 1.0e+6       # Rigidité en cisaillement (MPa/mm)
    shear_strength: float = 0.3           # Résistance en cisaillement (MPa)
    shear_Gc: float = 0.00056            # Énergie de rupture en cisaillement (N/mm)
    
    # Mixité des modes
    fixed_mixity: Optional[float] = 0.5   # 0.0 = Mode I pur, 1.0 = Mode II pur
    
    # Viscosité artificielle pour la stabilisation
    viscosity: float = 1.0e-4             # Paramètre de viscosité
    
    def __post_init__(self):
        # Calcul des paramètres dérivés
        self.normal_delta0 = self.normal_strength / self.normal_stiffness
        self.shear_delta0 = self.shear_strength / self.shear_stiffness
        self.normal_deltac = 2.0 * self.normal_Gc / self.normal_strength
        self.shear_deltac = 2.0 * self.shear_Gc / self.shear_strength


class MaterialManager:
    """Gestionnaire des propriétés matériaux"""
    
    def __init__(self, ice_props: MaterialProperties, 
                 substrate_props: MaterialProperties,
                 cohesive_props: CohesiveProperties,
                 plane_strain: bool = True,
                 k_res: float = 1.0e-10):
        
        self.materials = {
            MaterialType.ICE: ice_props,
            MaterialType.SUBSTRATE: substrate_props
        }
        self.cohesive = cohesive_props
        self.plane_strain = plane_strain
        self.k_res = k_res
        
        # Paramètre de décomposition spectrale
        self.use_decomposition = False
        self.l0 = 1.0  # Longueur caractéristique
        
        # Cache pour les matrices constitutives
        self._constitutive_cache: Dict[MaterialType, np.ndarray] = {}
        
    def get_properties(self, material_id: int) -> MaterialProperties:
        """Retourne les propriétés d'un matériau par son ID"""
        mat_type = MaterialType(material_id)
        return self.materials[mat_type]
    
    def get_constitutive_matrix(self, material_id: int) -> np.ndarray:
        """
        Retourne la matrice constitutive pour un matériau donné
        """
        mat_type = MaterialType(material_id)
        
        if mat_type in self._constitutive_cache:
            return self._constitutive_cache[mat_type].copy()
        
        props = self.materials[mat_type]
        D = self._compute_constitutive_matrix(props.E, props.nu)
        
        self._constitutive_cache[mat_type] = D.copy()
        
        return D
    
    def _compute_constitutive_matrix(self, E: float, nu: float) -> np.ndarray:
        """Calcule la matrice constitutive"""
        if self.plane_strain:
            # Déformation plane
            factor = E / ((1.0 + nu) * (1.0 - 2.0 * nu))
            D = np.array([
                [1.0 - nu, nu, 0.0],
                [nu, 1.0 - nu, 0.0],
                [0.0, 0.0, (1.0 - 2.0 * nu) / 2.0]
            ], dtype=np.float64) * factor
        else:
            # Contrainte plane
            factor = E / (1.0 - nu**2)
            D = np.array([
                [1.0, nu, 0.0],
                [nu, 1.0, 0.0],
                [0.0, 0.0, (1.0 - nu) / 2.0]
            ], dtype=np.float64) * factor
        
        return D
    
    def get_stiffness_sqrt_matrices(self, E: float, nu: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule C^(1/2) et C^(-1/2) pour la méthode SD3
        """
        if self.plane_strain:
            # Calcul direct pour le cas isotrope en déformation plane
            kappa = E / (3.0 * (1.0 - 2.0 * nu))  # Module de compressibilité
            mu = E / (2.0 * (1.0 + nu))            # Module de cisaillement
            
            sqrt_kappa = np.sqrt(kappa)
            sqrt_mu = np.sqrt(mu)
            
            # C^(1/2) en notation de Voigt
            C_sqrt = np.array([
                [sqrt_kappa/np.sqrt(2) + sqrt_mu/np.sqrt(2), 
                 sqrt_kappa/np.sqrt(2) - sqrt_mu/np.sqrt(2), 0],
                [sqrt_kappa/np.sqrt(2) - sqrt_mu/np.sqrt(2), 
                 sqrt_kappa/np.sqrt(2) + sqrt_mu/np.sqrt(2), 0],
                [0, 0, np.sqrt(2)*sqrt_mu]
            ], dtype=np.float64)
            
            # C^(-1/2)
            C_inv_sqrt = np.array([
                [1/(2*np.sqrt(2*kappa)) + 1/(2*np.sqrt(2*mu)), 
                 1/(2*np.sqrt(2*kappa)) - 1/(2*np.sqrt(2*mu)), 0],
                [1/(2*np.sqrt(2*kappa)) - 1/(2*np.sqrt(2*mu)), 
                 1/(2*np.sqrt(2*kappa)) + 1/(2*np.sqrt(2*mu)), 0],
                [0, 0, 1/np.sqrt(2*mu)]
            ], dtype=np.float64)
        else:
            # Pour le cas général, utiliser la décomposition propre
            C = self._compute_constitutive_matrix(E, nu)
            eigenvalues, eigenvectors = np.linalg.eigh(C)
            
            if np.any(eigenvalues <= 0):
                raise ValueError(f"Matrice de rigidité non définie positive")
            
            sqrt_eigenvalues = np.sqrt(eigenvalues)
            C_sqrt = eigenvectors @ np.diag(sqrt_eigenvalues) @ eigenvectors.T
            
            inv_sqrt_eigenvalues = 1.0 / sqrt_eigenvalues
            C_inv_sqrt = eigenvectors @ np.diag(inv_sqrt_eigenvalues) @ eigenvectors.T
        
        return C_sqrt, C_inv_sqrt
    
    def degradation_function(self, damage: float) -> float:
        """Fonction de dégradation g(d) = (1-d)² + k_res"""
        return (1.0 - damage)**2 + self.k_res
    
    def calculate_effective_properties(self, delta_n: float, delta_t: float) -> Dict[str, float]:
        """
        Calcule les propriétés effectives basées sur la mixité des modes
        """
        delta_n_pos = max(0.0, delta_n)
        
        if self.cohesive.fixed_mixity is not None:
            mixity = self.cohesive.fixed_mixity
        else:
            beta = abs(delta_t) / (abs(delta_n_pos) + 1e-12)
            beta_squared = beta**2
            mixity = beta_squared / (1.0 + beta_squared)
        
        delta0_eff = (self.cohesive.normal_delta0 + 
                      (self.cohesive.shear_delta0 - self.cohesive.normal_delta0) * mixity)
        deltac_eff = (self.cohesive.normal_deltac + 
                      (self.cohesive.shear_deltac - self.cohesive.normal_deltac) * mixity)
        Gc_eff = (self.cohesive.normal_Gc + 
                  (self.cohesive.shear_Gc - self.cohesive.normal_Gc) * mixity)
        
        return {
            'delta0_eff': delta0_eff,
            'deltac_eff': deltac_eff,
            'Gc_eff': Gc_eff,
            'mixity': mixity
        }


class SpectralDecomposition:
    """Décomposition spectrale orthogonale SD3 selon He and Shao"""
    
    def __init__(self, material_manager: MaterialManager):
        self.material_manager = material_manager
        self.ZERO_TOL = 1e-12
    
    def decompose(self, strain_vector: np.ndarray, E: float, nu: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Décomposition spectrale orthogonale SD3
        
        Returns:
            strain_pos: Partie positive de la déformation
            strain_neg: Partie négative de la déformation
            P_pos: Projecteur positif
            P_neg: Projecteur négatif
        """
        # Vérifier les petites déformations
        strain_norm = np.linalg.norm(strain_vector)
        if strain_norm < self.ZERO_TOL:
            zero_strain = np.zeros_like(strain_vector)
            P_pos = np.eye(3, dtype=np.float64)
            P_neg = np.zeros((3, 3), dtype=np.float64)
            return zero_strain, zero_strain, P_pos, P_neg
        
        # Obtenir les matrices racines carrées
        C_sqrt, C_inv_sqrt = self.material_manager.get_stiffness_sqrt_matrices(E, nu)
        
        # Transformer la déformation: ε̃ = C^(1/2) : ε
        strain_tilde = C_sqrt @ strain_vector
        
        # Calcul des valeurs propres et projecteurs de ε̃
        epsilon_tilde_1, epsilon_tilde_2, E_tilde_1, E_tilde_2 = self._compute_eigenvalues_2D(strain_tilde)
        
        # Parties positive et négative dans l'espace transformé
        eps1_tilde_pos = max(0.0, epsilon_tilde_1)
        eps1_tilde_neg = min(0.0, epsilon_tilde_1)
        eps2_tilde_pos = max(0.0, epsilon_tilde_2)
        eps2_tilde_neg = min(0.0, epsilon_tilde_2)
        
        strain_tilde_pos = eps1_tilde_pos * E_tilde_1 + eps2_tilde_pos * E_tilde_2
        strain_tilde_neg = eps1_tilde_neg * E_tilde_1 + eps2_tilde_neg * E_tilde_2
        
        # Retour à l'espace original: ε± = C^(-1/2) : ε̃±
        strain_pos = C_inv_sqrt @ strain_tilde_pos
        strain_neg = C_inv_sqrt @ strain_tilde_neg
        
        # Calculer les tenseurs de projection
        P_tilde_pos = np.zeros((3, 3), dtype=np.float64)
        P_tilde_neg = np.zeros((3, 3), dtype=np.float64)
        
        if epsilon_tilde_1 > 0:
            P_tilde_pos += np.outer(E_tilde_1, E_tilde_1)
        else:
            P_tilde_neg += np.outer(E_tilde_1, E_tilde_1)
        
        if epsilon_tilde_2 > 0:
            P_tilde_pos += np.outer(E_tilde_2, E_tilde_2)
        else:
            P_tilde_neg += np.outer(E_tilde_2, E_tilde_2)
        
        # Projecteurs dans l'espace original: P± = C^(-1/2) : P̃± : C^(1/2)
        P_pos = C_inv_sqrt @ P_tilde_pos @ C_sqrt
        P_neg = C_inv_sqrt @ P_tilde_neg @ C_sqrt
        
        return strain_pos, strain_neg, P_pos, P_neg
    
    def _compute_eigenvalues_2D(self, strain_tilde: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Calcule les valeurs propres et projecteurs propres en notation de Voigt
        """
        # Invariants
        I1 = strain_tilde[0] + strain_tilde[1]
        I2 = strain_tilde[0] * strain_tilde[1] - 0.25 * strain_tilde[2]**2
        
        discriminant = I1**2 - 4.0 * I2
        if discriminant < 0:
            discriminant = 0.0
        
        sqrt_disc = np.sqrt(discriminant)
        
        # Valeurs propres
        epsilon1 = 0.5 * (I1 + sqrt_disc)
        epsilon2 = 0.5 * (I1 - sqrt_disc)
        
        # Projecteurs
        if abs(epsilon1 - epsilon2) > self.ZERO_TOL:
            # Projecteur E1 en notation de Voigt
            denom = epsilon1 - epsilon2
            E1 = np.array([
                (strain_tilde[0] - epsilon2) / denom,
                (strain_tilde[1] - epsilon2) / denom,
                strain_tilde[2] / denom
            ], dtype=np.float64)
            
            # E2 = I - E1
            E2 = np.array([1.0 - E1[0], 1.0 - E1[1], -E1[2]], dtype=np.float64)
        else:
            # Valeurs propres égales
            E1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            E2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        
        return epsilon1, epsilon2, E1, E2