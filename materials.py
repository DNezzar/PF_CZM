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
    """Propriétés cohésives pour l'interface avec valeurs réalistes"""
    # Mode I (normal) - Valeurs réduites pour être réalistes
    normal_stiffness: float = 1.0e+2      # Rigidité normale (MPa/mm) - RÉDUIT
    normal_strength: float = 0.3          # Résistance normale (MPa)
    normal_Gc: float = 0.00056           # Énergie de rupture normale (N/mm)
    compression_factor: float = 50.0      # Facteur de pénalité en compression
    
    # Mode II (cisaillement) - Valeurs réduites pour être réalistes  
    shear_stiffness: float = 1.0e+2       # Rigidité en cisaillement (MPa/mm) - RÉDUIT
    shear_strength: float = 0.3           # Résistance en cisaillement (MPa)
    shear_Gc: float = 0.00056            # Énergie de rupture en cisaillement (N/mm)
    
    # Mixité des modes
    fixed_mixity: Optional[float] = 0.5   # 0.0 = Mode I pur, 1.0 = Mode II pur
    
    # Viscosité artificielle pour la stabilisation
    viscosity: float = 1.0e-4             # Paramètre de viscosité
    
    def __post_init__(self):
        # Calcul des paramètres dérivés avec les bonnes formules
        # delta0 = sigma_c / K (début de l'endommagement)
        self.normal_delta0 = self.normal_strength / self.normal_stiffness
        self.shear_delta0 = self.shear_strength / self.shear_stiffness
        
        # deltac = 2 * Gc / sigma_c (rupture complète)
        self.normal_deltac = 2.0 * self.normal_Gc / self.normal_strength
        self.shear_deltac = 2.0 * self.shear_Gc / self.shear_strength


class MaterialManager:
    """Gestionnaire des propriétés matériaux et des calculs associés"""
    
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
        self.k_res = k_res  # Rigidité résiduelle
        
        # Cache pour les matrices constitutives
        self._constitutive_cache: Dict[MaterialType, np.ndarray] = {}
        self._sqrt_cache: Dict[Tuple[float, float], Tuple[np.ndarray, np.ndarray]] = {}
        
    def get_properties(self, material_id: int) -> MaterialProperties:
        """Retourne les propriétés d'un matériau par son ID"""
        mat_type = MaterialType(material_id)
        return self.materials[mat_type]
    
    def get_constitutive_matrix(self, material_id: int) -> np.ndarray:
        """
        Retourne la matrice constitutive pour un matériau donné
        Utilise un cache pour éviter les recalculs
        """
        mat_type = MaterialType(material_id)
        
        # Vérifier le cache
        if mat_type in self._constitutive_cache:
            return self._constitutive_cache[mat_type].copy()
        
        # Calculer la matrice
        props = self.materials[mat_type]
        D = self._compute_constitutive_matrix(props.E, props.nu)
        
        # Stocker dans le cache
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
        Calcule C^(1/2) et C^(-1/2) pour la décomposition orthogonale SD3
        Équation (9) de Nguyen et al. (2020)
        """
        cache_key = (E, nu)
        
        # Vérifier le cache
        if cache_key in self._sqrt_cache:
            C_sqrt, C_inv_sqrt = self._sqrt_cache[cache_key]
            return C_sqrt.copy(), C_inv_sqrt.copy()
        
        # Validation des entrées
        if E <= 0:
            raise ValueError(f"Le module de Young doit être positif, reçu {E}")
        
        if nu >= 0.5:
            print(f"Attention: Coefficient de Poisson {nu} >= 0.5 (limite incompressible)")
            nu = 0.495
        elif nu <= -1.0:
            raise ValueError(f"Coefficient de Poisson invalide {nu} <= -1.0")
        
        # Calcul des modules
        kappa = E / (3.0 * (1.0 - 2.0 * nu))  # Module de compressibilité
        mu = E / (2.0 * (1.0 + nu))           # Module de cisaillement
        
        # Vérification des modules positifs
        if kappa <= 0 or mu <= 0:
            raise ValueError(f"Paramètres matériaux invalides: kappa={kappa}, mu={mu}")
        
        # C^(1/2) selon l'équation (9) de l'article
        sqrt_kappa = np.sqrt(kappa)
        sqrt_mu = np.sqrt(mu)
        sqrt_2mu = np.sqrt(2.0 * mu)
        
        C_sqrt = np.array([
            [sqrt_kappa/2.0 + sqrt_mu/2.0, sqrt_kappa/2.0 - sqrt_mu/2.0, 0.0],
            [sqrt_kappa/2.0 - sqrt_mu/2.0, sqrt_kappa/2.0 + sqrt_mu/2.0, 0.0],
            [0.0, 0.0, sqrt_2mu]
        ], dtype=np.float64)
        
        # C^(-1/2) selon l'équation (9) de l'article
        inv_sqrt_2kappa = 1.0 / np.sqrt(2.0 * kappa)
        inv_sqrt_2mu = 1.0 / np.sqrt(2.0 * mu)
        inv_sqrt_mu = 1.0 / sqrt_mu
        
        C_inv_sqrt = np.array([
            [inv_sqrt_2kappa + inv_sqrt_2mu, inv_sqrt_2kappa - inv_sqrt_2mu, 0.0],
            [inv_sqrt_2kappa - inv_sqrt_2mu, inv_sqrt_2kappa + inv_sqrt_2mu, 0.0],
            [0.0, 0.0, inv_sqrt_mu]
        ], dtype=np.float64)
        
        # Vérification
        identity_check = C_sqrt @ C_inv_sqrt
        error = np.linalg.norm(identity_check - np.eye(3))
        if error > 1e-10:
            print(f"Attention: C_sqrt @ C_inv_sqrt dévie de l'identité de {error:.2e}")
        
        # Stocker dans le cache
        self._sqrt_cache[cache_key] = (C_sqrt.copy(), C_inv_sqrt.copy())
        
        return C_sqrt, C_inv_sqrt
    
    def degradation_function(self, damage: float) -> float:
        """Fonction de dégradation g(d) = (1-d)² + k_res"""
        return (1.0 - damage)**2 + self.k_res
    
    def get_cohesive_delta0(self, mode: str) -> float:
        """Retourne delta0 pour le mode spécifié"""
        if mode.lower() == 'normal':
            return self.cohesive.normal_delta0
        elif mode.lower() == 'shear':
            return self.cohesive.shear_delta0
        else:
            raise ValueError(f"Mode inconnu: {mode}")
    
    def get_cohesive_deltac(self, mode: str) -> float:
        """Retourne deltac pour le mode spécifié"""
        if mode.lower() == 'normal':
            return self.cohesive.normal_deltac
        elif mode.lower() == 'shear':
            return self.cohesive.shear_deltac
        else:
            raise ValueError(f"Mode inconnu: {mode}")
    
    def calculate_effective_properties(self, delta_n: float, delta_t: float) -> Dict[str, float]:
        """
        Calcule les propriétés effectives basées sur la mixité des modes
        
        Returns:
            Dict contenant delta0_eff, deltac_eff, Gc_eff, et mixity
        """
        delta_n_pos = max(0.0, delta_n)
        
        if self.cohesive.fixed_mixity is not None:
            mixity = self.cohesive.fixed_mixity
        else:
            # Calcul dynamique de la mixité
            beta = abs(delta_t) / (abs(delta_n_pos) + 1e-12)
            beta_squared = beta**2
            mixity = beta_squared / (1.0 + beta_squared)
        
        # Propriétés effectives
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
    """Classe pour la décomposition spectrale orthogonale SD3"""
    
    def __init__(self, material_manager: MaterialManager):
        self.material_manager = material_manager
        self.ZERO_TOL = 1e-16
    
    def decompose(self, strain_vector: np.ndarray, E: float, nu: float, 
                  verif: bool = False, debug: bool = False) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Décomposition spectrale orthogonale de la déformation (SD3)
        Basée sur He and Shao (2019) - implémentée dans Nguyen et al. (2020)

        Returns:
            strain_pos: Partie positive de la déformation
            strain_neg: Partie négative de la déformation
            P_pos: Tenseur de projection positif
            P_neg: Tenseur de projection négatif
        """
        # Validation des entrées
        if np.any(np.isnan(strain_vector)) or np.any(np.isinf(strain_vector)):
            print(f"Attention: Vecteur de déformation invalide détecté: {strain_vector}")
            zero_strain = np.zeros_like(strain_vector)
            identity = np.eye(3, dtype=np.float64)
            return zero_strain, zero_strain, identity, identity

        # Vérifier les petites déformations
        strain_norm = np.linalg.norm(strain_vector)
        if strain_norm < self.ZERO_TOL:
            if debug:
                print("Déformation très petite détectée, retour de décomposition nulle")
            zero_strain = np.zeros_like(strain_vector)
            identity = np.eye(3, dtype=np.float64)
            return zero_strain, zero_strain, identity, identity

        try:
            # Obtenir les matrices racines carrées
            C_sqrt, C_inv_sqrt = self.material_manager.get_stiffness_sqrt_matrices(E, nu)
        except Exception as e:
            print(f"Erreur dans get_stiffness_sqrt_matrices: {e}")
            half_strain = strain_vector * 0.5
            identity = np.eye(3, dtype=np.float64)
            return half_strain, half_strain, identity, identity

        # Transformer la déformation: ε̃ = C^(1/2) : ε
        strain_tilde = C_sqrt @ strain_vector

        if debug:
            print(f"Déformation originale: {strain_vector}")
            print(f"Déformation transformée: {strain_tilde}")

        # Calcul des valeurs propres et décomposition spectrale de ε̃
        epsilon_tilde_1, epsilon_tilde_2, E_tilde_1, E_tilde_2 = self._compute_eigenvalues_2D(strain_tilde)

        # Appliquer les crochets de Macaulay
        eps1_tilde_pos = max(0.0, epsilon_tilde_1)
        eps1_tilde_neg = min(0.0, epsilon_tilde_1)
        eps2_tilde_pos = max(0.0, epsilon_tilde_2)
        eps2_tilde_neg = min(0.0, epsilon_tilde_2)

        # Reconstruire les parties positive et négative dans l'espace transformé
        strain_tilde_pos = eps1_tilde_pos * E_tilde_1 + eps2_tilde_pos * E_tilde_2
        strain_tilde_neg = eps1_tilde_neg * E_tilde_1 + eps2_tilde_neg * E_tilde_2

        # Retourner à l'espace original: ε± = C^(-1/2) : ε̃±
        strain_pos = C_inv_sqrt @ strain_tilde_pos
        strain_neg = C_inv_sqrt @ strain_tilde_neg

        # Calculer les tenseurs de projection
        P_pos, P_neg = self._compute_projection_tensors_SD3(
            E_tilde_1, E_tilde_2, epsilon_tilde_1, epsilon_tilde_2, 
            C_sqrt, C_inv_sqrt
        )

        # Vérification si demandée
        if verif or debug:
            self._verify_decomposition(strain_vector, strain_pos, strain_neg, 
                                     P_pos, P_neg, E, nu, debug)

        return strain_pos, strain_neg, P_pos, P_neg    
    
    def _compute_eigenvalues_2D(self, strain_tilde: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
        """
        Calcule les valeurs propres et projecteurs propres pour la déformation transformée ε̃
        en notation de Voigt
        """
        # Invariants de ε̃ (notation de Voigt)
        I1 = strain_tilde[0] + strain_tilde[1]  # trace
        I2 = strain_tilde[0] * strain_tilde[1] - 0.25 * strain_tilde[2]**2  # déterminant

        # Discriminant
        discriminant = I1**2 - 4.0 * I2

        if discriminant < 0:
            discriminant = 0.0

        sqrt_disc = np.sqrt(discriminant)

        # Valeurs propres
        epsilon1 = 0.5 * (I1 + sqrt_disc)
        epsilon2 = 0.5 * (I1 - sqrt_disc)

        # Cas où les valeurs propres sont distinctes
        if abs(epsilon1 - epsilon2) > self.ZERO_TOL:
            # Projecteur E1 en notation de Voigt
            E1 = np.zeros(3, dtype=np.float64)
            E1[0] = (strain_tilde[0] - epsilon2) / (epsilon1 - epsilon2)
            E1[1] = (strain_tilde[1] - epsilon2) / (epsilon1 - epsilon2)
            E1[2] = strain_tilde[2] / (epsilon1 - epsilon2)

            # Projecteur E2 = I - E1
            E2 = np.array([1.0 - E1[0], 1.0 - E1[1], -E1[2]], dtype=np.float64)
        else:
            # Valeurs propres égales - identité décomposée
            E1 = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            E2 = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        return epsilon1, epsilon2, E1, E2    
    
    def _compute_projection_tensors_SD3(self, E1: np.ndarray, E2: np.ndarray, 
                                       epsilon1: float, epsilon2: float,
                                       C_sqrt: np.ndarray, C_inv_sqrt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calcule les tenseurs de projection dans l'espace original selon SD3
        """
        # Initialiser les tenseurs de projection dans l'espace transformé
        P_tilde_pos = np.zeros((3, 3), dtype=np.float64)
        P_tilde_neg = np.zeros((3, 3), dtype=np.float64)

        # Construire les projecteurs dans l'espace transformé
        if epsilon1 > 0:
            P_tilde_pos += np.outer(E1, E1)
        else:
            P_tilde_neg += np.outer(E1, E1)

        if epsilon2 > 0:
            P_tilde_pos += np.outer(E2, E2)
        else:
            P_tilde_neg += np.outer(E2, E2)

        # Corriger pour la notation de Voigt (terme de cisaillement)
        # Les termes hors diagonale doivent être divisés par 2
        for i in range(2):
            P_tilde_pos[i, 2] *= 0.5
            P_tilde_pos[2, i] *= 0.5
            P_tilde_neg[i, 2] *= 0.5
            P_tilde_neg[2, i] *= 0.5
        P_tilde_pos[2, 2] *= 0.25
        P_tilde_neg[2, 2] *= 0.25

        # Transformer les tenseurs de projection dans l'espace original
        # P± = C^(-1/2) : P̃± : C^(1/2)
        P_pos = C_inv_sqrt @ P_tilde_pos @ C_sqrt
        P_neg = C_inv_sqrt @ P_tilde_neg @ C_sqrt

        return P_pos, P_neg
    
    def _verify_decomposition(self, strain: np.ndarray, strain_pos: np.ndarray, 
                            strain_neg: np.ndarray, P_pos: np.ndarray, P_neg: np.ndarray,
                            E: float, nu: float, debug: bool):
        """Vérifie la validité de la décomposition"""
        # Vérifier que strain = strain_pos + strain_neg
        strain_reconstructed = strain_pos + strain_neg
        recon_error = np.linalg.norm(strain - strain_reconstructed)
        if recon_error > 1e-10 or debug:
            print(f"Erreur de reconstruction: {recon_error:.3e}")
        
        # Vérifier l'orthogonalité
        D = self.material_manager._compute_constitutive_matrix(E, nu)
        ortho_check = np.abs(strain_pos @ D @ strain_neg)
        if ortho_check > 1e-10 or debug:
            print(f"Erreur d'orthogonalité: {ortho_check:.3e}")
        
        # Vérifier l'idempotence des projecteurs
        P_pos_idemp_error = np.linalg.norm(P_pos @ P_pos - P_pos)
        P_neg_idemp_error = np.linalg.norm(P_neg @ P_neg - P_neg)
        if (P_pos_idemp_error > 1e-10 or P_neg_idemp_error > 1e-10) or debug:
            print(f"Erreur d'idempotence P_pos: {P_pos_idemp_error:.3e}")
            print(f"Erreur d'idempotence P_neg: {P_neg_idemp_error:.3e}")