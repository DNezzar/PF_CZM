"""
Module de gestion du maillage pour le modèle PF-CZM
Gère la génération du maillage, les interfaces et les éléments cohésifs
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional


@dataclass
class MeshParameters:
    """Paramètres pour la génération du maillage"""
    length: float = 170.0
    ice_height: float = 6.4
    substrate_height: float = 6.4
    nx: int = 250
    ny_ice: int = 10
    ny_substrate: int = 5
    bc_type: str = 'left_edge_only'
    # Nouveau paramètre pour activer/désactiver les zones cohésives
    czm_mesh: bool = True
    # Paramètres d'intégration cohésive
    cohesive_integration: str = 'gauss-lobatto'
    cohesive_integration_points: int = 2
    # Paramètres de maillage progressif
    use_coarse_near_bc: bool = True
    coarse_zone_length: float = 20.0
    coarsening_ratio: float = 3.0
    coarse_zone_reduction: float = 0.5
    
    def __post_init__(self):
        self.total_height = self.ice_height + self.substrate_height
        self.ny = self.ny_ice + self.ny_substrate
        self.interface_y = self.substrate_height


def get_cohesive_integration_points(method: str = 'gauss-lobatto', 
                                   n_points: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retourne les points et poids d'intégration pour les éléments cohésifs
    
    Parameters:
        method: 'gauss-lobatto' ou 'newton-cotes'
        n_points: Nombre de points d'intégration (2, 3 ou 4)
        
    Returns:
        points: Coordonnées des points dans [-1, 1]
        weights: Poids d'intégration
    """
    if method == 'gauss-lobatto':
        if n_points == 2:
            points = np.array([-1.0, 1.0])
            weights = np.array([1.0, 1.0])
        elif n_points == 3:
            points = np.array([-1.0, 0.0, 1.0])
            weights = np.array([1.0/3.0, 4.0/3.0, 1.0/3.0])
        elif n_points == 4:
            # Points de Gauss-Lobatto pour n=4
            alpha = np.sqrt(1.0/5.0)
            points = np.array([-1.0, -alpha, alpha, 1.0])
            weights = np.array([1.0/6.0, 5.0/6.0, 5.0/6.0, 1.0/6.0])
        else:
            raise ValueError(f"Gauss-Lobatto avec {n_points} points non supporté. Utilisez 2, 3 ou 4.")
            
    elif method == 'newton-cotes':
        if n_points == 2:  # Règle du trapèze
            points = np.array([-1.0, 1.0])
            weights = np.array([1.0, 1.0])
        elif n_points == 3:  # Règle de Simpson
            points = np.array([-1.0, 0.0, 1.0])
            weights = np.array([1.0/3.0, 4.0/3.0, 1.0/3.0])
        elif n_points == 4:  # Règle de Simpson 3/8
            points = np.array([-1.0, -1.0/3.0, 1.0/3.0, 1.0])
            weights = np.array([1.0/4.0, 3.0/4.0, 3.0/4.0, 1.0/4.0])
        else:
            raise ValueError(f"Newton-Cotes avec {n_points} points non supporté. Utilisez 2, 3 ou 4.")
    
    else:
        raise ValueError(f"Méthode d'intégration '{method}' non reconnue. Utilisez 'gauss-lobatto' ou 'newton-cotes'.")
    
    # Normaliser les poids pour l'intervalle [-1, 1]
    weights = weights * 2.0 / np.sum(weights)
    
    print(f"Intégration cohésive: {method} avec {n_points} points")
    print(f"  Points: {points}")
    print(f"  Poids: {weights}")
    
    return points, weights


@dataclass
class CohesiveElement:
    """Structure pour un élément cohésif"""
    nodes: List[int]
    length: float
    gauss_points: np.ndarray
    gauss_weights: np.ndarray
    damage: np.ndarray = field(init=False)
    damage_prev: np.ndarray = field(init=False)
    
    def __post_init__(self):
        # Initialiser l'endommagement avec le bon nombre de points
        n_points = len(self.gauss_points)
        self.damage = np.zeros(n_points, dtype=np.float64)
        self.damage_prev = np.zeros(n_points, dtype=np.float64)


class MeshManager:
    """Gestionnaire du maillage et des interfaces"""
    
    def __init__(self, params: MeshParameters):
        self.params = params
        self.ZERO_TOL = 1.0e-12
        
        # Attributs du maillage
        self.nodes: Optional[np.ndarray] = None
        self.elements: Optional[np.ndarray] = None
        self.material_id: Optional[np.ndarray] = None
        self.cohesive_elements: List[CohesiveElement] = []
        
        # Nœuds d'interface (vides si czm_mesh=False)
        self.substrate_interface_nodes: List[int] = []
        self.ice_interface_nodes: List[int] = []
        
        # Conditions aux limites
        self.fixed_nodes_x: List[int] = []
        self.fixed_nodes_y: List[int] = []
        self.fully_fixed_nodes: List[int] = []
        
        # Mappings DOF
        self.dof_map_u: Optional[np.ndarray] = None
        self.dof_map_d: Optional[np.ndarray] = None
        
    def generate_mesh(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Génère le maillage structuré pour le système glace-substrat
        
        Returns:
            nodes: Coordonnées des nœuds
            elements: Connectivité des éléments
            material_id: ID des matériaux pour chaque élément
        """
        print("Generating mesh...")
        print(f"  Zones cohésives (CZM): {'Activées' if self.params.czm_mesh else 'Désactivées'}")
        
        # Calcul des tailles d'éléments
        self.hx = self.params.length / self.params.nx
        self.hy_sub = self.params.substrate_height / self.params.ny_substrate
        self.hy_ice = self.params.ice_height / self.params.ny_ice
        
        # Génération des coordonnées
        if self.params.use_coarse_near_bc:
            self._generate_node_coordinates_with_coarsening()
        else:
            self._generate_node_coordinates()
        
        # Création de la connectivité
        self._create_element_connectivity()
        
        # Attribution des matériaux
        self._assign_material_ids()
        
        if self.params.czm_mesh:
            # Mode CZM : créer les nœuds dupliqués et les éléments cohésifs
            self._create_interface_nodes()
            self._update_ice_connectivity()
            self._create_cohesive_elements()
        # Sinon, maillage classique sans duplication
        
        # Application des conditions aux limites
        self._apply_boundary_conditions()
        
        # Création des mappings DOF
        self._create_dof_mappings()
        
        print(f"Mesh generated with {self.num_nodes} nodes and {self.num_elements} elements.")
        if self.params.czm_mesh:
            print(f"Number of cohesive elements at interface: {len(self.cohesive_elements)}")
        else:
            print("Interface with perfect bonding (no cohesive elements)")
        
        return self.nodes, self.elements, self.material_id
    
    def _generate_node_coordinates(self):
        """Génère les coordonnées des nœuds"""
        # Nombre de nœuds en x
        self.num_nodes_x = self.params.nx + 1
        
        # Coordonnées x
        self.x = np.linspace(0.0, self.params.length, self.num_nodes_x, dtype=np.float64)
        
        # Coordonnées y avec espacement différent pour substrat et glace
        y_substrate = np.linspace(0.0, self.params.substrate_height, 
                                 self.params.ny_substrate + 1, dtype=np.float64)
        y_ice = np.linspace(self.params.substrate_height, self.params.total_height, 
                           self.params.ny_ice + 1, dtype=np.float64)
        
        # Stockage des coordonnées originales
        self.y_substrate = y_substrate
        self.y_ice = y_ice
        
        # Combinaison en supprimant le doublon à l'interface
        self.y = np.concatenate([y_substrate, y_ice[1:]])
        self.num_nodes_y = self.params.ny + 1
        self.num_nodes = self.num_nodes_x * self.num_nodes_y
        
        # Création du maillage
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.nodes = np.vstack((self.X.flatten(), self.Y.flatten())).T.astype(np.float64)
    
    def _generate_node_coordinates_with_coarsening(self):
        """Génère les coordonnées avec maillage grossier près de l'encastrement"""
        
        # Paramètres de transition
        x_coarse_end = self.params.coarse_zone_length
        coarsening_ratio = self.params.coarsening_ratio
        
        # Nombre d'éléments dans chaque zone
        n_coarse = int(self.params.nx * x_coarse_end / self.params.length * self.params.coarse_zone_reduction)
        n_normal = self.params.nx - n_coarse
        
        # Génération des coordonnées x avec progression géométrique inverse
        x_coarse = self._geometric_progression_inverse(
            0.0, x_coarse_end, n_coarse + 1, coarsening_ratio
        )
        
        # Maillage uniforme pour le reste
        if n_normal > 0:
            # Taille du dernier élément de la zone grossière
            dx_transition = x_coarse[-1] - x_coarse[-2]
            
            # Pour un bon raccordement, on commence le maillage uniforme directement après x_coarse_end
            # avec une taille d'élément égale à dx_transition
            dx_uniform = (self.params.length - x_coarse_end) / n_normal
            
            # Si la taille uniforme est trop différente de la transition, on fait une transition progressive
            if dx_uniform < dx_transition * 0.8:
                # Transition progressive sur quelques éléments
                n_transition = min(5, n_normal // 4)
                n_uniform = n_normal - n_transition
                
                # Zone de transition
                x_transition = np.zeros(n_transition + 1)
                x_transition[0] = x_coarse_end
                
                # Interpolation linéaire des tailles d'éléments
                for i in range(1, n_transition + 1):
                    factor = i / n_transition
                    dx = dx_transition * (1 - factor) + dx_uniform * factor
                    x_transition[i] = x_transition[i-1] + dx
                
                # Zone uniforme
                if n_uniform > 0:
                    x_uniform = np.linspace(
                        x_transition[-1],
                        self.params.length,
                        n_uniform + 1
                    )
                    x_normal = np.concatenate([x_transition, x_uniform[1:]])
                else:
                    x_normal = x_transition
            else:
                # Pas besoin de transition, maillage uniforme direct
                x_normal = np.linspace(x_coarse_end, self.params.length, n_normal + 1)
            
            # Concatener en évitant la duplication du point de jonction
            self.x = np.concatenate([x_coarse[:-1], x_normal])
        else:
            self.x = x_coarse
        
        # Coordonnées y (inchangées)
        y_substrate = np.linspace(0.0, self.params.substrate_height, 
                                 self.params.ny_substrate + 1, dtype=np.float64)
        y_ice = np.linspace(self.params.substrate_height, self.params.total_height, 
                           self.params.ny_ice + 1, dtype=np.float64)
        
        self.y_substrate = y_substrate
        self.y_ice = y_ice
        self.y = np.concatenate([y_substrate, y_ice[1:]])
        
        # Mise à jour des dimensions
        self.num_nodes_x = len(self.x)
        self.num_nodes_y = len(self.y)
        self.num_nodes = self.num_nodes_x * self.num_nodes_y
        
        # Création du maillage
        self.X, self.Y = np.meshgrid(self.x, self.y)
        self.nodes = np.vstack((self.X.flatten(), self.Y.flatten())).T.astype(np.float64)
        
        # Recalcul du nombre d'éléments
        self.params.nx = self.num_nodes_x - 1
        self.num_elements = self.params.nx * self.params.ny
        
        # Affichage des informations
        print(f"Maillage avec zone grossière près de l'encastrement:")
        print(f"  - Zone grossière: 0 à {x_coarse_end} mm ({n_coarse} éléments)")
        print(f"  - Zone normale: {x_coarse_end} à {self.params.length} mm ({n_normal} éléments)")
        print(f"  - Taille d'élément à x=0: {x_coarse[1]-x_coarse[0]:.3f} mm")
        print(f"  - Taille d'élément à x={x_coarse_end}: {x_coarse[-1]-x_coarse[-2]:.3f} mm")
        
        # Vérifier la continuité
        dx_all = np.diff(self.x)
        max_ratio = np.max(dx_all[1:] / dx_all[:-1])
        print(f"  - Ratio maximal entre éléments adjacents: {max_ratio:.3f}")
    
    def _geometric_progression_inverse(self, x_start: float, x_end: float, 
                                     n_points: int, ratio: float) -> np.ndarray:
        """
        Crée une progression géométrique inverse (gros → fin)
        
        Parameters:
            x_start: Coordonnée de départ
            x_end: Coordonnée de fin  
            n_points: Nombre de points
            ratio: Ratio de taille max/min (>1 pour grossier au début)
        """
        if n_points <= 2:
            return np.linspace(x_start, x_end, n_points)
        
        # Longueur totale
        L = x_end - x_start
        
        if abs(ratio - 1.0) < 1e-10:
            return np.linspace(x_start, x_end, n_points)
        
        # Facteur de progression inverse
        r = 1.0 / ratio**(1.0/(n_points-2))
        
        # Taille du premier segment (le plus gros)
        a = L * (1 - r) / (1 - r**(n_points-1))
        
        # Construction de la progression
        x = np.zeros(n_points)
        x[0] = x_start
        
        for i in range(1, n_points):
            x[i] = x[i-1] + a * r**(i-1)
        
        # Ajustement pour correspondre exactement à x_end
        x[-1] = x_end
        
        return x
    
    def _create_element_connectivity(self):
        """Crée la connectivité des éléments"""
        self.num_elements = self.params.nx * self.params.ny
        self.elements = np.zeros((self.num_elements, 4), dtype=int)
        
        for ey in range(self.params.ny):
            for ex in range(self.params.nx):
                e = ey * self.params.nx + ex
                n0 = ey * self.num_nodes_x + ex
                n1 = n0 + 1
                n2 = n0 + self.num_nodes_x + 1
                n3 = n0 + self.num_nodes_x
                self.elements[e, :] = [n0, n1, n2, n3]
    
    def _assign_material_ids(self):
        """Assigne les IDs de matériaux (0=substrat, 1=glace)"""
        self.material_id = np.zeros(self.num_elements, dtype=int)
        
        for e in range(self.num_elements):
            element_nodes = self.elements[e]
            y_center = np.mean(self.nodes[element_nodes, 1])
            if y_center > self.params.interface_y:
                self.material_id[e] = 1  # Glace
    
    def _create_interface_nodes(self):
        """Crée les nœuds dupliqués à l'interface (uniquement si czm_mesh=True)"""
        if not self.params.czm_mesh:
            return
            
        # Trouve les nœuds du substrat à l'interface
        for i in range(self.num_nodes):
            if abs(self.nodes[i, 1] - self.params.interface_y) < self.ZERO_TOL:
                self.substrate_interface_nodes.append(i)
        
        # Crée des nœuds dupliqués pour le côté glace
        for node in self.substrate_interface_nodes:
            new_node_id = len(self.nodes)
            new_node_coords = self.nodes[node].copy()
            self.nodes = np.vstack((self.nodes, new_node_coords))
            self.ice_interface_nodes.append(new_node_id)
        
        # Met à jour le nombre total de nœuds
        self.num_nodes = len(self.nodes)
    
    def _update_ice_connectivity(self):
        """Met à jour la connectivité des éléments de glace pour utiliser les nœuds dupliqués"""
        if not self.params.czm_mesh:
            return
            
        for e in range(self.num_elements):
            if self.material_id[e] == 1:  # Élément de glace
                element_nodes = self.elements[e].copy()
                for i in range(4):
                    node = element_nodes[i]
                    if node in self.substrate_interface_nodes:
                        idx = self.substrate_interface_nodes.index(node)
                        self.elements[e, i] = self.ice_interface_nodes[idx]
    
    def _create_cohesive_elements(self):
        """Crée les éléments cohésifs à épaisseur nulle à l'interface"""
        if not self.params.czm_mesh:
            return
            
        self.cohesive_elements = []
        
        # Obtenir les points et poids d'intégration selon la méthode choisie
        gauss_points, gauss_weights = get_cohesive_integration_points(
            method=self.params.cohesive_integration,
            n_points=self.params.cohesive_integration_points
        )
        
        # IMPORTANT: Ordre des nœuds cohérent
        # Pour chaque élément cohésif, l'ordre sera TOUJOURS :
        # [sub_node1, sub_node2, ice_node1, ice_node2]
        # où 1 = gauche, 2 = droite dans la direction x
        
        for i in range(len(self.substrate_interface_nodes) - 1):
            # Nœuds du substrat (ordre: gauche vers droite)
            sub_node1 = self.substrate_interface_nodes[i]      # gauche
            sub_node2 = self.substrate_interface_nodes[i+1]    # droite
            
            # Nœuds de la glace (ordre: gauche vers droite)
            ice_node1 = self.ice_interface_nodes[i]            # gauche
            ice_node2 = self.ice_interface_nodes[i+1]          # droite
            
            # Vérifier l'ordre (sub1 doit être à gauche de sub2)
            if self.nodes[sub_node1, 0] > self.nodes[sub_node2, 0]:
                sub_node1, sub_node2 = sub_node2, sub_node1
                ice_node1, ice_node2 = ice_node2, ice_node1
            
            # Calculer la longueur réelle de l'élément
            elem_length = self.nodes[sub_node2, 0] - self.nodes[sub_node1, 0]
            
            # ORDRE DÉFINITIF : [sub_gauche, sub_droite, ice_gauche, ice_droite]
            cohesive_element = CohesiveElement(
                nodes=[sub_node1, sub_node2, ice_node1, ice_node2],
                length=elem_length,
                gauss_points=gauss_points.copy(),
                gauss_weights=gauss_weights.copy()
            )
            
            self.cohesive_elements.append(cohesive_element)
        
        print(f"Créé {len(self.cohesive_elements)} éléments cohésifs avec {len(gauss_points)} points d'intégration")
        print(f"Ordre des nœuds: [sub_gauche, sub_droite, ice_gauche, ice_droite]")
    
    def _apply_boundary_conditions(self):
        """Applique les conditions aux limites"""
        self.fixed_nodes_x = []
        self.fixed_nodes_y = []
        self.fully_fixed_nodes = []
        
        for i in range(self.num_nodes):
            node_x = self.nodes[i, 0]
            node_y = self.nodes[i, 1]
            
            if node_y < self.params.interface_y:  # Nœud du substrat
                if node_x < self.ZERO_TOL:  # Bord gauche
                    if self.params.bc_type == 'left_edge_only':
                        self.fully_fixed_nodes.append(i)
                    else:  # 'original' BC
                        self.fixed_nodes_x.append(i)
                        if abs(node_y) < self.ZERO_TOL:
                            self.fully_fixed_nodes.append(i)
                            if i in self.fixed_nodes_x:
                                self.fixed_nodes_x.remove(i)
                
                elif (self.params.bc_type != 'left_edge_only' and 
                      abs(node_y) < self.ZERO_TOL and 
                      node_x <= 50.0):
                    self.fixed_nodes_y.append(i)
    
    def _create_dof_mappings(self):
        """Crée les mappings des degrés de liberté"""
        self.num_dofs_u = 2 * self.num_nodes  # 2 DOFs par nœud pour le déplacement
        self.num_dofs_d = self.num_nodes      # 1 DOF par nœud pour l'endommagement
        
        self.dof_map_u = np.arange(self.num_dofs_u, dtype=int).reshape(self.num_nodes, 2)
        self.dof_map_d = np.arange(self.num_dofs_d, dtype=int)
        
        # Configuration pour la visualisation
        self.viz_nodes_x = self.num_nodes_x
        self.viz_nodes_y = self.num_nodes_y
    
    def get_element_size(self, element_id: int) -> float:
        """Retourne la taille d'un élément"""
        element_nodes = self.elements[element_id]
        
        # Calculer la taille réelle de l'élément
        x_coords = self.nodes[element_nodes, 0]
        y_coords = self.nodes[element_nodes, 1]
        
        dx = np.max(x_coords) - np.min(x_coords)
        dy = np.max(y_coords) - np.min(y_coords)
        
        return dx * dy
    
    def get_boundary_conditions(self) -> Dict[str, List[int]]:
        """Retourne les conditions aux limites"""
        return {
            'fixed_x': self.fixed_nodes_x,
            'fixed_y': self.fixed_nodes_y,
            'fully_fixed': self.fully_fixed_nodes
        }
    
    def get_interface_info(self) -> Dict[str, any]:
        """Retourne les informations sur l'interface"""
        return {
            'y_position': self.params.interface_y,
            'substrate_nodes': self.substrate_interface_nodes,
            'ice_nodes': self.ice_interface_nodes,
            'cohesive_elements': self.cohesive_elements,
            'czm_active': self.params.czm_mesh
        }