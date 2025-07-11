"""
Module de fonctions utilitaires pour le modèle PF-CZM
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import os
import json
from datetime import datetime


@dataclass
class SimulationParameters:
    """Paramètres de simulation centralisés"""
    # Temps
    dt_initial: float = 1.0e-2
    dt_min: float = 1.0e-10
    dt_max: float = 1.0e-2
    total_time: float = 1.0
    ramp_time: float = 1.0
    
    # Chargement
    omega: float = 830.1135  # Vitesse angulaire (rad/s)
    
    # Solveur
    max_newton_iter: int = 5
    newton_tol: float = 1.0e-4
    max_staggered_iter: int = 10
    staggered_tol: float = 1.0e-2
    
    # HHT-alpha
    alpha_HHT: float = 0.05
    
    # Seuils d'endommagement
    damage_threshold: float = 0.90
    interface_damage_threshold: float = 0.90
    
    def __post_init__(self):
        """Calcul des paramètres dérivés"""
        self.beta = (1.0 + self.alpha_HHT)**2 / 4.0
        self.gamma = 0.5 + self.alpha_HHT


class DamageChecker:
    """Vérificateur de seuils d'endommagement"""
    
    def __init__(self, damage_threshold: float = 0.90, 
                 interface_damage_threshold: float = 0.90):
        self.damage_threshold = damage_threshold
        self.interface_damage_threshold = interface_damage_threshold
    
    def check_threshold_exceeded(self, prev_d: np.ndarray, current_d: np.ndarray,
                               prev_interface_damage: List[np.ndarray],
                               current_interface_damage: List[np.ndarray]) -> Tuple[bool, Dict]:
        """
        Vérifie si l'augmentation d'endommagement dépasse les seuils
        
        Returns:
            exceeded: True si le seuil est dépassé
            info: Dictionnaire avec les détails de l'excès
        """
        info = {
            'bulk_exceeded': False,
            'interface_exceeded': False,
            'max_bulk_increase': 0.0,
            'max_interface_increase': 0.0,
            'bulk_location': None,
            'interface_location': None
        }
        
        # Vérifier les nœuds du volume
        for i in range(len(current_d)):
            damage_increase = current_d[i] - prev_d[i]
            
            if damage_increase > info['max_bulk_increase']:
                info['max_bulk_increase'] = damage_increase
                info['bulk_location'] = i
            
            if damage_increase > self.damage_threshold:
                info['bulk_exceeded'] = True
                print(f"  Seuil d'endommagement dépassé au nœud {i}: "
                      f"{damage_increase:.4f} > {self.damage_threshold:.4f}")
                print(f"  Précédent: {prev_d[i]:.4f}, Actuel: {current_d[i]:.4f}")
        
        # Vérifier les points d'intégration de l'interface si CZM actif
        if current_interface_damage:
            for e in range(len(current_interface_damage)):
                for i in range(len(current_interface_damage[e])):
                    damage_increase = current_interface_damage[e][i] - prev_interface_damage[e][i]
                    
                    if damage_increase > info['max_interface_increase']:
                        info['max_interface_increase'] = damage_increase
                        info['interface_location'] = (e, i)
                    
                    if damage_increase > self.interface_damage_threshold:
                        info['interface_exceeded'] = True
                        print(f"  Seuil d'endommagement d'interface dépassé à l'élément {e}, "
                              f"point de Gauss {i}: {damage_increase:.4f} > "
                              f"{self.interface_damage_threshold:.4f}")
                        print(f"  Précédent: {prev_interface_damage[e][i]:.4f}, "
                              f"Actuel: {current_interface_damage[e][i]:.4f}")
        
        exceeded = info['bulk_exceeded'] or info['interface_exceeded']
        return exceeded, info


class BoundaryConditionApplier:
    """Applique les conditions aux limites aux matrices et vecteurs"""
    
    @staticmethod
    def apply_to_matrix(K, M, bc_dict: Dict[str, List[int]], dof_map_u: np.ndarray):
        """
        Applique les conditions aux limites aux matrices de rigidité et de masse
        
        Parameters:
            K: Matrice de rigidité (sparse)
            M: Matrice de masse (sparse)
            bc_dict: Dictionnaire des conditions aux limites
            dof_map_u: Mapping des degrés de liberté
        """
        # Nœuds complètement fixés
        for node in bc_dict.get('fully_fixed', []):
            for dof in range(2):
                dof_idx = dof_map_u[node, dof]
                K[dof_idx, :] = 0.0
                K[:, dof_idx] = 0.0
                K[dof_idx, dof_idx] = 1.0
                
                M[dof_idx, :] = 0.0
                M[:, dof_idx] = 0.0
                M[dof_idx, dof_idx] = 1.0
        
        # Nœuds fixés seulement en x
        for node in bc_dict.get('fixed_x', []):
            dof_idx = dof_map_u[node, 0]
            K[dof_idx, :] = 0.0
            K[:, dof_idx] = 0.0
            K[dof_idx, dof_idx] = 1.0
            
            M[dof_idx, :] = 0.0
            M[:, dof_idx] = 0.0
            M[dof_idx, dof_idx] = 1.0
        
        # Nœuds fixés seulement en y
        for node in bc_dict.get('fixed_y', []):
            dof_idx = dof_map_u[node, 1]
            K[dof_idx, :] = 0.0
            K[:, dof_idx] = 0.0
            K[dof_idx, dof_idx] = 1.0
            
            M[dof_idx, :] = 0.0
            M[:, dof_idx] = 0.0
            M[dof_idx, dof_idx] = 1.0
    
    @staticmethod
    def apply_to_vector(f: np.ndarray, bc_dict: Dict[str, List[int]], 
                       dof_map_u: np.ndarray):
        """Applique les conditions aux limites à un vecteur de forces"""
        # Nœuds complètement fixés
        for node in bc_dict.get('fully_fixed', []):
            for dof in range(2):
                dof_idx = dof_map_u[node, dof]
                f[dof_idx] = 0.0
        
        # Nœuds fixés seulement en x
        for node in bc_dict.get('fixed_x', []):
            dof_idx = dof_map_u[node, 0]
            f[dof_idx] = 0.0
        
        # Nœuds fixés seulement en y
        for node in bc_dict.get('fixed_y', []):
            dof_idx = dof_map_u[node, 1]
            f[dof_idx] = 0.0


class DataLogger:
    """Enregistreur de données de simulation"""
    
    def __init__(self, output_dir: str = 'results'):
        self.output_dir = output_dir
        self.data = {
            'time': [],
            'max_damage': [],
            'max_interface_damage': [],
            'energies': {
                'strain': [],
                'fracture': [],
                'interface': [],
                'kinetic': [],
                'total': []
            },
            'convergence': []
        }
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Fichier de log
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(output_dir, f'simulation_log_{timestamp}.txt')
    
    def log_step(self, step: int, time: float, damages: Dict, 
                 energies: Dict, convergence_info: Dict):
        """Enregistre les données d'un pas de temps"""
        self.data['time'].append(time)
        self.data['max_damage'].append(damages['max_bulk'])
        self.data['max_interface_damage'].append(damages['max_interface'])
        
        for key in ['strain', 'fracture', 'interface', 'kinetic', 'total']:
            self.data['energies'][key].append(energies.get(key, 0.0))
        
        self.data['convergence'].append(convergence_info)
        
        # Écrire dans le fichier de log
        with open(self.log_file, 'a') as f:
            f.write(f"\nStep {step}: Time = {time:.6f}\n")
            f.write(f"  Max bulk damage: {damages['max_bulk']:.6f}\n")
            f.write(f"  Max interface damage: {damages['max_interface']:.6f}\n")
            f.write(f"  Total energy: {energies['total']:.6f}\n")
            if convergence_info:
                f.write(f"  Convergence: {convergence_info}\n")
    
    def save_data(self):
        """Sauvegarde toutes les données dans un fichier JSON"""
        output_file = os.path.join(self.output_dir, 'simulation_data.json')
        with open(output_file, 'w') as f:
            json.dump(self.data, f, indent=2)
        print(f"Données sauvegardées dans {output_file}")
    
    def save_arrays(self, arrays_dict: Dict[str, np.ndarray]):
        """Sauvegarde des arrays NumPy"""
        for name, array in arrays_dict.items():
            np.save(os.path.join(self.output_dir, f'{name}.npy'), array)


class ProgressTracker:
    """Suivi de la progression de la simulation"""
    
    def __init__(self, total_time: float):
        self.total_time = total_time
        self.start_time = datetime.now()
        self.last_update = self.start_time
    
    def update(self, current_time: float, step: int):
        """Met à jour et affiche la progression"""
        progress = current_time / self.total_time * 100
        elapsed = (datetime.now() - self.start_time).total_seconds()
        
        if current_time > 0:
            estimated_total = elapsed / (current_time / self.total_time)
            remaining = estimated_total - elapsed
            
            # Afficher seulement toutes les 10 secondes ou à 10% d'intervalle
            now = datetime.now()
            if (now - self.last_update).total_seconds() > 10 or progress % 10 < 0.1:
                print(f"\n{'='*50}")
                print(f"Progression: {progress:.1f}% (Pas {step})")
                print(f"Temps simulé: {current_time:.4f} / {self.total_time:.4f}")
                print(f"Temps écoulé: {self._format_time(elapsed)}")
                print(f"Temps restant estimé: {self._format_time(remaining)}")
                print(f"{'='*50}\n")
                self.last_update = now
    
    def _format_time(self, seconds: float) -> str:
        """Formate le temps en heures:minutes:secondes"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class InterfaceDebugger:
    """Outils de débogage pour l'interface"""
    
    @staticmethod
    def print_interface_state(mesh, cohesive_manager, u: np.ndarray, 
                            location: str = "left_edge"):
        """Affiche l'état de l'interface à un endroit spécifique"""
        if not mesh.params.czm_mesh:
            print("Interface debugging: CZM désactivé")
            return
            
        print(f"\n{'='*60}")
        print(f"État de l'interface au {location}")
        print(f"{'='*60}")
        
        if location == "left_edge":
            # Trouver les éléments cohésifs au bord gauche
            for i, elem in enumerate(mesh.cohesive_elements):
                node = elem.nodes[0]  # Premier nœud du substrat
                if mesh.nodes[node, 0] < 1e-6:  # x ≈ 0
                    InterfaceDebugger._print_cohesive_element_state(
                        i, elem, mesh, cohesive_manager, u
                    )
                    break
    
    @staticmethod
    def _print_cohesive_element_state(elem_idx: int, elem, mesh, 
                                     cohesive_manager, u: np.ndarray):
        """Affiche l'état détaillé d'un élément cohésif"""
        # Nœuds
        sub_node1, sub_node2, ice_node1, ice_node2 = elem.nodes
        
        # Déplacements
        u_sub1 = np.array([
            u[mesh.dof_map_u[sub_node1, 0]],
            u[mesh.dof_map_u[sub_node1, 1]]
        ])
        u_ice1 = np.array([
            u[mesh.dof_map_u[ice_node1, 0]],
            u[mesh.dof_map_u[ice_node1, 1]]
        ])
        
        # Sauts
        delta = u_ice1 - u_sub1
        
        # Endommagement
        damage = elem.damage[0]  # Premier point de Gauss
        
        print(f"\nÉlément cohésif {elem_idx}:")
        print(f"  Nœud substrat {sub_node1}: u = [{u_sub1[0]:.6e}, {u_sub1[1]:.6e}]")
        print(f"  Nœud glace {ice_node1}: u = [{u_ice1[0]:.6e}, {u_ice1[1]:.6e}]")
        print(f"  Saut de déplacement: Δu = [{delta[0]:.6e}, {delta[1]:.6e}]")
        print(f"  Endommagement: {damage:.6f}")
        
        # Calculer les tractions
        delta_n, delta_t, T = cohesive_manager._compute_local_kinematics(
            elem.nodes, -1.0, u  # xi = -1 pour le bord gauche
        )
        traction = cohesive_manager._compute_tractions(delta_n, delta_t, damage)
        
        print(f"  Saut normal: {delta_n:.6e} mm")
        print(f"  Saut tangentiel: {delta_t:.6e} mm")
        print(f"  Traction normale: {traction.normal:.6e} MPa")
        print(f"  Traction tangentielle: {traction.tangential:.6e} MPa")
        
        # Phase de la loi cohésive
        if delta_n >= 0:
            if delta_n <= cohesive_manager.cohesive_props.normal_delta0:
                phase_n = "Élastique"
            elif delta_n <= cohesive_manager.cohesive_props.normal_deltac:
                phase_n = "Adoucissement"
            else:
                phase_n = "Rupture"
        else:
            phase_n = "Compression"
        
        if abs(delta_t) <= cohesive_manager.cohesive_props.shear_delta0:
            phase_t = "Élastique"
        elif abs(delta_t) <= cohesive_manager.cohesive_props.shear_deltac:
            phase_t = "Adoucissement"
        else:
            phase_t = "Rupture"
        
        print(f"  Phase normale: {phase_n}")
        print(f"  Phase tangentielle: {phase_t}")


def create_results_directory(base_dir: str = 'results') -> str:
    """Crée un répertoire de résultats avec horodatage"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = os.path.join(base_dir, f'simulation_{timestamp}')
    os.makedirs(results_dir, exist_ok=True)
    
    # Créer les sous-répertoires
    for subdir in ['plots', 'data', 'logs', 'checkpoints']:
        os.makedirs(os.path.join(results_dir, subdir), exist_ok=True)
    
    return results_dir


def save_simulation_parameters(params: dict, output_dir: str):
    """Sauvegarde les paramètres de simulation"""
    params_file = os.path.join(output_dir, 'parameters.json')
    
    # Convertir les arrays numpy en listes pour JSON
    params_serializable = {}
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            params_serializable[key] = value.tolist()
        elif isinstance(value, (np.float64, np.int64)):
            params_serializable[key] = float(value) if isinstance(value, np.float64) else int(value)
        elif isinstance(value, dict):
            # Récursion pour les dictionnaires imbriqués
            params_serializable[key] = {}
            for k, v in value.items():
                if isinstance(v, np.ndarray):
                    params_serializable[key][k] = v.tolist()
                elif isinstance(v, (np.float64, np.int64)):
                    params_serializable[key][k] = float(v) if isinstance(v, np.float64) else int(v)
                else:
                    params_serializable[key][k] = v
        else:
            params_serializable[key] = value
    
    with open(params_file, 'w') as f:
        json.dump(params_serializable, f, indent=2)
    
    print(f"Paramètres sauvegardés dans {params_file}")