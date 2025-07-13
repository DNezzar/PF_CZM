"""
Module des solveurs pour le modèle PF-CZM
Inclut les solveurs HHT-alpha, Newton-Raphson et le schéma alterné
"""

import numpy as np
from scipy.sparse.linalg import spsolve
from typing import Dict, Tuple, Optional, Callable
from dataclasses import dataclass
import time as timer


@dataclass
class SolverParameters:
    """Paramètres des solveurs"""
    # HHT-alpha
    alpha_HHT: float = 0.05
    
    # Newton-Raphson
    max_newton_iter: int = 5
    newton_tol: float = 1.0e-4
    
    # Schéma alterné
    max_staggered_iter: int = 10
    staggered_tol: float = 1.0e-2
    keep_previous_staggered: bool = True  # Option pour garder la solution précédente convergée
    
    # Pas de temps adaptatif
    dt_initial: float = 1.0e-2
    dt_min: float = 1.0e-10
    dt_max: float = 1.0e-2
    
    # Facteurs d'adaptation du pas de temps (tous modifiables)
    dt_increase_factor: float = 1.1          # Augmentation normale
    dt_increase_fast: float = 1.2            # Augmentation rapide (convergence très rapide)
    dt_decrease_factor: float = 0.5          # Réduction normale
    dt_decrease_slow: float = 0.7            # Réduction légère (convergence lente)
    
    # Seuils pour l'adaptation
    staggered_iter_fast: int = 2             # Convergence rapide si <= ce seuil
    staggered_iter_slow: int = 8             # Convergence lente si >= ce seuil
    
    # Seuils d'endommagement
    damage_threshold: float = 0.90
    interface_damage_threshold: float = 0.90
    
    def __post_init__(self):
        """Calcul des paramètres dérivés pour HHT-alpha"""
        self.beta = (1.0 + self.alpha_HHT)**2 / 4.0
        self.gamma = 0.5 + self.alpha_HHT


@dataclass
class ConvergenceInfo:
    """Informations de convergence"""
    converged: bool
    iterations: int
    residual_norm: float
    relative_residual: float
    reason: str = ""
    
    def __str__(self):
        status = "Convergé" if self.converged else "Non convergé"
        return (f"{status} en {self.iterations} itérations, "
                f"résidu: {self.residual_norm:.3e}, "
                f"résidu relatif: {self.relative_residual:.3e}")


class HHTAlphaSolver:
    """Solveur temporel utilisant le schéma HHT-alpha"""
    
    def __init__(self, system_assembler, params: SolverParameters):
        self.assembler = system_assembler
        self.params = params
        
        # Statistiques
        self.total_newton_iterations = 0
        self.total_linear_solves = 0
    
    def solve_time_step(self, u_prev: np.ndarray, v_prev: np.ndarray, a_prev: np.ndarray,
                       d: np.ndarray, time: float, dt: float,
                       loading_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, ConvergenceInfo]:
        """
        Résout un pas de temps avec le schéma HHT-alpha

        Returns:
            u: Déplacements au temps n+1
            v: Vitesses au temps n+1
            a: Accélérations au temps n+1
            info: Informations de convergence
        """
        # Prédicteurs (Newmark)
        u_pred = u_prev + dt * v_prev + 0.5 * dt**2 * ((1.0 - 2.0 * self.params.beta) * a_prev)
        v_pred = v_prev + dt * ((1.0 - self.params.gamma) * a_prev)

        # Initialisation
        u = u_pred.copy()
        v = v_pred.copy()
        a = np.zeros_like(a_prev)

        # Paramètres pour le calcul du résidu
        residual_params = {
            'alpha_HHT': self.params.alpha_HHT,
            'loading_params': loading_params,
            'use_decomposition': self.assembler.materials.use_decomposition  # CORRECTION
        }

        # Adapter la tolérance selon dt (pour petits pas de temps)
        adaptive_tol = self.params.newton_tol
        #if dt < 1e-4:
        #    adaptive_tol = min(1e-2, self.params.newton_tol * 100)
        #    print(f"    Tolérance adaptée: {adaptive_tol:.3e} (dt petit)")

        # Itérations de Newton-Raphson
        for newton_iter in range(self.params.max_newton_iter):
            # Calculer l'accélération avec Newmark
            a = self._compute_acceleration(u, u_prev, v_prev, a_prev, dt)

            # Assembler les matrices
            #K_curr, M, f_ext_curr = self.assembler.assemble_system(
            #    u, d, time, loading_params, 
            #    use_decomposition=self.assembler.materials.use_decomposition  
            #)
            # Assembler les matrices avec u_prev # CORRECTION
            K_curr, M, f_ext_curr = self.assembler.assemble_system(
                u, u_prev, d, time, loading_params, 
                use_decomposition=self.assembler.materials.use_decomposition
            )

            # Calculer le résidu
            residual = self._compute_residual(
                u, v, a, u_prev, v_prev, a_prev, d, 
                K_curr, M, f_ext_curr, time, dt, residual_params
            )

            # Vérifier la convergence
            residual_norm = np.linalg.norm(residual)
            u_norm = np.linalg.norm(u) + 1e-10
            relative_residual = residual_norm / u_norm

            print(f"    Newton iter {newton_iter+1}: "
                  f"résidu = {residual_norm:.6e}, "
                  f"résidu relatif = {relative_residual:.6e}")

            if relative_residual < adaptive_tol:
                # Mettre à jour la vitesse
                v = v_prev + dt * ((1.0 - self.params.gamma) * a_prev + self.params.gamma * a)

                info = ConvergenceInfo(
                    converged=True,
                    iterations=newton_iter+1,
                    residual_norm=residual_norm,
                    relative_residual=relative_residual,
                    reason="Tolérance atteinte"
                )

                self.total_newton_iterations += newton_iter + 1
                return u, v, a, info

            # Calculer la matrice effective
            K_eff = self._compute_effective_stiffness(K_curr, M, dt)

            # Appliquer les conditions aux limites
            bc_dict = self.assembler.mesh.get_boundary_conditions()
            self._apply_bc_to_system(K_eff, residual, bc_dict)

            # Résoudre le système linéaire
            try:
                du = spsolve(K_eff, -residual)
                self.total_linear_solves += 1
            except Exception as e:
                print(f"    Erreur dans la résolution linéaire: {e}")
                break
            
            # Recherche linéaire pour stabiliser
            alpha = 1.0
            u_trial = u + alpha * du

            # Si le résidu augmente trop, réduire alpha
            for _ in range(5):
                a_trial = self._compute_acceleration(u_trial, u_prev, v_prev, a_prev, dt)
                residual_trial = self._compute_residual(
                    u_trial, v, a_trial, u_prev, v_prev, a_prev, d,
                    K_curr, M, f_ext_curr, time, dt, residual_params
                )

                if np.linalg.norm(residual_trial) < residual_norm * 1.1:
                    break
                
                alpha *= 0.5
                u_trial = u + alpha * du

                if alpha < 0.1:
                    break
                
            # Mettre à jour le déplacement
            u = u_trial

        # Non convergé
        info = ConvergenceInfo(
            converged=False,
            iterations=self.params.max_newton_iter,
            residual_norm=residual_norm,
            relative_residual=relative_residual,
            reason="Nombre maximal d'itérations atteint"
        )

        return u, v, a, info
    
    def _compute_acceleration(self, u: np.ndarray, u_prev: np.ndarray,
                            v_prev: np.ndarray, a_prev: np.ndarray, dt: float) -> np.ndarray:
        """Calcule l'accélération avec la formule de Newmark"""
        return ((u - u_prev - dt * v_prev) / (self.params.beta * dt**2) - 
                ((1.0 - 2.0 * self.params.beta) / (2.0 * self.params.beta)) * a_prev)
    
    def _compute_residual(self, u: np.ndarray, v: np.ndarray, a: np.ndarray,
                        u_prev: np.ndarray, v_prev: np.ndarray, a_prev: np.ndarray,
                        d: np.ndarray, K_curr, M, f_ext_curr,
                        time: float, dt: float, params: Dict) -> np.ndarray:
        """Calcule le résidu pour le schéma HHT-alpha"""
        alpha = params['alpha_HHT']

        # Forces internes au temps n+1
        f_int_curr = K_curr @ u

        # Forces au temps précédent si nécessaire
        if abs(alpha) > 1e-10:
            # Recalculer les forces externes au temps précédent
            K_prev, _, f_ext_prev = self.assembler.assemble_system(
                u_prev, u_prev, d, time - dt, params['loading_params'], 
                use_decomposition=params.get('use_decomposition', False)
            )
            # Forces internes au temps précédent
            f_int_prev = K_prev @ u_prev
        else:
            f_int_prev = np.zeros_like(f_int_curr)
            f_ext_prev = np.zeros_like(f_ext_curr)

        # Résidu HHT-alpha
        residual = (M @ a + 
                   (1.0 + alpha) * f_int_curr - 
                   alpha * f_int_prev - 
                   (1.0 + alpha) * f_ext_curr + 
                   alpha * f_ext_prev)

        return residual
    
    def _compute_effective_stiffness(self, K, M, dt: float):
        """Calcule la matrice de rigidité effective pour Newton"""
        return (1.0 + self.params.alpha_HHT) * K + (1.0 / (self.params.beta * dt**2)) * M
    
    def _apply_bc_to_system(self, K_eff, residual: np.ndarray, bc_dict: Dict):
        """Applique les conditions aux limites au système"""
        # Nœuds complètement fixés
        for node in bc_dict.get('fully_fixed', []):
            for dof in range(2):
                dof_idx = self.assembler.mesh.dof_map_u[node, dof]
                K_eff[dof_idx, :] = 0.0
                K_eff[:, dof_idx] = 0.0
                K_eff[dof_idx, dof_idx] = 1.0
                residual[dof_idx] = 0.0
        
        # Nœuds fixés en x
        for node in bc_dict.get('fixed_x', []):
            dof_idx = self.assembler.mesh.dof_map_u[node, 0]
            K_eff[dof_idx, :] = 0.0
            K_eff[:, dof_idx] = 0.0
            K_eff[dof_idx, dof_idx] = 1.0
            residual[dof_idx] = 0.0
        
        # Nœuds fixés en y
        for node in bc_dict.get('fixed_y', []):
            dof_idx = self.assembler.mesh.dof_map_u[node, 1]
            K_eff[dof_idx, :] = 0.0
            K_eff[:, dof_idx] = 0.0
            K_eff[dof_idx, dof_idx] = 1.0
            residual[dof_idx] = 0.0


class StaggeredSolver:
    """Solveur utilisant un schéma alterné pour le couplage mécanique-endommagement"""
    
    def __init__(self, mechanical_solver: HHTAlphaSolver, phase_field_solver,
                 cohesive_manager, params: SolverParameters):
        self.mech_solver = mechanical_solver
        self.pf_solver = phase_field_solver
        self.cohesive = cohesive_manager
        self.params = params
        
        # Statistiques
        self.total_staggered_iterations = 0
        
        # État de la dernière itération convergée
        self.last_converged_state = None
    
    def solve_coupled_step(self, u_prev: np.ndarray, v_prev: np.ndarray, a_prev: np.ndarray,
                         d_prev: np.ndarray, time: float, dt: float,
                         loading_params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
        """
        Résout un pas de temps couplé avec le schéma alterné
        
        Returns:
            u, v, a, d: Solutions mises à jour
            info: Informations de convergence et statistiques
        """
        # Initialisation
        u = u_prev.copy()
        v = v_prev.copy()
        a = a_prev.copy()
        d = d_prev.copy()
        
        # Sauvegarder l'état des éléments cohésifs si CZM actif
        interface_damage_prev = []
        if self.cohesive:
            for elem in self.cohesive.mesh.cohesive_elements:
                interface_damage_prev.append(elem.damage.copy())
        
        info = {
            'staggered_converged': False,
            'staggered_iterations': 0,
            'mechanical_converged': True,
            'damage_evolution': {},
            'final_residual': float('inf'),
            'last_converged_iteration': 0
        }
        
        # Réinitialiser l'état de la dernière convergence
        self.last_converged_state = None
        
        # Itérations du schéma alterné
        for stag_iter in range(self.params.max_staggered_iter):
            print(f"  Itération alterné {stag_iter+1}/{self.params.max_staggered_iter}")
            
            # Sauvegarder pour la convergence
            d_old = d.copy()
            
            # 1. Résoudre le problème mécanique
            print("    Résolution du problème mécanique...")
            u, v, a, mech_info = self.mech_solver.solve_time_step(
                u_prev, v_prev, a_prev, d, time, dt, loading_params
            )
            
            if not mech_info.converged:
                info['mechanical_converged'] = False
                info['final_residual'] = mech_info.residual_norm
                print("    Problème mécanique non convergé")
                break
            
            # 2. Mettre à jour l'endommagement de l'interface si CZM actif
            if self.cohesive:
                print("    Mise à jour de l'endommagement de l'interface...")
                self.cohesive.update_damage(u, dt)
            
            # 3. Résoudre le problème du champ de phase
            print("    Résolution du champ de phase...")
            d = self.pf_solver.solve(u, d_prev)
            
            # 4. Vérifier la convergence (UNIQUEMENT sur l'endommagement)
            d_diff = np.linalg.norm(d - d_old)
            d_norm = np.linalg.norm(d) + 1e-10
            #d_residual = d_diff / d_norm
            d_residual = np.max(np.abs(d - d_old))
            
            print(f"    Résidu endommagement: {d_residual:.6e}")
            
            info['final_residual'] = d_residual
            
            # Sauvegarder l'état si convergé (mécanique + phase field)
            if self.params.keep_previous_staggered:
                self.last_converged_state = {
                    'u': u.copy(),
                    'v': v.copy(),
                    'a': a.copy(),
                    'd': d.copy(),
                    'iteration': stag_iter + 1
                }
                if self.cohesive:
                    self.last_converged_state['interface_damage'] = []
                    for elem in self.cohesive.mesh.cohesive_elements:
                        self.last_converged_state['interface_damage'].append(elem.damage.copy())
                
                info['last_converged_iteration'] = stag_iter + 1
            
            if d_residual < self.params.staggered_tol:
                info['staggered_converged'] = True
                info['staggered_iterations'] = stag_iter + 1
                print("    Schéma alterné convergé")
                break
        
        # Si non convergé et keep_previous_staggered activé, restaurer le dernier état convergé
        if not info['staggered_converged'] and self.params.keep_previous_staggered and self.last_converged_state:
            print(f"    Restauration de l'état convergé (itération {self.last_converged_state['iteration']})")
            u = self.last_converged_state['u']
            v = self.last_converged_state['v']
            a = self.last_converged_state['a']
            d = self.last_converged_state['d']
            
            if self.cohesive and 'interface_damage' in self.last_converged_state:
                for i, elem in enumerate(self.cohesive.mesh.cohesive_elements):
                    elem.damage = self.last_converged_state['interface_damage'][i]
        
        # Statistiques finales
        self.total_staggered_iterations += info['staggered_iterations']
        
        # Informations sur l'évolution de l'endommagement
        info['damage_evolution'] = {
            'max_bulk_damage': np.max(d),
            'max_interface_damage': self.cohesive.get_max_interface_damage() if self.cohesive else 0.0,
            'bulk_damage_increase': np.max(d - d_prev),
            'num_damaged_nodes': np.sum(d > 0.01)
        }
        
        return u, v, a, d, info


class AdaptiveTimeStepping:
    """Gestion du pas de temps adaptatif"""
    
    def __init__(self, params: SolverParameters, damage_checker):
        self.params = params
        self.damage_checker = damage_checker
        self.dt = params.dt_initial
        
        # Historique
        self.dt_history = [self.dt]
        self.adaptation_history = []
    
    def compute_next_timestep(self, convergence_info: Dict, damage_info: Dict,
                            current_dt: float) -> Tuple[float, str]:
        """
        Calcule le pas de temps suivant basé sur la convergence et l'évolution de l'endommagement
        
        Returns:
            new_dt: Nouveau pas de temps
            reason: Raison du changement
        """
        # Vérifier si l'endommagement évolue trop rapidement
        damage_exceeded = damage_info.get('damage_exceeded', False)
        
        # Vérifier la convergence
        converged = convergence_info.get('staggered_converged', False)
        stag_iter = convergence_info.get('staggered_iterations', 0)
        
        # Logique d'adaptation simplifiée
        if not converged:
            # Non-convergence
            new_dt = current_dt * self.params.dt_decrease_factor
            reason = "Non-convergence"
        
        elif damage_exceeded:
            # Évolution rapide de l'endommagement
            new_dt = current_dt * self.params.dt_decrease_factor
            reason = "Évolution rapide de l'endommagement"
        
        elif stag_iter >= self.params.staggered_iter_slow:
            # Convergence lente
            new_dt = current_dt * self.params.dt_decrease_slow
            reason = f"Convergence lente ({stag_iter} itérations)"
        
        elif stag_iter <= self.params.staggered_iter_fast:
            # Convergence rapide
            new_dt = current_dt * self.params.dt_increase_fast
            reason = f"Convergence rapide ({stag_iter} itérations)"
        
        else:
            # Convergence normale
            new_dt = current_dt * self.params.dt_increase_factor
            reason = f"Convergence normale ({stag_iter} itérations)"
        
        # Appliquer les limites
        new_dt = np.clip(new_dt, self.params.dt_min, self.params.dt_max)
        
        # Enregistrer
        self.dt = new_dt
        self.dt_history.append(new_dt)
        self.adaptation_history.append({
            'old_dt': current_dt,
            'new_dt': new_dt,
            'reason': reason,
            'converged': converged,
            'iterations': stag_iter
        })
        
        return new_dt, reason
    
    def check_damage_evolution(self, d_prev: np.ndarray, d_curr: np.ndarray,
                             interface_damage_prev, interface_damage_curr) -> Dict:
        """
        Vérifie si l'évolution de l'endommagement nécessite une réduction du pas de temps
        """
        exceeded, info = self.damage_checker.check_threshold_exceeded(
            d_prev, d_curr, interface_damage_prev, interface_damage_curr
        )

        return {
            'damage_exceeded': exceeded,
            'max_bulk_increase': info['max_bulk_increase'],
            'max_interface_increase': info['max_interface_increase'],
            'info': info
        }
    
    def get_statistics(self) -> Dict:
        """Retourne des statistiques sur l'adaptation du pas de temps"""
        dt_array = np.array(self.dt_history)
        
        return {
            'min_dt': np.min(dt_array),
            'max_dt': np.max(dt_array),
            'mean_dt': np.mean(dt_array),
            'num_reductions': sum(1 for a in self.adaptation_history if 'réduction' in a['reason'].lower() or 'decrease' in a['reason'].lower()),
            'num_increases': sum(1 for a in self.adaptation_history if 'augmentation' in a['reason'].lower() or 'increase' in a['reason'].lower()),
            'total_adaptations': len(self.adaptation_history)
        }


class MainSolver:
    """Solveur principal orchestrant la simulation complète"""
    
    def __init__(self, model):
        self.model = model
        
        # Créer les sous-solveurs
        self.hht_solver = HHTAlphaSolver(
            model.system_assembler,
            model.solver_params
        )
        
        self.staggered_solver = StaggeredSolver(
            self.hht_solver,
            model.phase_field_solver,
            model.cohesive_manager,
            model.solver_params
        )
        
        self.time_stepper = AdaptiveTimeStepping(
            model.solver_params,
            model.damage_checker
        )
        
        # État de la simulation
        self.current_time = 0.0
        self.step = 0
        
    def solve(self, total_time: float, output_interval: int = 1,
             callback: Optional[Callable] = None) -> Dict:
        """
        Lance la simulation complète
        
        Parameters:
            total_time: Temps total de simulation
            output_interval: Intervalle de sortie (tous les N pas)
            callback: Fonction appelée après chaque pas convergé
            
        Returns:
            Dictionnaire avec les résultats et statistiques
        """
        print("Démarrage de la simulation...")
        print(f"Temps total: {total_time}, dt initial: {self.time_stepper.dt}")
        print(f"Option keep_previous_staggered: {self.model.solver_params.keep_previous_staggered}")
        print(f"Éléments cohésifs: {'Activés' if self.model.mesh.params.czm_mesh else 'Désactivés'}")
        
        # Initialisation
        results = {
            'success': False,
            'final_time': 0.0,
            'total_steps': 0,
            'convergence_history': [],
            'energy_history': []
        }
        
        start_time = timer.time()
        
        # Boucle temporelle principale
        while self.current_time < total_time:
            self.step += 1
            dt = self.time_stepper.dt
            
            print(f"\n{'='*60}")
            print(f"Pas {self.step}: Temps = {self.current_time:.6f}, dt = {dt:.6e}")
            print(f"{'='*60}")
            
            # Sauvegarder l'état précédent complet
            state_backup = {
                'u': self.model.u.copy(),
                'v': self.model.v.copy(),
                'a': self.model.a.copy(),
                'd': self.model.d.copy(),
                'time': self.current_time,
                'step': self.step,
                #'H_gauss': self.model.phase_field_solver.history.H_gauss.copy()
            }
            
            # Sauvegarder l'endommagement de l'interface si CZM actif
            if self.model.cohesive_manager:
                state_backup['interface_damage'] = []
                for elem in self.model.mesh.cohesive_elements:
                    state_backup['interface_damage'].append(elem.damage.copy())
            
            # Résoudre le pas de temps couplé
            u_new, v_new, a_new, d_new, conv_info = self.staggered_solver.solve_coupled_step(
                state_backup['u'], state_backup['v'], state_backup['a'], state_backup['d'],
                self.current_time + dt, dt,
                self.model.loading_params
            )
            
            # Vérifier la convergence
            step_converged = conv_info['staggered_converged'] and conv_info['mechanical_converged']
            
            if step_converged:
                # Vérifier l'évolution de l'endommagement
                interface_damage_curr = []
                if self.model.cohesive_manager:
                    for elem in self.model.mesh.cohesive_elements:
                        interface_damage_curr.append(elem.damage.copy())
                
                damage_info = self.time_stepper.check_damage_evolution(
                    state_backup['d'], d_new, 
                    state_backup.get('interface_damage', []), 
                    interface_damage_curr
                )
                
                if damage_info['damage_exceeded']:
                    print(f"  Seuil d'endommagement dépassé")
                    # Restaurer l'état précédent
                    self._restore_state(state_backup)
                    # Réduire dt et refaire le même pas
                    dt *= self.model.solver_params.dt_decrease_factor
                    self.time_stepper.dt = dt
                    print(f"  Réduction du pas de temps à {dt:.6e} et nouveau calcul du même pas")
                    continue
                
                # Pas convergé avec succès
                self.model.u = u_new
                self.model.v = v_new
                self.model.a = a_new
                self.model.d = d_new
                self.current_time += dt
                
                # Calculer les énergies
                energies = self.model.energy_calculator.calculate_all_energies(
                    self.model.u, self.model.v, self.model.d
                )
                
                # Enregistrer les résultats
                step_results = {
                    'step': self.step,
                    'time': self.current_time,
                    'dt': dt,
                    'convergence': conv_info,
                    'energies': energies.to_dict(),
                    'damage': {
                        'max_bulk': np.max(self.model.d),
                        'max_interface': self.model.cohesive_manager.get_max_interface_damage() if self.model.cohesive_manager else 0.0
                    }
                }
                
                results['convergence_history'].append(step_results)
                
                # Callback utilisateur
                if callback and (self.step % output_interval == 0):
                    callback(self.model, step_results)
                
                # Adapter le pas de temps pour la prochaine itération
                new_dt, reason = self.time_stepper.compute_next_timestep(
                    conv_info, damage_info, dt
                )
                
                if new_dt != dt:
                    print(f"  Adaptation du pas de temps: {dt:.6e} -> {new_dt:.6e} ({reason})")
                
                self.time_stepper.dt = new_dt
                
            else:
                # Non convergé
                print(f"\n⚠️ Pas non convergé")
                
                if self.model.solver_params.keep_previous_staggered and conv_info.get('last_converged_iteration', 0) > 0:
                    # Option activée : garder la dernière solution convergée et passer au pas suivant
                    print(f"  Conservation de la solution convergée (itération {conv_info['last_converged_iteration']})")
                    
                    # Mettre à jour l'état du modèle avec la solution partiellement convergée
                    self.model.u = u_new
                    self.model.v = v_new
                    self.model.a = a_new
                    self.model.d = d_new
                    self.current_time += dt
                    
                    # Calculer les énergies pour la solution partiellement convergée
                    energies = self.model.energy_calculator.calculate_all_energies(
                        self.model.u, self.model.v, self.model.d
                    )
                    
                    # Créer les résultats du pas même s'il n'est que partiellement convergé
                    step_results = {
                        'step': self.step,
                        'time': self.current_time,
                        'dt': dt,
                        'convergence': conv_info,
                        'partially_converged': True,
                        'energies': energies.to_dict(),
                        'damage': {
                            'max_bulk': np.max(self.model.d),
                            'max_interface': self.model.cohesive_manager.get_max_interface_damage() if self.model.cohesive_manager else 0.0
                        }
                    }
                    
                    results['convergence_history'].append(step_results)
                    
                    # Appeler le callback pour générer les plots
                    if callback and (self.step % output_interval == 0):
                        print(f"  Génération des plots pour la solution partiellement convergée...")
                        callback(self.model, step_results)
                    
                    # Adapter le pas de temps
                    new_dt = dt * self.model.solver_params.dt_decrease_factor
                    self.time_stepper.dt = new_dt
                    print(f"  Passage au pas suivant avec dt réduit: {new_dt:.6e}")
                    
                else:
                    # Option désactivée ou pas de solution convergée : refaire le même pas
                    print(f"  Restauration de l'état précédent")
                    self._restore_state(state_backup)
                    
                    # Réduire dt
                    new_dt = dt * self.model.solver_params.dt_decrease_factor
                    
                    if new_dt < self.model.solver_params.dt_min:
                        print(f"  ERREUR: Pas de temps minimal atteint")
                        results['success'] = False
                        results['final_time'] = self.current_time
                        results['total_steps'] = self.step
                        return results
                    
                    self.time_stepper.dt = new_dt
                    print(f"  Nouveau calcul du même pas avec dt = {new_dt:.6e}")
                    self.step -= 1  # Pour compenser l'incrémentation au début de la boucle
                    continue
                
            # Affichage du résumé
            print(f"\nRésumé du pas {self.step}:")
            print(f"  Temps: {self.current_time:.6f}")
            print(f"  Endommagement max (volume): {np.max(self.model.d):.6f}")
            if self.model.cohesive_manager:
                print(f"  Endommagement max (interface): {self.model.cohesive_manager.get_max_interface_damage():.6f}")
            
            # Vérifier si on a atteint le temps final
            if self.current_time >= total_time:
                print(f"\nSimulation terminée avec succès!")
                results['success'] = True
                break
            
        # Statistiques finales
        elapsed_time = timer.time() - start_time
        
        results['final_time'] = self.current_time
        results['total_steps'] = self.step
        results['computation_time'] = elapsed_time
        results['statistics'] = {
            'time_stepping': self.time_stepper.get_statistics(),
            'newton_iterations': self.hht_solver.total_newton_iterations,
            'staggered_iterations': self.staggered_solver.total_staggered_iterations,
            'linear_solves': self.hht_solver.total_linear_solves,
            'average_newton_per_step': self.hht_solver.total_newton_iterations / max(self.step, 1),
            'average_staggered_per_step': self.staggered_solver.total_staggered_iterations / max(self.step, 1)
        }
        
        print(f"\n{'='*60}")
        print("Statistiques de la simulation:")
        print(f"  Temps de calcul: {elapsed_time:.2f} s")
        print(f"  Nombre total de pas: {self.step}")
        print(f"  Itérations Newton totales: {results['statistics']['newton_iterations']}")
        print(f"  Itérations alternées totales: {results['statistics']['staggered_iterations']}")
        print(f"  Résolutions linéaires: {results['statistics']['linear_solves']}")
        print(f"{'='*60}")
        
        return results
    
    def _restore_state(self, state_backup: Dict):
        """Restaure l'état complet du modèle"""
        self.model.u = state_backup['u']
        self.model.v = state_backup['v']
        self.model.a = state_backup['a']
        self.model.d = state_backup['d']

        #if 'H_gauss' in state_backup:
            #self.model.phase_field_solver.history.H_gauss = state_backup['H_gauss'].copy()
        
        if self.model.cohesive_manager and 'interface_damage' in state_backup:
            for i, elem in enumerate(self.model.mesh.cohesive_elements):
                elem.damage = state_backup['interface_damage'][i]