"""
Classe principale du modèle PF-CZM intégrant tous les modules
"""

import numpy as np
import os
from typing import Dict, Optional, Callable

# Import des modules
from mesh import MeshManager, MeshParameters
from materials import MaterialManager, MaterialProperties, CohesiveProperties, SpectralDecomposition
from cohesive_zone import CohesiveZoneManager
from phase_field import PhaseFieldSolver, PhaseFieldParameters, HistoryField
from mechanics import ElementMatrices, SystemAssembler, LoadingParameters
from energies import EnergyCalculator, EnergyMonitor, EnergyComponents
from solver import MainSolver, SolverParameters
from visualization import PlotManager, PlotSettings, StressFieldCalculator
from utils import (SimulationParameters, DamageChecker, DataLogger, 
                   ProgressTracker, InterfaceDebugger, create_results_directory,
                   save_simulation_parameters)


class IceSubstratePhaseFieldFracture:
    """
    Modèle principal de rupture par champ de phase avec zone cohésive
    pour le système glace-substrat
    """
    
    def __init__(self, **kwargs):
        """
        Initialise le modèle avec tous les paramètres
        
        Les paramètres peuvent être passés individuellement ou via des dictionnaires
        """
        print("="*60)
        print("Initialisation du modèle PF-CZM Ice-Substrate")
        print("="*60)
        
        # 1. Créer le répertoire de résultats EN PREMIER
        self.results_dir = create_results_directory(
            kwargs.get('output_dir', 'results')
        )
        
        # 2. Extraire et organiser les paramètres (maintenant que results_dir existe)
        self._parse_parameters(kwargs)
        
        # 3. Initialiser les modules dans l'ordre
        self._initialize_modules()
        
        # 4. Initialiser les champs de solution
        self._initialize_solution_fields()
        
        # 5. Sauvegarder les paramètres
        save_simulation_parameters(self._get_all_parameters(), self.results_dir)
        
        print("\nModèle initialisé avec succès!")
        self._print_configuration_summary()
    
    def _parse_parameters(self, kwargs):
        """Parse et organise tous les paramètres d'entrée"""
        # Paramètres de maillage
        self.mesh_params = MeshParameters(
            length=kwargs.get('length', 170.0),
            ice_height=kwargs.get('ice_height', 6.4),
            substrate_height=kwargs.get('substrate_height', 6.4),
            nx=kwargs.get('nx', 250),
            ny_ice=kwargs.get('ny_ice', 10),
            ny_substrate=kwargs.get('ny_substrate', 5),
            bc_type=kwargs.get('bc_type', 'left_edge_only'),
            czm_mesh=kwargs.get('czm_mesh', True),
            cohesive_integration=kwargs.get('cohesive_integration', 'gauss-lobatto'),
            cohesive_integration_points=kwargs.get('cohesive_integration_points', 2),
            # Paramètres de maillage progressif
            use_coarse_near_bc=kwargs.get('use_coarse_near_bc', True),
            coarse_zone_length=kwargs.get('coarse_zone_length', 20.0),
            coarsening_ratio=kwargs.get('coarsening_ratio', 3.0),
            coarse_zone_reduction=kwargs.get('coarse_zone_reduction', 0.5)
        )

        # Propriétés des matériaux
        self.ice_props = MaterialProperties(
            E=kwargs.get('E_ice', 1500.0),
            nu=kwargs.get('nu_ice', 0.31),
            rho=kwargs.get('rho_ice', 0.917e-9),
            Gc=kwargs.get('Gc_ice', 0.001),
            name="Ice"
        )

        self.substrate_props = MaterialProperties(
            E=kwargs.get('E_sub', 69000.0),
            nu=kwargs.get('nu_sub', 0.325),
            rho=kwargs.get('rho_sub', 2.7e-9),
            Gc=kwargs.get('Gc_sub', 1.0e+8),
            name="Substrate"
        )

        # Propriétés cohésives
        self.cohesive_props = CohesiveProperties(
            normal_stiffness=kwargs.get('coh_normal_stiffness', 1.0e+5),
            normal_strength=kwargs.get('coh_normal_strength', 0.3),
            normal_Gc=kwargs.get('coh_normal_Gc', 0.00056),
            compression_factor=kwargs.get('coh_compression_factor', 50.0),
            shear_stiffness=kwargs.get('coh_shear_stiffness', 1.0e+5),
            shear_strength=kwargs.get('coh_shear_strength', 0.3),
            shear_Gc=kwargs.get('coh_shear_Gc', 0.00056),
            fixed_mixity=kwargs.get('fixed_mixity', 0.5)
        )

        # Paramètres du champ de phase
        self.phase_field_params = PhaseFieldParameters(
            l0=kwargs.get('l0', 1.0),
            k_res=kwargs.get('k_res', 1.0e-10),
            irreversibility=kwargs.get('irreversibility', True),
            threshold=kwargs.get('damage_threshold_min', 1e-6)
        )

        # Paramètres de chargement
        self.loading_params = LoadingParameters(
            omega=kwargs.get('omega', 830.1135),
            ramp_time=kwargs.get('ramp_time', 1.0),
            load_type=kwargs.get('load_type', 'centrifugal')
        )

        # Paramètres du solveur
        self.solver_params = SolverParameters(
            alpha_HHT=kwargs.get('alpha_HHT', 0.05),
            max_newton_iter=kwargs.get('max_newton_iter', 5),
            newton_tol=kwargs.get('newton_tol', 1.0e-4),
            max_staggered_iter=kwargs.get('max_staggered_iter', 10),
            staggered_tol=kwargs.get('staggered_tol', 1.0e-2),
            keep_previous_staggered=kwargs.get('keep_previous_staggered', True),
            dt_initial=kwargs.get('dt', 1.0e-2),
            dt_min=kwargs.get('dt_min', 1.0e-10),
            dt_max=kwargs.get('dt_max', 1.0e-2),
            dt_increase_factor=kwargs.get('dt_increase_factor', 1.1),
            dt_increase_fast=kwargs.get('dt_increase_fast', 1.2),
            dt_decrease_factor=kwargs.get('dt_decrease_factor', 0.5),
            dt_decrease_slow=kwargs.get('dt_decrease_slow', 0.7),
            staggered_iter_fast=kwargs.get('staggered_iter_fast', 2),
            staggered_iter_slow=kwargs.get('staggered_iter_slow', 8),
            damage_threshold=kwargs.get('damage_threshold', 0.90),
            interface_damage_threshold=kwargs.get('interface_damage_threshold', 0.90)
        )

        # Paramètres de visualisation
        self.plot_settings = PlotSettings(
            save_plots=kwargs.get('save_plots', True),
            display_plots=kwargs.get('display_plots', True),
            output_dir=os.path.join(self.results_dir, 'plots')
        )

        # Options du modèle - CORRECTION : utiliser use_decomposition partout
        self.plane_strain = kwargs.get('plane_strain', True)
        self.use_decomposition = kwargs.get('use_decomposition', kwargs.get('use_decomposition', False))
        self.total_time = kwargs.get('T', 1.0)    
    
    def _initialize_modules(self):
        """Initialise tous les modules dans l'ordre approprié"""
        print("\nInitialisation des modules...")

        # 1. Maillage
        print("  - Génération du maillage...")
        self.mesh = MeshManager(self.mesh_params)
        self.nodes, self.elements, self.material_id = self.mesh.generate_mesh()
        # AJOUT DEBUG
        print(f"\nDÉBUG après génération du maillage:")
        print(f"  - mesh.params.czm_mesh = {self.mesh.params.czm_mesh}")
        print(f"  - len(cohesive_elements) = {len(self.mesh.cohesive_elements)}")
        if len(self.mesh.cohesive_elements) > 0:
            print(f"  - Premier élément cohésif: {self.mesh.cohesive_elements[0].nodes}")

        # Récupérer les dimensions
        self.num_nodes = self.mesh.num_nodes
        self.num_elements = self.mesh.num_elements
        self.num_dofs_u = self.mesh.num_dofs_u
        self.num_dofs_d = self.mesh.num_dofs_d

        # 2. Matériaux
        print("  - Configuration des matériaux...")
        self.material_manager = MaterialManager(
            ice_props=self.ice_props,
            substrate_props=self.substrate_props,
            cohesive_props=self.cohesive_props,
            plane_strain=self.plane_strain,
            k_res=self.phase_field_params.k_res
        )

        # Ajouter l0 au gestionnaire de matériaux
        self.material_manager.l0 = self.phase_field_params.l0
        # CORRECTION : propager use_decomposition au material_manager
        self.material_manager.use_decomposition = self.use_decomposition

        # 3. Décomposition spectrale
        if self.use_decomposition:
            print("  - Initialisation de la décomposition spectrale...")
            self.material_manager.spectral_decomp = SpectralDecomposition(self.material_manager)

        # 4. Zone cohésive
        print("  - Configuration de la zone cohésive...")
        self.cohesive_manager = CohesiveZoneManager(
            material_manager=self.material_manager,
            mesh_manager=self.mesh
        )

        # 5. Matrices élémentaires
        print("  - Préparation des matrices élémentaires...")
        self.element_matrices = ElementMatrices(
            mesh_manager=self.mesh,
            material_manager=self.material_manager
        )

        # 6. Assembleur système
        print("  - Configuration de l'assembleur...")
        self.system_assembler = SystemAssembler(
            mesh_manager=self.mesh,
            material_manager=self.material_manager,
            cohesive_manager=self.cohesive_manager,
            element_matrices=self.element_matrices
        )

        # 7. Calculateur d'énergie
        print("  - Initialisation du calculateur d'énergie...")
        self.energy_calculator = EnergyCalculator(
            mesh_manager=self.mesh,
            material_manager=self.material_manager,
            cohesive_manager=self.cohesive_manager
        )

        # 8. Solveur du champ de phase
        print("  - Configuration du solveur de champ de phase...")
        self.phase_field_solver = PhaseFieldSolver(
            mesh_manager=self.mesh,
            material_manager=self.material_manager,
            energy_calculator=self.energy_calculator,
            params=self.phase_field_params
        )

        # 9. Utilitaires
        print("  - Initialisation des utilitaires...")
        self.damage_checker = DamageChecker(
            damage_threshold=self.solver_params.damage_threshold,
            interface_damage_threshold=self.solver_params.interface_damage_threshold
        )

        self.data_logger = DataLogger(
            output_dir=os.path.join(self.results_dir, 'data')
        )

        self.progress_tracker = ProgressTracker(
            total_time=self.total_time
        )

        # 10. Visualisation
        print("  - Configuration de la visualisation...")
        self.plot_manager = PlotManager(
            mesh_manager=self.mesh,
            material_manager=self.material_manager,
            settings=self.plot_settings
        )

        self.stress_calculator = StressFieldCalculator(
            mesh_manager=self.mesh,
            material_manager=self.material_manager,
            cohesive_manager=self.cohesive_manager
        )

        # 11. Moniteur d'énergie
        self.energy_monitor = EnergyMonitor()

        print("  Tous les modules initialisés avec succès!")


    def _print_configuration_summary(self):
        """Affiche un résumé de la configuration"""
        print("\n" + "="*60)
        print("RÉSUMÉ DE LA CONFIGURATION")
        print("="*60)

        print(f"\nMaillage:")
        print(f"  - Dimensions: {self.mesh_params.length} × {self.mesh_params.total_height} mm")
        print(f"  - Éléments: {self.num_elements} ({self.mesh_params.nx} × {self.mesh_params.ny})")
        print(f"  - Nœuds: {self.num_nodes}")
        print(f"  - Éléments cohésifs: {len(self.mesh.cohesive_elements)}")
        print(f"  - Intégration cohésive: {self.mesh_params.cohesive_integration} "
              f"({self.mesh_params.cohesive_integration_points} points)")

        if self.mesh_params.use_coarse_near_bc:
            print(f"  - Maillage progressif activé:")
            print(f"    • Zone grossière: 0 à {self.mesh_params.coarse_zone_length} mm")
            print(f"    • Ratio de taille: {self.mesh_params.coarsening_ratio}")

        print(f"\nMatériaux:")
        print(f"  Glace:")
        print(f"    - E = {self.ice_props.E} MPa, ν = {self.ice_props.nu}")
        print(f"    - Gc = {self.ice_props.Gc} N/mm")
        print(f"  Substrat:")
        print(f"    - E = {self.substrate_props.E} MPa, ν = {self.substrate_props.nu}")
        print(f"    - Gc = {self.substrate_props.Gc} N/mm")

        print(f"\nInterface cohésive:")
        print(f"  Mode I: σc = {self.cohesive_props.normal_strength} MPa, "
              f"Gc = {self.cohesive_props.normal_Gc} N/mm")
        print(f"  Mode II: τc = {self.cohesive_props.shear_strength} MPa, "
              f"Gc = {self.cohesive_props.shear_Gc} N/mm")
        print(f"  Mixité fixe: {self.cohesive_props.fixed_mixity}")

        print(f"\nChargement:")
        print(f"  - Type: {self.loading_params.load_type}")
        print(f"  - Vitesse angulaire: {self.loading_params.omega} rad/s")
        print(f"  - Temps de rampe: {self.loading_params.ramp_time} s")

        print(f"\nSolveur:")
        print(f"  - Option keep_previous_staggered: {self.solver_params.keep_previous_staggered}")
        print(f"  - Facteurs d'adaptation dt:")
        print(f"    • Augmentation: {self.solver_params.dt_increase_factor} (normal), "
              f"{self.solver_params.dt_increase_fast} (rapide)")
        print(f"    • Réduction: {self.solver_params.dt_decrease_factor} (normal), "
              f"{self.solver_params.dt_decrease_slow} (lente)")

        print(f"\nOptions:")
        print(f"  - {'Déformation plane' if self.plane_strain else 'Contrainte plane'}")
        print(f"  - Décomposition spectrale: {'Activée' if self.use_decomposition else 'Désactivée'}")
        print(f"  - Longueur caractéristique l0: {self.phase_field_params.l0} mm")

        print(f"\nRépertoire de sortie: {self.results_dir}")
        print("="*60)
    
    def _initialize_solution_fields(self):
        """Initialise tous les champs de solution"""
        print("\nInitialisation des champs de solution...")
        
        # Déplacements
        self.u = np.zeros(self.num_dofs_u, dtype=np.float64)
        self.u_prev = np.zeros(self.num_dofs_u, dtype=np.float64)
        
        # Vitesses
        self.v = np.zeros(self.num_dofs_u, dtype=np.float64)
        self.v_prev = np.zeros(self.num_dofs_u, dtype=np.float64)
        
        # Accélérations
        self.a = np.zeros(self.num_dofs_u, dtype=np.float64)
        self.a_prev = np.zeros(self.num_dofs_u, dtype=np.float64)
        
        # Champ d'endommagement
        self.d = np.zeros(self.num_dofs_d, dtype=np.float64)
        self.d_prev = np.zeros(self.num_dofs_d, dtype=np.float64)
        
        # Historiques pour les graphiques
        self.time_history = []
        self.max_damage_history = []
        self.max_interface_damage_history = []
        
        print("  Champs de solution initialisés")
    
    
    def _create_energy_components(self, energy_dict: Dict) -> EnergyComponents:
        """Crée un objet EnergyComponents à partir d'un dictionnaire, en filtrant les champs"""
        return EnergyComponents(
            strain=energy_dict.get('strain', 0.0),
            fracture=energy_dict.get('fracture', 0.0),
            interface=energy_dict.get('interface', 0.0),
            kinetic=energy_dict.get('kinetic', 0.0),
            external_work=energy_dict.get('external_work', 0.0)
        )
    
    def _get_all_parameters(self) -> Dict:
        """Retourne tous les paramètres sous forme de dictionnaire"""
        return {
            'mesh': vars(self.mesh_params),
            'materials': {
                'ice': vars(self.ice_props),
                'substrate': vars(self.substrate_props),
                'cohesive': vars(self.cohesive_props)
            },
            'phase_field': vars(self.phase_field_params),
            'loading': vars(self.loading_params),
            'solver': vars(self.solver_params),
            'options': {
                'plane_strain': self.plane_strain,
                'use_decomposition': self.use_decomposition,  # CORRECTION
                'total_time': self.total_time
            }
        }
    
    def solve(self, callback: Optional[Callable] = None, 
             plot_interval: int = 1,
             save_interval: int = 10) -> Dict:
        """
        Lance la simulation complète
        
        Parameters:
            callback: Fonction appelée après chaque pas convergé
            plot_interval: Intervalle pour les graphiques (tous les N pas)
            save_interval: Intervalle pour la sauvegarde des données
            
        Returns:
            Dictionnaire avec les résultats de la simulation
        """
        print("\n" + "="*60)
        print("DÉMARRAGE DE LA SIMULATION")
        print("="*60)
        
        # Créer le solveur principal
        main_solver = MainSolver(self)
        
        # Callback personnalisé qui combine visualisation et sauvegarde
        def simulation_callback(model, step_results):
            step = step_results.get('step', main_solver.step)
            
            # Mise à jour du suivi de progression
            self.progress_tracker.update(step_results['time'], step)
            
            # Enregistrement des données
            self.data_logger.log_step(
                step=step,
                time=step_results['time'],
                damages={
                    'max_bulk': step_results['damage']['max_bulk'],
                    'max_interface': step_results['damage']['max_interface']
                },
                energies=step_results['energies'],
                convergence_info=step_results.get('convergence', {})
            )
            
            # Historiques
            self.time_history.append(step_results['time'])
            self.max_damage_history.append(step_results['damage']['max_bulk'])
            self.max_interface_damage_history.append(step_results['damage']['max_interface'])
            
            # Moniteur d'énergie
            energies = self._create_energy_components(step_results['energies'])
            self.energy_monitor.add_record(
                time=step_results['time'],
                energies=energies,
                external_work=0.0
            )
            
            # Visualisation
            if step % plot_interval == 0 or step == 1:
                print(f"\nCréation des graphiques pour le pas {step}...")
                
                # État actuel
                self.plot_manager.plot_current_state(
                    u=self.u,
                    d=self.d,
                    interface_damage=[{
                        'nodes': elem.nodes,
                        'gauss_points': elem.gauss_points,
                        'damage': elem.damage
                    } for elem in self.mesh.cohesive_elements],
                    step=step,
                    time=step_results['time'],
                    energies=step_results['energies']
                )
                
                # Champs aux points de Gauss
                field_data = self.stress_calculator.calculate_stress_field(
                    self.u, self.d, self.use_decomposition
                )
                field_data['damage'] = field_data.pop('damage', self.d)
                
                self.plot_manager.plot_field_at_gauss_points(
                    field_data=field_data,
                    field_type='damage',
                    step=step,
                    time=step_results['time']
                )
                
                # Profils de contraintes
                stress_profiles = self.stress_calculator.extract_stress_profiles(
                    self.u, self.d
                )
                self.plot_manager.plot_stress_profiles(
                    stress_data=stress_profiles,
                    step=step,
                    time=step_results['time']
                )
                
                # Débogage de l'interface (si demandé)
                if hasattr(self, 'debug_interface') and self.debug_interface:
                    InterfaceDebugger.print_interface_state(
                        self.mesh, self.cohesive_manager, self.u
                    )
            
            # Sauvegarde périodique
            if step % save_interval == 0:
                self._save_checkpoint(step, step_results['time'])
            
            # Callback utilisateur
            if callback:
                callback(self, step_results)
        
        # Tracer l'état initial
        print("\nÉtat initial:")
        initial_energies = self.energy_calculator.calculate_all_energies(
            self.u, self.v, self.d
        )
        initial_results = {
            'time': 0.0,
            'step': 0,
            'damage': {
                'max_bulk': 0.0,
                'max_interface': 0.0
            },
            'energies': initial_energies.to_dict()
        }
        simulation_callback(self, initial_results)
        
        # Lancer la simulation
        results = main_solver.solve(
            total_time=self.total_time,
            output_interval=1,
            callback=simulation_callback
        )
        
        # Post-traitement final
        if results['success']:
            print("\nPost-traitement final...")
            
            # Sauvegarder toutes les données
            self.data_logger.save_data()
            self._save_final_state()
            
            # Graphiques finaux
            history = {
                'time': self.time_history,
                'max_damage': self.max_damage_history,
                'max_interface_damage': self.max_interface_damage_history,
                'energies': self.energy_monitor.history,
                'final_u': self.u,
                'final_d': self.d
            }
            self.plot_manager.plot_final_results(history)
            
            # Créer l'animation
            print("\nCréation de l'animation...")
            self.plot_manager.create_animation(
                output_file='simulation_animation.mp4'
            )
            
            # Bilan énergétique
            balanced, energy_info = self.energy_monitor.check_energy_balance()
            if not balanced:
                print(f"\nAttention: Bilan énergétique non respecté!")
                print(f"  Erreur relative: {energy_info['relative_error']:.3e}")
        
        return results
    
    def _save_checkpoint(self, step: int, time: float):
        """Sauvegarde un point de contrôle de la simulation"""
        checkpoint_dir = os.path.join(self.results_dir, 'checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_file = os.path.join(checkpoint_dir, f'checkpoint_step_{step:04d}.npz')
        
        np.savez(checkpoint_file,
                 time=time,
                 step=step,
                 u=self.u,
                 v=self.v,
                 a=self.a,
                 d=self.d,
                 u_prev=self.u_prev,
                 v_prev=self.v_prev,
                 a_prev=self.a_prev,
                 d_prev=self.d_prev)
        
        print(f"  Point de contrôle sauvegardé: {checkpoint_file}")
    
    def _save_final_state(self):
        """Sauvegarde l'état final de la simulation"""
        final_dir = os.path.join(self.results_dir, 'final_state')
        os.makedirs(final_dir, exist_ok=True)
        
        # Sauvegarder les champs
        np.save(os.path.join(final_dir, 'displacements.npy'), self.u)
        np.save(os.path.join(final_dir, 'velocities.npy'), self.v)
        np.save(os.path.join(final_dir, 'damage.npy'), self.d)
        
        # Sauvegarder l'endommagement de l'interface
        interface_damage = []
        for elem in self.mesh.cohesive_elements:
            interface_damage.append({
                'nodes': elem.nodes,
                'damage': elem.damage
            })
        np.save(os.path.join(final_dir, 'interface_damage.npy'), interface_damage)
        
        # Sauvegarder le champ d'histoire
        if hasattr(self.phase_field_solver, 'history'):
            np.save(os.path.join(final_dir, 'history_field.npy'), 
                   self.phase_field_solver.history.H_gauss)
    
    def load_checkpoint(self, checkpoint_file: str):
        """Charge un point de contrôle pour reprendre la simulation"""
        print(f"\nChargement du point de contrôle: {checkpoint_file}")
        
        data = np.load(checkpoint_file)
        
        self.u = data['u']
        self.v = data['v']
        self.a = data['a']
        self.d = data['d']
        self.u_prev = data['u_prev']
        self.v_prev = data['v_prev']
        self.a_prev = data['a_prev']
        self.d_prev = data['d_prev']
        
        return float(data['time']), int(data['step'])


# Fonction utilitaire pour créer et exécuter facilement une simulation
def run_simulation(**kwargs):
    """
    Fonction de commodité pour lancer une simulation
    
    Example:
        results = run_simulation(
            nx=100,
            ny_ice=5,
            E_ice=1000,
            T=0.5,
            save_plots=True,
            keep_previous_staggered=True  # Nouvelle option
        )
    """
    # Créer le modèle
    model = IceSubstratePhaseFieldFracture(**kwargs)
    
    # Lancer la simulation
    results = model.solve()
    
    return model, results


# Fonction pour lancer avec un fichier de configuration JSON
def run_simulation_from_config(config_file: str = 'example_config.json', **kwargs):
    """
    Lance une simulation en chargeant les paramètres depuis un fichier JSON
    
    Parameters:
        config_file: Chemin vers le fichier de configuration JSON
        **kwargs: Paramètres additionnels qui écrasent ceux du JSON
    """
    import json
    
    # Charger les paramètres du JSON
    try:
        with open(config_file, 'r') as f:
            params = json.load(f)
        print(f"Paramètres chargés depuis {config_file}")
    except FileNotFoundError:
        print(f"Fichier {config_file} non trouvé, utilisation des paramètres par défaut")
        params = {}
    
    # Écraser avec les kwargs
    params.update(kwargs)
    
    # Lancer la simulation
    return run_simulation(**params)


if __name__ == "__main__":
    # Exemple d'utilisation avec les nouvelles options
    print("Démarrage de la simulation d'exemple...")
    
    # Paramètres de la simulation
    params = {
        # Maillage
        'nx': 100,
        'ny_ice': 5,
        'ny_substrate': 3,
        
        # Temps
        'T': 0.1,
        'dt': 1e-3,
        
        # Nouvelles options de solveur
        'keep_previous_staggered': True,  # Garder la solution convergée
        'dt_increase_factor': 1.1,
        'dt_increase_fast': 1.2,
        'dt_decrease_factor': 0.5,
        'dt_decrease_slow': 0.7,
        'staggered_iter_fast': 2,
        'staggered_iter_slow': 8,
        
        # Options
        'use_decomposition': False,
        'save_plots': True,
        'display_plots': False,
        
        # Intégration cohésive
        'cohesive_integration': 'gauss-lobatto',
        'cohesive_integration_points': 2,
        
        # Maillage progressif
        'use_coarse_near_bc': True,
        'coarse_zone_length': 25.0,
        'coarsening_ratio': 4.0,
        'coarse_zone_reduction': 0.4
    }
    
    # Lancer la simulation
    model, results = run_simulation(**params)
    
    if results['success']:
        print("\nSimulation terminée avec succès!")
        print(f"Résultats sauvegardés dans: {model.results_dir}")
    else:
        print("\nLa simulation a échoué!")