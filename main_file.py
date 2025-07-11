#!/usr/bin/env python3
"""
Point d'entrée principal pour le modèle PF-CZM Ice-Substrate
"""

import argparse
import json
import sys
from model import IceSubstratePhaseFieldFracture, run_simulation


def load_parameters_from_file(filename):
    """Charge les paramètres depuis un fichier JSON"""
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    """Fonction principale"""
    # Parser d'arguments en ligne de commande
    parser = argparse.ArgumentParser(
        description="Simulation de rupture par champ de phase avec zone cohésive"
    )
    
    # Arguments principaux
    parser.add_argument('--config', type=str, help='Fichier de configuration JSON')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Répertoire de sortie')
    
    # Paramètres de maillage
    parser.add_argument('--nx', type=int, default=250,
                       help='Nombre d\'éléments en x')
    parser.add_argument('--ny-ice', type=int, default=10,
                       help='Nombre d\'éléments dans la glace')
    parser.add_argument('--ny-substrate', type=int, default=5,
                       help='Nombre d\'éléments dans le substrat')
    
    # Paramètres temporels
    parser.add_argument('--total-time', type=float, default=1.0,
                       help='Temps total de simulation')
    parser.add_argument('--dt', type=float, default=1e-2,
                       help='Pas de temps initial')
    
    # Options du modèle
    parser.add_argument('--stress-decomposition', action='store_true',
                       help='Utiliser la décomposition spectrale')
    parser.add_argument('--plane-stress', action='store_true',
                       help='Utiliser contrainte plane (défaut: déformation plane)')
    
    # Options de visualisation
    parser.add_argument('--no-plots', action='store_true',
                       help='Désactiver les graphiques')
    parser.add_argument('--plot-interval', type=int, default=1,
                       help='Intervalle entre les graphiques')
    parser.add_argument('--no-animation', action='store_true',
                       help='Désactiver la création de l\'animation')
    
    # Mode de simulation
    parser.add_argument('--quick-test', action='store_true',
                       help='Test rapide avec maillage grossier')
    parser.add_argument('--high-resolution', action='store_true',
                       help='Simulation haute résolution')
    
    args = parser.parse_args()
    
    # Charger les paramètres
    if args.config:
        print(f"Chargement de la configuration depuis {args.config}")
        params = load_parameters_from_file(args.config)
    else:
        params = {}
    
    # Appliquer les arguments de ligne de commande
    params.update({
        'output_dir': args.output_dir,
        'nx': args.nx,
        'ny_ice': args.ny_ice,
        'ny_substrate': args.ny_substrate,
        'T': args.total_time,
        'dt': args.dt,
        'use_stress_decomposition': args.stress_decomposition,
        'plane_strain': not args.plane_stress,
        'save_plots': not args.no_plots,
        'display_plots': False,  # Désactivé pour les runs en batch
    })
    
    # Modes spéciaux
    if args.quick_test:
        print("\nMode test rapide activé")
        params.update({
            'nx': 50,
            'ny_ice': 3,
            'ny_substrate': 2,
            'T': 0.1,
            'dt': 1e-2,
            'max_newton_iter': 3,
            'max_staggered_iter': 5
        })
    elif args.high_resolution:
        print("\nMode haute résolution activé")
        params.update({
            'nx': 500,
            'ny_ice': 20,
            'ny_substrate': 10,
            'dt': 5e-3,
            'newton_tol': 1e-5,
            'staggered_tol': 1e-3
        })
    
    # Afficher la configuration
    print("\n" + "="*60)
    print("CONFIGURATION DE LA SIMULATION")
    print("="*60)
    print(f"Maillage: {params['nx']} × ({params['ny_ice']} + {params['ny_substrate']}) éléments")
    print(f"Temps total: {params['T']} s, dt initial: {params['dt']} s")
    print(f"Décomposition spectrale: {'Oui' if params['use_stress_decomposition'] else 'Non'}")
    print(f"Répertoire de sortie: {params['output_dir']}")
    print("="*60)
    
    try:
        # Créer et exécuter le modèle
        print("\nCréation du modèle...")
        model = IceSubstratePhaseFieldFracture(**params)
        
        # Définir un callback pour afficher la progression
        def progress_callback(model, step_results):
            if step_results.get('step', 0) % 10 == 0:
                print(f"\n[Progression] Pas {step_results.get('step')}: "
                      f"t = {step_results['time']:.4f} s, "
                      f"Endommagement max = {step_results['damage']['max_bulk']:.4f}")
        
        # Lancer la simulation
        print("\nDémarrage de la simulation...")
        results = model.solve(
            callback=progress_callback if not args.no_plots else None,
            plot_interval=args.plot_interval
        )
        
        # Résumé final
        if results['success']:
            print("\n" + "="*60)
            print("SIMULATION TERMINÉE AVEC SUCCÈS")
            print("="*60)
            print(f"Temps simulé: {results['final_time']:.4f} s")
            print(f"Nombre de pas: {results['total_steps']}")
            print(f"Temps de calcul: {results['computation_time']:.2f} s")
            
            stats = results['statistics']
            print(f"\nStatistiques:")
            print(f"  - Itérations Newton moyennes/pas: {stats['average_newton_per_step']:.2f}")
            print(f"  - Itérations décalées moyennes/pas: {stats['average_staggered_per_step']:.2f}")
            print(f"  - Pas de temps min/max: {stats['time_stepping']['min_dt']:.2e} / "
                  f"{stats['time_stepping']['max_dt']:.2e}")
            
            print(f"\nRésultats sauvegardés dans: {model.results_dir}")
            
            # Créer l'animation si demandé
            if not args.no_animation and not args.no_plots:
                print("\nCréation de l'animation finale...")
                model.plot_manager.create_animation()
            
            return 0
        else:
            print("\n" + "="*60)
            print("ÉCHEC DE LA SIMULATION")
            print("="*60)
            print(f"Arrêt à t = {results['final_time']:.4f} s (pas {results['total_steps']})")
            return 1
            
    except KeyboardInterrupt:
        print("\nSimulation interrompue par l'utilisateur")
        return 2
    except Exception as e:
        print(f"\nErreur lors de la simulation: {e}")
        import traceback
        traceback.print_exc()
        return 3


# Configurations prédéfinies pour différents cas de test
PRESETS = {
    "default": {
        "description": "Configuration par défaut",
        "params": {}
    },
    
    "fine_mesh": {
        "description": "Maillage fin pour haute précision",
        "params": {
            "nx": 400,
            "ny_ice": 16,
            "ny_substrate": 8,
            "dt": 5e-3
        }
    },
    
    "coarse_mesh": {
        "description": "Maillage grossier pour tests rapides",
        "params": {
            "nx": 100,
            "ny_ice": 4,
            "ny_substrate": 2,
            "dt": 1e-2
        }
    },
    
    "high_speed": {
        "description": "Vitesse de rotation élevée",
        "params": {
            "omega": 1200.0,
            "dt": 5e-3,
            "ramp_time": 0.5
        }
    },
    
    "brittle_interface": {
        "description": "Interface fragile",
        "params": {
            "coh_normal_strength": 0.1,
            "coh_shear_strength": 0.1,
            "coh_normal_Gc": 0.0001,
            "coh_shear_Gc": 0.0001
        }
    },
    
    "ductile_interface": {
        "description": "Interface ductile",
        "params": {
            "coh_normal_strength": 0.5,
            "coh_shear_strength": 0.5,
            "coh_normal_Gc": 0.001,
            "coh_shear_Gc": 0.001
        }
    }
}


def list_presets():
    """Liste les configurations prédéfinies disponibles"""
    print("\nConfigurations prédéfinies disponibles:")
    print("-" * 40)
    for name, preset in PRESETS.items():
        print(f"{name:20} - {preset['description']}")
    print("-" * 40)


def run_preset(preset_name, **kwargs):
    """Lance une simulation avec une configuration prédéfinie"""
    if preset_name not in PRESETS:
        print(f"Erreur: Configuration '{preset_name}' non trouvée")
        list_presets()
        return None, None
    
    preset = PRESETS[preset_name]
    print(f"\nUtilisation de la configuration: {preset['description']}")
    
    # Fusionner les paramètres
    params = preset['params'].copy()
    params.update(kwargs)
    
    # Lancer la simulation
    return run_simulation(**params)


if __name__ == "__main__":
    # Si lancé directement, utiliser la fonction main
    sys.exit(main())
