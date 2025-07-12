"""
Package PF-CZM FEM pour la simulation de rupture par champ de phase
avec modèle de zones cohésives
"""

__version__ = "1.0.0"
__author__ = "Dorian Nezzar"
__email__ = "dorian.nezzar@hotmail.fr"

# Import des classes et fonctions principales
from .model import IceSubstratePhaseFieldFracture, run_simulation
from .mesh import MeshManager, MeshParameters
from .materials import (MaterialManager, MaterialProperties, 
                       CohesiveProperties, SpectralDecomposition)
from .phase_field import PhaseFieldSolver, PhaseFieldParameters
from .solvers import SolverParameters
from .visualization import PlotManager, PlotSettings

# Import des utilitaires
from .utils import (SimulationParameters, create_results_directory,
                   save_simulation_parameters)

# Liste des exports publics
__all__ = [
    # Classes principales
    'IceSubstratePhaseFieldFracture',
    'MeshManager',
    'MaterialManager',
    'PhaseFieldSolver',
    'PlotManager',
    
    # Paramètres
    'MeshParameters',
    'MaterialProperties',
    'CohesiveProperties',
    'PhaseFieldParameters',
    'SolverParameters',
    'SimulationParameters',
    'PlotSettings',
    
    # Fonctions
    'run_simulation',
    'create_results_directory',
    'save_simulation_parameters',
    
    # Méta
    '__version__',
]

# Message de bienvenue (optionnel)
def _print_info():
    """Affiche les informations du package"""
    print(f"PF-CZM FEM Package v{__version__}")
    print("Pour démarrer rapidement:")
    print("  from pf_czm_fem import run_simulation")
    print("  model, results = run_simulation()")

# Vérification des dépendances
def check_dependencies():
    """Vérifie que toutes les dépendances sont installées"""
    dependencies = {
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    for module, name in dependencies.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(name)
    
    if missing:
        print(f"Attention: Dépendances manquantes: {', '.join(missing)}")
        print("Installer avec: pip install numpy scipy matplotlib")
        return False
    return True

# Vérifier les dépendances au chargement
_deps_ok = check_dependencies()
