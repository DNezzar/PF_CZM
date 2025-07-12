#!/usr/bin/env python3
"""
Script de simulation avec tous les paramètres configurables
"""
from model import IceSubstratePhaseFieldFracture

# ============================================================
# PARAMÈTRES DE LA SIMULATION
# ============================================================

# --- Paramètres du maillage ---
nx = 150                   # Nombre d'éléments en X
ny_ice = 5                 # Nombre d'éléments dans la glace
ny_substrate = 5            # Nombre d'éléments dans le substrat
length = 170.0              # Longueur du domaine (mm)
ice_height = 6.4            # Hauteur de la couche de glace (mm)
substrate_height = 6.4      # Hauteur du substrat (mm)


# --- Paramètres de maillage progressif ---
use_coarse_near_bc = True   # Activer le maillage grossier près de l'encastrement
coarse_zone_length = 15.0   # Zone grossière de 0 à 25 mm
coarsening_ratio = 5.0      # Éléments 4x plus gros à x=0
coarse_zone_reduction = 0.5 # Utiliser seulement 40% des éléments dans cette zone

# --- Propriétés des matériaux ---
# Glace
E_ice = 1500.0              # Module de Young (MPa)
nu_ice = 0.31               # Coefficient de Poisson
rho_ice = 0.917e-9          # Densité (ton/mm³)
Gc_ice = 0.001             # Énergie de rupture (N/mm)

# Substrat (aluminium)
E_sub = 69000.0             # Module de Young (MPa)
nu_sub = 0.325              # Coefficient de Poisson
rho_sub = 2.7e-9            # Densité (ton/mm³)
Gc_sub = 1.0e+8             # Énergie de rupture (N/mm) - très élevée

# --- Propriétés cohésives de l'interface ---
czm_mesh = True             # True: utilise des éléments cohésifs, False: maillage classique
coh_normal_stiffness = 1.0e+8      # Rigidité normale (MPa/mm)
coh_shear_stiffness = 1.0e+8      # Rigidité en cisaillement (MPa/mm)

# RÉSISTANCES
coh_normal_strength = 0.4e+6          # Résistance normale (MPa)
coh_shear_strength = 0.4e+6           # Résistance en cisaillement (MPa)

# ÉNERGIES DE RUPTURE
coh_normal_Gc = 0.001e+6              # Énergie de rupture normale (N/mm)
coh_shear_Gc = 0.001e+6               # Énergie de rupture en cisaillement (N/mm)

# AUTRES PARAMÈTRES COHÉSIFS
coh_compression_factor = 50.0       # Facteur de pénalité en compression
fixed_mixity = 0.5                  # Mixité des modes (0=Mode I, 1=Mode II)
coh_viscosity = 0.0                 # Viscosité artificielle pour stabilisation

# --- Paramètres du champ de phase ---
l0 = 1.0                     # Longueur caractéristique
k_res = 1.0e-10             # Rigidité résiduelle
irreversibility = True       # Activer l'irréversibilité

# --- Paramètres de chargement ---
omega = 830.1135            # Vitesse angulaire (rad/s)
ramp_time = 1.0             # Temps de montée en charge (s)

# --- Paramètres temporels ---
T = 1.0                     # Temps total de simulation (s)
dt = 1.0e-2                 # Pas de temps initial
dt_min = 1.0e-10            # Pas de temps minimal
dt_max = 1.0e-2             # Pas de temps maximal

# --- Paramètres du solveur ---
max_newton_iter = 30         # Itérations Newton max
newton_tol = 1.0e-4         # Tolérance Newton
max_staggered_iter = 5     # Itérations décalées max
staggered_tol = 1.0e-2      # Tolérance schéma décalé
alpha_HHT = 0.05           # Paramètre HHT-alpha --> alpha [0;1/3]

# --- Paramètres d'adaptation du pas de temps ---
keep_previous_staggered = False    # Garder la solution convergée du schéma alterné
dt_increase_factor = 1.2           # Facteur d'augmentation normale
dt_increase_fast = 1.2             # Facteur d'augmentation rapide
dt_decrease_factor = 0.5           # Facteur de réduction normale
dt_decrease_slow = 0.5             # Facteur de réduction lente

# Seuils d'itérations pour l'adaptation
staggered_iter_fast = 1            # Convergence rapide si <= ce seuil
staggered_iter_slow = 5            # Convergence lente si >= ce seuil

# Seuils d'endommagement
damage_threshold = 0.9             # Seuil d'évolution rapide (volume)
interface_damage_threshold = 0.9   # Seuil d'évolution rapide (interface)

# --- Options ---
use_stress_decomposition = True   # Décomposition spectrale
plane_strain = True              # Déformation plane (vs contrainte plane)
save_plots = True                # Sauvegarder les graphiques
display_plots = False            # Afficher les graphiques
output_dir = 'results'           # Répertoire de sortie

# ============================================================
# CRÉATION ET LANCEMENT DE LA SIMULATION
# ============================================================

# Créer le modèle avec tous les paramètres
model = IceSubstratePhaseFieldFracture(
    # Maillage
    nx=nx,
    ny_ice=ny_ice,
    ny_substrate=ny_substrate,
    length=length,
    ice_height=ice_height,
    substrate_height=substrate_height,
    
    # Zone cohésive
    czm_mesh=czm_mesh,
    
    # Maillage progressif
    use_coarse_near_bc=use_coarse_near_bc,
    coarse_zone_length=coarse_zone_length,
    coarsening_ratio=coarsening_ratio,
    coarse_zone_reduction=coarse_zone_reduction,
    
    # Matériaux
    E_ice=E_ice,
    nu_ice=nu_ice,
    rho_ice=rho_ice,
    Gc_ice=Gc_ice,
    E_sub=E_sub,
    nu_sub=nu_sub,
    rho_sub=rho_sub,
    Gc_sub=Gc_sub,
    
    # Interface cohésive
    coh_normal_stiffness=coh_normal_stiffness,
    coh_shear_stiffness=coh_shear_stiffness,
    coh_normal_strength=coh_normal_strength,
    coh_shear_strength=coh_shear_strength,
    coh_normal_Gc=coh_normal_Gc,
    coh_shear_Gc=coh_shear_Gc,
    coh_compression_factor=coh_compression_factor,
    fixed_mixity=fixed_mixity,
    coh_viscosity=coh_viscosity,
    
    # Champ de phase
    l0=l0,
    k_res=k_res,
    irreversibility=irreversibility,
    
    # Chargement
    omega=omega,
    ramp_time=ramp_time,
    
    # Temps
    T=T,
    dt=dt,
    dt_min=dt_min,
    dt_max=dt_max,
    
    # Solveur de base
    max_newton_iter=max_newton_iter,
    newton_tol=newton_tol,
    max_staggered_iter=max_staggered_iter,
    staggered_tol=staggered_tol,
    alpha_HHT=alpha_HHT,
    
    # Adaptation du pas de temps
    keep_previous_staggered=keep_previous_staggered,
    dt_increase_factor=dt_increase_factor,
    dt_increase_fast=dt_increase_fast,
    dt_decrease_factor=dt_decrease_factor,
    dt_decrease_slow=dt_decrease_slow,
    staggered_iter_fast=staggered_iter_fast,
    staggered_iter_slow=staggered_iter_slow,
    
    # Seuils d'endommagement
    damage_threshold=damage_threshold,
    interface_damage_threshold=interface_damage_threshold,
    
    # Options
    use_stress_decomposition=use_stress_decomposition,
    plane_strain=plane_strain,
    save_plots=save_plots,
    display_plots=display_plots,
    output_dir=output_dir
)

# Lancer la simulation
print("\n" + "="*60)
print("LANCEMENT DE LA SIMULATION")
print("="*60)
print("Configuration principale:")
print(f"  - Éléments cohésifs: {'Activés' if czm_mesh else 'Désactivés'}")
print(f"  - Rigidité cohésive: {coh_normal_stiffness} MPa/mm")
print(f"  - Pas de temps initial: {dt} s")
print(f"  - Seuils d'endommagement: {damage_threshold}")
print("="*60 + "\n")

results = model.solve()

if results['success']:
    print(f"\nSimulation réussie!")
    print(f"  Résultats dans: {model.results_dir}")
    print(f"\nStatistiques d'adaptation du pas de temps:")
    ts_stats = results['statistics']['time_stepping']
    print(f"  - dt min/max: {ts_stats['min_dt']:.3e} / {ts_stats['max_dt']:.3e}")
    print(f"  - Nombre de réductions: {ts_stats['num_reductions']}")
    print(f"  - Nombre d'augmentations: {ts_stats['num_increases']}")
else:
    print(f"\nÉchec de la simulation")
    print(f"  Temps atteint: {results['final_time']:.4f} s")
    print(f"  Pas effectués: {results['total_steps']}")