# PF-CZM FEM: Phase Field - Cohesive Zone Model for Ice-Substrate Shedding

## 📋 Description

PF-CZM FEM is a finite element simulation framework that combines Phase Field fracture modeling with Cohesive Zone modeling to simulate the detachement and delamination of ice from a substrate under centrifugal loading. This model is particularly relevant for understanding ice shedding in aerospace applications, wind turbines, and other engineering systems where ice accretion is a concern.

## 🚀 Key Features

### Physics Modeling
- **Hybrid Fracture Model**: Combines Phase Field method for bulk fracture with Cohesive Zone Model for interface delamination
- **Two-material System**: Handles ice-substrate systems with distinct material properties
- **Centrifugal Loading**: Simulates distributed forces typical in turbomachinery applications

### Numerical Methods
- **HHT-α Time Integration**: Robust implicit time stepping with numerical damping
- **Staggered Scheme**: Alternating minimization between mechanical and damage fields
- **Adaptive Time Stepping**: Automatic adjustment based on convergence and damage evolution

## 📦 Installation

### Prerequisites
```bash
Python >= 3.7
NumPy >= 1.19.0
SciPy >= 1.5.0
Matplotlib >= 3.3.0
```

### Install from source
```bash
git clone https://github.com/your-username/pf-czm-fem.git
cd pf-czm-fem
pip install -r requirements.txt
```

### Optional dependencies for animation features
```bash
# For animations
pip install ffmpeg-python imageio

# For parallel computing
pip install mpi4py numba
```

## 🎯 Quick Start

### Basic simulation
```python
from model import run_simulation

# Run with default parameters
model, results = run_simulation()

# Custom parameters
model, results = run_simulation(
    nx=200,                    # Mesh resolution in x
    ny_ice=10,                 # Elements in ice layer
    ny_substrate=5,            # Elements in substrate
    T=1.0,                     # Total simulation time
    omega=500.0,            # Angular velocity (rad/s)
    czm_mesh=True,             # Enable cohesive zones
    save_plots=True            # Save visualization
)
```

### Using the launch script
```bash
python launch.py
```

### Command line interface
```bash
python main_file.py --nx 200 --total-time 1.0 --stress-decomposition
```

## 📁 Project Structure

```
pf-czm-fem/
├── results/                 # Results repository (stress plot, nodal bulk damage and displacement, bulk and interface damage at gauss point)
├── model.py                 # Main model class and simulation orchestration
├── mesh.py                  # Mesh generation and cohesive element management
├── materials.py             # Material properties and constitutive models
├── cohesive_zone.py         # Cohesive zone implementation
├── phase_field.py           # Phase field solver for bulk fracture
├── mechanics.py             # Mechanical system assembly
├── solver.py                # HHT-α and staggered solvers
├── energies.py              # Energy calculations and monitoring
├── visualization.py         # Plotting and post-processing
├── utils.py                 # Utility functions and helpers
├── launch.py                # Configurable simulation launcher
└── main_file.py             # Command line interface
```

## ⚙️ Key Parameters

### Mesh Parameters
```python
length = 150.0               # Domain length (mm)
ice_height = 6.4             # Ice layer thickness (mm)
substrate_height = 6.4       # Substrate thickness (mm)
nx = 200                     # Elements in x-direction
ny_ice = 10                  # Elements in ice layer
ny_substrate = 5             # Elements in substrate
```

### Material Properties
```python
# Ice
E_ice = 9000.0               # Young's modulus (MPa)
nu_ice = 0.31                # Poisson's ratio
Gc_ice = 0.001               # Fracture energy (N/mm)

# Substrate (Aluminum)
E_sub = 69000.0              # Young's modulus (MPa)
nu_sub = 0.325               # Poisson's ratio
```

### Cohesive Interface
```python
coh_normal_strength = 0.2    # Normal strength (MPa)
coh_shear_strength = 0.2     # Shear strength (MPa)
coh_normal_Gc = 0.00025      # Mode I fracture energy (N/mm)
coh_shear_Gc = 0.00025       # Mode II fracture energy (N/mm)
```

### Solver Settings
```python
alpha_HHT = 0.05             # HHT-α parameter (0-1/3)
max_newton_iter = 5          # Newton iterations
max_staggered_iter = 10      # Staggered iterations
dt = 1.0e-2                  # Initial time step
```

## 📊 Example Results

The simulation produces comprehensive outputs including:

1. **Displacement and Damage Fields**: Evolution of deformation and fracture
2. **Interface Damage**: Progression of delamination along the interface
3. **Stress Profiles**: Distribution of stresses in the system
4. **Energy Evolution**: Tracking of different energy components
5. **Animations**: Time-lapse visualization of the fracture process

## 🔧 Advanced Usage

### Custom Material Model
```python
from materials import MaterialProperties

custom_ice = MaterialProperties(
    E=2000.0,      # MPa
    nu=0.35,
    rho=0.92e-9,   # ton/mm³
    Gc=0.002,      # N/mm
    name="Custom Ice"
)
```

### Adaptive Mesh Refinement near-edge singularity
```python
model = IceSubstratePhaseFieldFracture(
    use_coarse_near_bc=True,
    coarse_zone_length=25.0,
    coarsening_ratio=4.0,
    coarse_zone_reduction=0.4
)
```

### Post-Processing
```python
# Extract crack paths
crack_info = model.phase_field_solver.get_crack_path(
    d=model.d, 
    threshold=0.95
)

# Calculate energy release rate
G = model.phase_field_solver.compute_energy_release_rate(
    model.d, 
    dd_dt
)
```


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📚 References

1. Bourdin, B., Francfort, G. A., & Marigo, J. J. (2008). The variational approach to fracture.
2. Johann Rannou, Christophe Bovet. Domain decomposition methods and acceleration techniques for the phase field fracture staggered solver. International Journal for Numerical Methods in Engineering, 2023, ⟨10.1002/nme.7544⟩. ⟨hal-03938084v2⟩


## 👥 Authors

- Dorian Nezzar - https://github.com/DNezzar/PF_CZM
