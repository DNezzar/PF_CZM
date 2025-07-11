"""
Script d'installation pour le package PF-CZM FEM
"""

from setuptools import setup, find_packages

# Lire le README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Lire les dépendances depuis requirements.txt si disponible
try:
    with open("requirements.txt", "r") as f:
        requirements = f.read().splitlines()
except FileNotFoundError:
    requirements = [
        "numpy>=1.19.0",
        "scipy>=1.5.0",
        "matplotlib>=3.3.0",
    ]

setup(
    name="pf-czm-fem",
    version="1.0.0",
    author="Votre nom",
    author_email="votre.email@example.com",
    description="Phase Field - Cohesive Zone Model for ice-substrate delamination",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/votre-username/pf-czm-fem",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "sphinx",
            "sphinx-rtd-theme",
        ],
        "viz": [
            "ffmpeg-python",  # Pour les animations
            "imageio",        # Alternative pour les animations
        ],
        "parallel": [
            "mpi4py",         # Pour la parallélisation MPI
            "numba",          # Pour l'accélération JIT
        ],
    },
    entry_points={
        "console_scripts": [
            "pfczm=main:main",
            "pfczm-sim=main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.md"],
    },
)
