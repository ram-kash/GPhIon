#!/usr/bin/env python3
"""
GPHIon - Glassy Phase Ionic Conductor Analysis Package
=====================================================

A comprehensive Python package for analyzing ion diffusion in glassy phase 
ionic conductors using molecular dynamics simulations.
"""

from setuptools import setup, find_packages
import os
import re

# Read version from __init__.py
def get_version():
    version_file = os.path.join('gphion', '__init__.py')
    with open(version_file, 'r') as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
        raise RuntimeError("Unable to find version string.")

# Read long description from README
def get_long_description():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# Read requirements
def get_requirements(filename='requirements.txt'):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    # Basic package information
    name="gphion",
    version=get_version(),
    author="Dr. Ram Sewak",
    author_email="ram.sewak@iitgn.ac.in",
    description="Glassy Phase Ionic Conductor Analysis Package",
    long_description=get_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/gphion",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/gphion/issues",
        "Documentation": "https://gphion.readthedocs.io",
        "Source Code": "https://github.com/yourusername/gphion",
    },
    
    # Package structure
    packages=find_packages(),
    package_data={
        'gphion': [
            'data/*.dat',
            'examples/*.py',
            'tests/*.py',
        ],
    },
    include_package_data=True,
    
    # Dependencies
    install_requires=[
        "numpy>=1.18.0",
        "scipy>=1.5.0",
        "matplotlib>=3.2.0",
        "MDAnalysis>=2.0.0",
        "tqdm>=4.50.0",
        "networkx>=2.5",
    ],
    
    # Optional dependencies
    extras_require={
        'full': [
            "seaborn>=0.11.0",
            "scikit-image>=0.17.0",
            "pandas>=1.2.0",
            "jupyter>=1.0.0",
            "ipywidgets>=7.6.0",
        ],
        'dev': [
            "pytest>=6.0.0",
            "pytest-cov>=2.10.0",
            "black>=21.0.0",
            "flake8>=3.8.0",
            "sphinx>=3.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "pre-commit>=2.10.0",
        ],
        'docs': [
            "sphinx>=3.0.0",
            "sphinx-rtd-theme>=0.5.0",
            "nbsphinx>=0.8.0",
            "pandoc>=1.0.0",
        ],
        'plotting': [
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
            "bokeh>=2.3.0",
        ],
        'performance': [
            "numba>=0.53.0",
            "dask>=2021.3.0",
            "joblib>=1.0.0",
        ]
    },
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Classification
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Physics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for searchability
    keywords=[
        "molecular dynamics",
        "ionic conductors",
        "glass",
        "diffusion",
        "ion transport",
        "TAOG",
        "van hove",
        "jump analysis",
        "materials science",
        "computational chemistry",
        "solid electrolytes",
        "battery materials"
    ],
    
    # Entry points for command-line interface
    entry_points={
        'console_scripts': [
            'gphion-analyze=gphion.cli:main',
            'gphion-plot=gphion.visualization.cli:main',
        ],
    },
    
    # Additional metadata
    license="MIT",
    zip_safe=False,
    platforms=["any"],
    
    # Test suite
    test_suite="tests",
)

