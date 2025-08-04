"""
Van Hove Correlation Function Analysis Module
============================================

This module calculates both self and distinct parts of the Van Hove correlation function
for analyzing particle dynamics in molecular dynamics simulations.

The Van Hove correlation function G(r,t) describes the probability of finding a particle
at distance r from its initial position after time t, providing insights into:
- Diffusion dynamics
- Local structural correlations  
- Particle jump mechanisms
"""

import numpy as np
from tqdm import tqdm
import warnings
import os

class VanHoveAnalyzer:
    """
    Calculates Van Hove correlation functions for diffusing ions.
    
    The Van Hove correlation function has two parts:
    - Self part: G_self(r,t) = ⟨δ(r - |r_i(t) - r_i(0)|)⟩
    - Distinct part: G_distinct(r,t) = ⟨δ(r - |r_j(t) - r_i(0)|)⟩ where i≠j
    """
    
    def __init__(self, sim_data, diffusing_element=None):
        """
        Initialize the Van Hove analyzer.
        
        Args:
            sim_data (SimData): The main simulation data object
            diffusing_element (str, optional): Element to analyze. If None, uses sim_data.diff_elem
        """
        self.sim_data = sim_data
        self.universe = sim_data.universe
        self.diffusing_element = diffusing_element or sim_data.diff_elem
        
        # Select diffusing atoms
        self.diffusing_atoms = self.universe.select_atoms(f"element {self.diffusing_element}")
        self.n_particles = len(self.diffusing_atoms)
        self.n_frames = len(self.universe.trajectory)
        
        print(f"\n--- Van Hove Correlation Function Analysis ---")
        print(f"Analyzing {self.n_particles} {self.diffusing_element} atoms")
        print(f"Trajectory has {self.n_frames} frames")
        
        # Validate trajectory
        self._validate_trajectory()
        
    def _validate_trajectory(self):
        """Validate trajectory and extract box dimensions."""
        print(f"Timestep: {self.universe.trajectory.dt} fs")
        
        # Get box dimensions from first frame
        self.universe.trajectory[0]
        self.box = self.universe.dimensions[:3]
        print(f"Box dimensions: {self.box} Å")
        
        # Check if box is roughly cubic
        if not np.allclose(self.box, self.box[0], rtol=0.01):
            warnings.warn("Box is not cubic. This may affect the analysis.")
            
        # Set maximum distance for analysis
        self.max_r = min(self.box) / 2  # Avoid double counting across PBC
        
    def _apply_minimum_image(self, disp):
        """Apply minimum image convention for periodic boundary conditions."""
        return disp - self.box * np.round(disp / self.box)
    
    def setup_analysis_parameters(self, time_lags=None, max_r=None, n_bins=2000):
        """
        Set up parameters for Van Hove analysis.
        
        Args:
            time_lags (list, optional): Time lags in frame units. Default: [10, 100, 300, 500, 700, 900]
            max_r (float, optional): Maximum distance for analysis. Default: half smallest box dimension
            n_bins (int): Number of radial bins
        """
        # Default time lags
        if time_lags is None:
            self.time_lags = [10, 100, 300, 500, 700, 900]
        else:
            self.time_lags = time_lags
            
        # Filter time lags that are too large for the trajectory
        self.time_lags = [lag for lag in self.time_lags if lag < self.n_frames]
        
        if not self.time_lags:
            raise ValueError("No valid time lags found. Trajectory might be too short.")
            
        # Set maximum distance
        if max_r is not None:
            self.max_r = min(max_r, min(self.box) / 2)
        
        # Set up binning
        self.n_bins = n_bins
        self.bin_edges = np.linspace(0, self.max_r, self.n_bins + 1)
        self.r = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])  # Bin centers
        self.dr = self.bin_edges[1] - self.bin_edges[0]
        
        print(f"Analysis parameters:")
        print(f"  Max r: {self.max_r:.2f} Å")
        print(f"  Bin width: {self.dr:.4f} Å")
        print(f"  Time lags: {self.time_lags}")
        print(f"  Valid frames for analysis: {self.n_frames - max(self.time_lags)}")
        
    def calculate_self_van_hove(self):
        """
        Calculate the self-part of the van Hove correlation function.
        G_self(r,t) = ⟨δ(r - |r_i(t) - r_i(0)|)⟩
        
        Returns:
            dict: Dictionary with time lags as keys and histograms as values
        """
        results = {lag: np.zeros(self.n_bins) for lag in self.time_lags}
        
        print("Computing self-part of van Hove function...")
        
        for i in tqdm(range(self.n_frames - max(self.time_lags)), 
                     desc="Self Van Hove"):
            # Get reference positions
            self.universe.trajectory[i]
            ref_pos = self.diffusing_atoms.positions.copy()
            current_box = self.universe.dimensions[:3]
            
            for lag in self.time_lags:
                # Get positions at time t + lag
                self.universe.trajectory[i + lag]
                cur_pos = self.diffusing_atoms.positions.copy()
                
                # Calculate displacement with minimum image convention
                disp = cur_pos - ref_pos
                disp = disp - current_box * np.round(disp / current_box)
                
                # Calculate distances
                distances = np.linalg.norm(disp, axis=1)
                
                # Histogram and accumulate
                hist, _ = np.histogram(distances, bins=self.bin_edges)
                results[lag] += hist
                
        self.self_results = results
        return results
    
    def calculate_distinct_van_hove(self):
        """
        Calculate the distinct-part of the van Hove correlation function.
        G_distinct(r,t) = ⟨δ(r - |r_j(t) - r_i(0)|)⟩ where i≠j
        
        Returns:
            dict: Dictionary with time lags as keys and histograms as values
        """
        results = {lag: np.zeros(self.n_bins) for lag in self.time_lags}
        
        print("Computing distinct-part of van Hove function...")
        
        for frame_idx in tqdm(range(self.n_frames - max(self.time_lags)), 
                             desc="Distinct Van Hove"):
            # Get reference positions
            self.universe.trajectory[frame_idx]
            ref_pos = self.diffusing_atoms.positions.copy()
            current_box = self.universe.dimensions[:3]
            
            for lag in self.time_lags:
                # Get positions at time t + lag
                self.universe.trajectory[frame_idx + lag]
                cur_pos = self.diffusing_atoms.positions.copy()
                
                # Calculate all i,j pairs where i≠j
                distances = []
                for i in range(self.n_particles):
                    for j in range(self.n_particles):
                        if i != j:
                            # Distance from particle j at time t to particle i at time 0
                            disp = cur_pos[j] - ref_pos[i]
                            disp = disp - current_box * np.round(disp / current_box)
                            dist = np.linalg.norm(disp)
                            distances.append(dist)
                
                # Histogram and accumulate
                if distances:  # Only if we have valid distances
                    hist, _ = np.histogram(distances, bins=self.bin_edges)
                    results[lag] += hist
                    
        self.distinct_results = results
        return results
    
    def normalize_results(self):
        """
        Normalize van Hove correlation function results.
        
        Returns:
            tuple: (normalized_self_results, normalized_distinct_results)
        """
        if not hasattr(self, 'self_results') or not hasattr(self, 'distinct_results'):
            raise ValueError("Must calculate van Hove functions before normalizing")
            
        # Shell volumes for spherical normalization
        shell_volumes = 4 * np.pi * self.r**2 * self.dr
        
        # Avoid division by zero at r=0
        shell_volumes[0] = 4 * np.pi * (self.dr/2)**2 * self.dr
        
        # Normalize self-part
        self.self_normalized = {}
        for lag in self.time_lags:
            n_time_origins = self.n_frames - lag
            normalization = self.n_particles * n_time_origins * shell_volumes
            self.self_normalized[lag] = self.self_results[lag] / normalization
            
        # Normalize distinct-part  
        self.distinct_normalized = {}
        for lag in self.time_lags:
            n_time_origins = self.n_frames - lag
            normalization = (self.n_particles * (self.n_particles - 1) * 
                           n_time_origins * shell_volumes)
            self.distinct_normalized[lag] = self.distinct_results[lag] / normalization
            
        print("✅ Van Hove normalization complete")
        return self.self_normalized, self.distinct_normalized
    
    def save_results(self, output_dir=None):
        """
        Save Van Hove results to files.
        
        Args:
            output_dir (str, optional): Directory to save results
        """
        if output_dir is None:
            output_dir = "."
            
        os.makedirs(output_dir, exist_ok=True)
        
        if hasattr(self, 'self_normalized') and hasattr(self, 'distinct_normalized'):
            # Save normalized data
            self._save_van_hove_data(
                self.r, self.self_normalized, 
                os.path.join(output_dir, "g_self_rt.dat"), "self"
            )
            self._save_van_hove_data(
                self.r, self.distinct_normalized,
                os.path.join(output_dir, "g_distinct_rt.dat"), "distinct"
            )
        else:
            print("⚠️  Normalized results not available. Run normalize_results() first.")
            
    def _save_van_hove_data(self, r, results, filename, data_type):
        """Save van Hove data to file with proper headers."""
        output = np.column_stack([r] + [results[lag] for lag in self.time_lags])
        header = f"r (Å) " + " ".join([f"G_{data_type}(r, t={lag})" for lag in self.time_lags])
        np.savetxt(filename, output, header=header, fmt="%.8e")
        print(f"Saved: {filename}")
        
    def run_complete_analysis(self, time_lags=None, max_r=None, n_bins=2000, 
                             output_dir=None):
        """
        Run complete Van Hove analysis workflow.
        
        Args:
            time_lags (list, optional): Time lags for analysis
            max_r (float, optional): Maximum distance
            n_bins (int): Number of radial bins
            output_dir (str, optional): Output directory
            
        Returns:
            dict: Complete results dictionary
        """
        print("🔄 Starting complete Van Hove analysis...")
        
        # Setup parameters
        self.setup_analysis_parameters(time_lags, max_r, n_bins)
        
        # Calculate Van Hove functions
        self.calculate_self_van_hove()
        self.calculate_distinct_van_hove()
        
        # Normalize results
        self.normalize_results()
        
        # Save results
        if output_dir:
            self.save_results(output_dir)
            
        print("✅ Van Hove analysis complete!")
        
        return {
            'self_normalized': self.self_normalized,
            'distinct_normalized': self.distinct_normalized,
            'r_values': self.r,
            'time_lags': self.time_lags,
            'analysis_parameters': {
                'max_r': self.max_r,
                'n_bins': self.n_bins,
                'dr': self.dr,
                'n_particles': self.n_particles,
                'n_frames': self.n_frames
            }
        }

