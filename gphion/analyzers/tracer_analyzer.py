"""Tracer diffusivity and conductivity analysis from MSD."""

import numpy as np

class TracerPropertyAnalyzer:
    """Calculates tracer diffusivity and conductivity based on MSD of diffusing ions."""

    def __init__(self, sim_data):
        """
        Args:
            sim_data (SimData): The main simulation data object.
        """
        print("\n--- Stage: Analyzing Tracer Properties (MSD) ---")
        self.sim_data = sim_data
        
        if not hasattr(sim_data, 'displacement') or self.sim_data.displacement is None:
            raise AttributeError("SimData object must have 'displacement' calculated first.")

    def calculate_tracer_properties(self, diffusion_dim=3, z_ion=1.0):
        """
        Calculate tracer properties based on the simulation data.
        
        Args:
            diffusion_dim (int): The number of dimensions for diffusion (e.g., 3).
            z_ion (float): The ionic charge of the diffusing species (e.g., 1.0 for Li+).
            
        Returns:
            dict: A dictionary containing the calculated tracer properties.
        """
        # Extract Parameters from SimData
        sim_data = self.sim_data
        ang2m = 1E-10  # Conversion from Angstroms to meters

        sim_data.diffusion_dim = diffusion_dim
        sim_data.ion_charge = z_ion

        # Set default temperature if not available
        if not hasattr(sim_data, 'temperature') or sim_data.temperature is None:
            sim_data.temperature = 300.0  # Assume 300 K if not specified
            print(f"WARNING: Simulation temperature not found. Assuming {sim_data.temperature} K.")

        print("-------------------------------------------")
        print("Calculating tracer diffusion and conductivity based on:")
        print(f"{diffusion_dim} dimensional diffusion, and an ion with a charge of {z_ion}")

        # Calculate Particle Density
        if not hasattr(sim_data, 'volume') or sim_data.volume is None or sim_data.volume <= 0:
            # Fallback to calculate from the final lattice dimensions
            sim_data.volume = np.linalg.det(sim_data.lattice) * 1e-30
            
        if sim_data.volume <= 0:
            raise ValueError("Cannot calculate density with zero or negative volume.")

        particle_density = sim_data.nr_diffusing / sim_data.volume  # in particles/m^3
        mol_per_liter = particle_density / (sim_data.avogadro * 1000)  # 1000 L/m^3

        # Calculate Mean Squared Displacement (MSD) over time
        import scipy.stats
        msd_time_series = np.mean(sim_data.displacement**2, axis=0) # shape (nr_steps,)
        time_points = np.arange(sim_data.nr_steps) * sim_data.time_step
        
        # Calculate Tracer Diffusivity using linear regression (D = slope / (2 * dimensions))
        if len(time_points) > 2:
            start_idx = max(1, len(time_points) // 10) # Skip initial ballistic regime
            slope, _, _, _, _ = scipy.stats.linregress(time_points[start_idx:], msd_time_series[start_idx:])
            tracer_diff = (slope * ang2m**2) / (2 * diffusion_dim)
        else:
            # Fallback for very short trajectories
            tracer_diff = (msd_time_series[-1] * ang2m**2) / (2 * diffusion_dim * sim_data.total_time)
            
        msd_angstrom_sq = msd_time_series[-1]

        # Calculate Tracer Conductivity (Nernst-Einstein Equation)
        # Conductivity = q^2 * z^2 * D * n / (k_B * T)
        numerator = (sim_data.e_charge**2) * (z_ion**2) * tracer_diff * particle_density
        denominator = sim_data.k_boltzmann * sim_data.temperature
        tracer_conduc = numerator / denominator

        # Print and Return Results
        print(f"Tracer diffusivity determined to be (in m^2/s): {tracer_diff:.4e}")
        print(f"Tracer conductivity determined to be (in Siemens/meter): {tracer_conduc:.4f}")
        print("-------------------------------------------")

        results = {
            'tracer_diffusivity': tracer_diff,
            'tracer_conductivity': tracer_conduc,
            'particle_density': particle_density,
            'mol_per_liter': mol_per_liter,
            'msd': msd_angstrom_sq
        }

        return results

