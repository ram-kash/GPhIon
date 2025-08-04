"""Coordinate processing utilities."""

import numpy as np

class CoordinateProcessor:
    """Processes coordinates to calculate fractional positions and displacement."""

    def __init__(self):
        print("DEBUG: CoordinateProcessor instance created.")

    def frac_and_disp(self, sim_data):
        """Calculate fractional coordinates and displacement."""
        print('DEBUG: Calculating fractional coordinates and displacement...')
        
        sim_data.frac_pos = np.zeros((3, sim_data.nr_atoms, sim_data.nr_steps))
        sim_data.displacement = np.zeros((sim_data.nr_atoms, sim_data.nr_steps))
        
        lattice_inv = np.linalg.inv(sim_data.lattice)

        for atom in range(sim_data.nr_atoms):
            uc = np.zeros(3)  # Unit cell crossings tracker
            start_pos = sim_data.cart_pos[:, atom, 0]
            
            for time in range(sim_data.nr_steps):
                current_frac = np.dot(lattice_inv, sim_data.cart_pos[:, atom, time])
                
                if time > 0:
                    prev_frac = sim_data.frac_pos[:, atom, time - 1]
                    frac_diff = current_frac - prev_frac
                    uc -= np.round(frac_diff)  # Unfold coordinates

                sim_data.frac_pos[:, atom, time] = current_frac

                # Calculate total displacement from start, accounting for PBC
                unfolded_cart_pos = sim_data.cart_pos[:, atom, time] + np.dot(sim_data.lattice, uc)
                displacement_vector = unfolded_cart_pos - start_pos
                sim_data.displacement[atom, time] = np.linalg.norm(displacement_vector)

        print("DEBUG: Displacement calculation finished.")
        return sim_data

