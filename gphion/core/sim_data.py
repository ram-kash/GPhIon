"""Data holding class for simulation data and parameters."""

class SimData:
    """Holds raw simulation data and parameters after reading the trajectory."""

    def __init__(self):
        # Physical constants for calculations.
        self.e_charge = 1.60217657e-19  # Electron charge (C)
        self.k_boltzmann = 1.3806488e-23  # Boltzmann's constant (J/K)
        self.avogadro = 6.022140857e23  # Avogadro's number (mol^-1)

        # Simulation parameters
        self.time_step = None
        self.total_time = 0.0
        self.nr_atoms = 0
        self.nr_diffusing = 0
        self.diffusing_atom = None
        self.diffusing_indices = None  # Original indices of diffusing atoms in full system
        self.nr_steps = 0
        self.cart_pos = None
        self.frac_pos = None
        self.displacement = None
        self.universe = None
        self.lattice = None
        self.diff_elem = None
        
        # Analysis results
        self.attempt_freq = 0.0
        self.vibration_amp = 0.0
        self.std_attempt_freq = 0.0
        
        # Additional properties
        self.temperature = None
        self.volume = None
        self.diffusion_dim = None
        self.ion_charge = None

