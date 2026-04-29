"""Jump detection and analysis between discovered sites with occupancy tracking."""

import numpy as np
from MDAnalysis.analysis.base import AnalysisBase

class JumpAnalyzer(AnalysisBase):
    """Finds atomic jumps based on a given set of site coordinates and tracks site occupancy.
    
    The site radius is automatically set to 2.0 * mean atomic vibration amplitude, 
    which defines the sphere around each discovered site.
    
    References: https://docs.mdanalysis.org/2.9.0/documentation_pages/analysis/base.html"""
    
    def __init__(self, diffusing_atom_group, site_coordinates_cart, vibration_amp=None, **kwargs):
        super().__init__(diffusing_atom_group.universe.trajectory, **kwargs)
        
        self._ag = diffusing_atom_group
        # The discovered_sites_cart are stored here
        self.site_cart_pos = site_coordinates_cart
        self.site_radius = None
        self.vibration_amp = vibration_amp 
        self._previous_sites = np.zeros(len(self._ag), dtype=int)
        
        # Initialize n_frames properly from trajectory
        self.n_frames = self._trajectory.n_frames
        
        # Initialize occupancy tracking
        self.n_sites = len(site_coordinates_cart)
        self.site_occupancy_frames = np.zeros((self.n_frames, self.n_sites))
        self.site_residence_times = {i: [] for i in range(self.n_sites)}
        self.current_residence_start = np.full(len(self._ag), -1)  # Track when atoms entered sites
        
    def _prepare(self):
        
        self.frame_gap = 1  
        
        if self.n_frames > 1:
            first_frame_time = self._trajectory[0].time
            second_frame_time = self._trajectory[1].time
            actual_frame_gap = second_frame_time - first_frame_time
            
            if actual_frame_gap != 0:
                self.frame_gap = actual_frame_gap / self._trajectory.dt
            else:
                self.frame_gap = 1
        
        self.new_dt = self._trajectory.dt / self.frame_gap
        
        print(f"Calculated frame gap: {self.frame_gap}")
        print(f"Time step used (dt): {self.new_dt} ps")
        
        self._trajectory.rewind()
        
        # Calculate vibration_amp if not provided, and set site_radius
        if self.vibration_amp is None:
            self._trajectory.rewind()
            # Need at least two frames to calculate vibration
            if self.n_frames >= 2:
                pos0 = self._ag.positions.copy()
                self._trajectory[1]
                pos1 = self._ag.positions.copy()
                self._trajectory.rewind()
                self.vibration_amp = np.mean(np.linalg.norm(pos1 - pos0, axis=1))
            else:
                # Fallback value if only one frame is present
                self.vibration_amp = 0.2 
                print("WARNING: Only one frame available. Using default vibration_amp of 0.2 Å.")
            
            print(f"DEBUG: Calculated/Assumed vibration_amp: {self.vibration_amp:.4f} Å")
        else:
            print(f"DEBUG: Using supplied vibration_amp: {self.vibration_amp:.4f} Å")
        
        # Define the sphere radius: 2.0 * vibration_amp
        if self.site_radius is None:
            self.site_radius = 2.0 * self.vibration_amp
            print(f"DEBUG: site_radius set internally to {self.site_radius:.4f} Å")
        else:
            print(f"DEBUG: site_radius supplied externally: {self.site_radius:.4f} Å")
        
        self.results.all_trans = []
        
        # Initialize occupancy results
        self.results.site_occupancies = np.zeros(self.n_sites)
        self.results.average_occupancy_factors = np.zeros(self.n_sites)
        self.results.occupancy_time_series = np.zeros((self.n_frames, self.n_sites))

    def _single_frame(self):
        frame_idx = self._trajectory.frame
        current_sites = np.zeros(len(self._ag), dtype=int)
        close_sqrd = self.site_radius**2
        
        # Calculate current time based on frame index and original timestep
        current_time = (frame_idx * self._trajectory.dt)
        
        frame_occupancy = np.zeros(self.n_sites)
        
        # Site Mapping Logic: Check if the atom is within the sphere of any site
        for i, atom_pos in enumerate(self._ag.positions):
            distances_sq = np.sum((self.site_cart_pos - atom_pos)**2, axis=1)
            min_dist_sq = np.min(distances_sq)
            
            if min_dist_sq < close_sqrd:
                found_site_idx = np.argmin(distances_sq)
                current_sites[i] = found_site_idx + 1  # 1-based index (Site 1 to N_sites)
                
                # Track occupancy
                frame_occupancy[found_site_idx] += 1
                
                # Track residence time start
                if self.current_residence_start[i] == -1:
                    self.current_residence_start[i] = current_time
            else:
                # Atom left site, record residence time
                if self.current_residence_start[i] != -1:
                    prev_site_idx = self._previous_sites[i] - 1  # Convert to 0-based
                    if 0 <= prev_site_idx < self.n_sites:
                        residence_time = current_time - self.current_residence_start[i]
                        self.site_residence_times[prev_site_idx].append(residence_time)
                    self.current_residence_start[i] = -1
        
        # Store frame occupancy
        self.site_occupancy_frames[frame_idx] = frame_occupancy
        self.results.occupancy_time_series[frame_idx] = frame_occupancy
        
        # Log jumps: prev_site > 0 (was in a site), curr_site > 0 (is in a site), curr_site != prev_site (jumped)
        for i in range(len(self._ag)):
            prev_site, curr_site = self._previous_sites[i], current_sites[i]
            if prev_site > 0 and curr_site > 0 and curr_site != prev_site:
                # Log [atom_index, from_site (1-based), to_site (1-based), jump_time]
                self.results.all_trans.append([i, prev_site, curr_site, current_time])
        
        self._previous_sites = current_sites

    def _conclude(self):
        self.results.all_trans = np.array(self.results.all_trans)
        
        # Calculate final occupancy statistics
        self._calculate_occupancy_factors()
        
        print(f"DEBUG: Jump analysis complete with occupancy tracking")
        print(f" Total jumps detected: {len(self.results.all_trans)}")
        print(f" Average site occupancies: {self.results.site_occupancies}")
        print(f" Used timestep (normalized): {self._trajectory.dt / self.frame_gap:.4f} ps")

    def _calculate_occupancy_factors(self):
        """Calculate occupancy factors for each site."""
        total_frames = self.n_frames
        n_diffusing_atoms = len(self._ag)
        
        # Calculate time-averaged occupancy for each site
        for site_idx in range(self.n_sites):
            avg_occupancy = np.mean(self.site_occupancy_frames[:, site_idx])
            self.results.site_occupancies[site_idx] = avg_occupancy
            
            # Occupancy factor: probability that the site is occupied (max occupancy is 1)
            self.results.average_occupancy_factors[site_idx] = avg_occupancy
        
        # Calculate residence times statistics
        self.results.residence_time_stats = {}
        for site_idx in range(self.n_sites):
            if self.site_residence_times[site_idx]:
                times = np.array(self.site_residence_times[site_idx])
                self.results.residence_time_stats[site_idx] = {
                    'mean': np.mean(times),
                    'std': np.std(times),
                    'min': np.min(times),
                    'max': np.max(times),
                    'count': len(times)
                }
            else:
                self.results.residence_time_stats[site_idx] = {
                    'mean': 0, 'std': 0, 'min': 0, 'max': 0, 'count': 0
                }

    def get_site_occupancy_factor(self, site_index):
        """Get occupancy factor for a specific site (0-based index)."""
        if hasattr(self.results, 'average_occupancy_factors'):
            return self.results.average_occupancy_factors[site_index]
        else:
            print("Warning: Occupancy factors not calculated. Run analysis first.")
            return 1.0

    def get_jump_specific_occupancy_factors(self):
        """Get occupancy factors for each jump type."""
        if not hasattr(self.results, 'average_occupancy_factors'):
            print("Warning: Occupancy factors not calculated.")
            return {}
        
        occupancy_factors = {}
        if self.results.all_trans.size > 0:
            for jump in self.results.all_trans:
                from_site = int(jump[1]) - 1 
                to_site = int(jump[2]) - 1
                jump_type = f"Site_{from_site}->Site_{to_site}"
                
                # Use occupancy factor of the origin site (where the jump starts)
                if 0 <= from_site < self.n_sites:
                    occupancy_factors[jump_type] = self.results.average_occupancy_factors[from_site]
                else:
                    occupancy_factors[jump_type] = 1.0
        
        return occupancy_factors

    def print_occupancy_statistics(self):
        """Print detailed occupancy statistics."""
        if not hasattr(self.results, 'site_occupancies'):
            print("No occupancy data available. Run analysis first.")
            return
        
        print("\n" + "="*60)
        print("SITE OCCUPANCY ANALYSIS")
        print("="*60)
        
        for site_idx in range(self.n_sites):
            avg_occ = self.results.site_occupancies[site_idx]
            occ_factor = self.results.average_occupancy_factors[site_idx]
            
            print(f"\nSite {site_idx} (Site_{site_idx}):")
            print(f"  Average Occupancy: {avg_occ:.3f} atoms")
            print(f"  Occupancy Factor: {occ_factor:.4f}")
            
            if site_idx in self.results.residence_time_stats:
                stats = self.results.residence_time_stats[site_idx]
                if stats['count'] > 0:
                    print(f"  Residence Time: {stats['mean']:.2f} ± {stats['std']:.2f} ps")
                    print(f"  Residence Events: {stats['count']}")
                else:
                    print(f"  No residence events recorded")
        
        print(f"\nOverall Statistics:")
        print(f"  Total Sites: {self.n_sites}")
        print(f"  Average Occupancy Factor: {np.mean(self.results.average_occupancy_factors):.4f}")
        print(f"  Most Occupied Site: Site_{np.argmax(self.results.site_occupancies)} "
              f"({np.max(self.results.site_occupancies):.3f} atoms)")
        print(f"  Least Occupied Site: Site_{np.argmin(self.results.site_occupancies)} "
              f"({np.min(self.results.site_occupancies):.3f} atoms)")
        print(f"  Normalized timestep: {self._trajectory.dt / self.frame_gap:.4f} ps")