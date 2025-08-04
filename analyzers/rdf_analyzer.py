"""Radial Distribution Function (RDF) analysis."""

import numpy as np
import matplotlib.pyplot as plt
import MDAnalysis as mda
from MDAnalysis.analysis.rdf import InterRDF

class RDFData:
    """A data-holding class for RDF results."""

    def __init__(self):
        self.distributions = {}  # Site-specific RDFs
        self.integrated = {}  # Integrated site-specific RDFs (coordination numbers)
        self.total_rdf = {}  # Overall RDF from all diffusing ions
        self.elements = []
        self.rdf_names = []
        self.max_dist = 0
        self.resolution = 0
        self.bins = None
        self.nbins = 0

class RDFAnalyzer:
    """Calculates total and site-specific Radial Distribution Functions (RDFs)."""

    def __init__(self, sim_data, site_coordinates_cart, site_radius):
        """
        Args:
            sim_data (SimData): The main simulation data object.
            site_coordinates_cart (np.ndarray): Cartesian coordinates of discovered sites.
            site_radius (float): The radius defining a site's boundary.
        """
        self.sim_data = sim_data
        self.sites_cart = site_coordinates_cart
        self.site_radius_sq = site_radius**2
        self.u = sim_data.universe
        self.diffusing_atoms_group = self.u.select_atoms(f"element {sim_data.diff_elem}")
        print("\n--- Stage 5: Preparing for RDF Analysis ---")

    def _map_atoms_to_sites_over_time(self):
        """Determine which site each diffusing atom occupies at every frame."""
        print("DEBUG: Mapping all diffusing atoms to sites for every frame...")
        
        num_diff_atoms = len(self.diffusing_atoms_group)
        frame_to_site_map = {}

        for ts in self.u.trajectory:
            atom_to_site = np.full(num_diff_atoms, -1, dtype=int)
            
            for i, atom_pos in enumerate(self.diffusing_atoms_group.positions):
                distances_sq = np.sum((self.sites_cart - atom_pos)**2, axis=1)
                if np.min(distances_sq) < self.site_radius_sq:
                    atom_to_site[i] = np.argmin(distances_sq)

            frame_to_site_map[ts.frame] = atom_to_site

        print("DEBUG: Atom-to-site mapping complete.")
        return frame_to_site_map

    def calculate_rdfs(self, resolution=0.1, max_dist=10.0, show_plot=True):
        """
        Calculate all RDFs using the efficient MDAnalysis InterRDF module.
        
        Args:
            resolution (float): Bin width for the RDF in Angstroms.
            max_dist (float): Maximum distance for the RDF in Angstroms.
            show_plot (bool): If True, displays plots of the calculated RDFs.
            
        Returns:
            RDFData: An object containing all RDF results.
        """
        # 1. Setup RDF parameters
        rdf_results = RDFData()
        rdf_results.resolution = resolution
        rdf_results.max_dist = max_dist
        rdf_results.nbins = int(max_dist / resolution)
        rdf_results.bins = np.linspace(0, max_dist, rdf_results.nbins + 1)

        all_elements = np.unique(self.u.atoms.elements)
        rdf_results.elements = [el for el in all_elements if el != self.sim_data.diff_elem]

        # 2. Map atoms to sites for all frames
        atom_site_map = self._map_atoms_to_sites_over_time()

        # 3. Calculate Overall RDF
        print("Calculating overall RDF for all diffusing atoms...")
        for element in rdf_results.elements:
            pair_group = self.u.select_atoms(f"element {element}")
            rdf = InterRDF(self.diffusing_atoms_group, pair_group, 
                          nbins=rdf_results.nbins, range=(0.0, max_dist))
            rdf.run()
            rdf_results.total_rdf[f"{self.sim_data.diff_elem}-{element}"] = rdf.results.rdf

        # 4. Calculate Site-Specific RDFs
        print("Calculating site-specific RDFs...")
        num_sites = len(self.sites_cart)
        rdf_results.rdf_names = [f"Site_{i}" for i in range(num_sites)]

        for site_idx in range(num_sites):
            site_name = rdf_results.rdf_names[site_idx]
            print(f" - Processing {site_name}...")

            # Collect atom indices that are in this site AT ANY TIME
            atom_indices_in_site = [
                self.diffusing_atoms_group.indices[i]
                for frame_map in atom_site_map.values()
                for i, s_idx in enumerate(frame_map) if s_idx == site_idx
            ]

            if not atom_indices_in_site:
                print(f" WARNING: No atoms ever occupied {site_name}. Skipping.")
                continue

            # Create a temporary AtomGroup of these atoms
            center_group = self.u.atoms[list(set(atom_indices_in_site))]
            rdf_results.distributions[site_name] = {}

            for element in rdf_results.elements:
                pair_group = self.u.select_atoms(f"element {element}")
                rdf = InterRDF(center_group, pair_group, 
                              nbins=rdf_results.nbins, range=(0.0, max_dist))
                rdf.run()
                rdf_results.distributions[site_name][f"{self.sim_data.diff_elem}-{element}"] = rdf.results.rdf

        # 5. Integrate RDFs to get coordination numbers
        for site_name, rdfs in rdf_results.distributions.items():
            rdf_results.integrated[site_name] = {}
            for pair_name, g_r in rdfs.items():
                # Coordination number N = 4 * pi * rho * integral(r^2 * g(r) dr)
                rdf_results.integrated[site_name][pair_name] = np.cumsum(g_r) * rdf_results.resolution

        if show_plot:
            self.plot_rdfs(rdf_results)

        return rdf_results

    def plot_rdfs(self, rdf_results):
        """Visualize the calculated RDFs."""
        print("Plotting RDFs...")
        bin_centers = (rdf_results.bins[:-1] + rdf_results.bins[1:]) / 2

        # Plot Overall RDF
        plt.figure(figsize=(12, 7))
        for pair_name, g_r in rdf_results.total_rdf.items():
            plt.plot(bin_centers, g_r, label=pair_name)
        plt.title(f'Overall Radial Distribution Function from {self.sim_data.diff_elem}')
        plt.xlabel('Distance (Å)')
        plt.ylabel('g(r)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.show()

        # Plot Site-Specific RDFs
        for site_name, rdfs in rdf_results.distributions.items():
            plt.figure(figsize=(12, 7))
            for pair_name, g_r in rdfs.items():
                plt.plot(bin_centers, g_r, label=pair_name)
            plt.title(f'Site-Specific RDF for {site_name}')
            plt.xlabel('Distance (Å)')
            plt.ylabel('g(r)')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.show()

