"""
Activation Energy Analysis Module
=================================

This module calculates activation energies for ion diffusion using:
- Jump rates from MD simulations at various temperatures
- Effective jump frequencies and attempt frequencies
- Temperature dependence of diffusion coefficients
- Site occupancy corrections from jump analysis

The activation energy (ΔE_i^A) for a type of jump (i) can be determined by:
ΔE_i^A = -k_B T ln(Γ_i^eff / ν*)

where:
- k_B is Boltzmann's constant
- T is temperature in Kelvin  
- Γ_i^eff is the effective jump frequency
- ν* is the attempt frequency
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import linregress
import warnings

class ActivationEnergyAnalyzer:
    """
    Calculates activation energies for ion diffusion jumps.
    
    This class provides methods to:
    - Calculate activation energies from jump frequencies
    - Analyze temperature dependence of jump rates
    - Fit Arrhenius relationships
    - Account for site occupancy effects
    - Integrate with enhanced jump analysis results
    """
    
    def __init__(self, sim_data):
        """
        Initialize the Activation Energy analyzer.
        
        Args:
            sim_data (SimData): The main simulation data object
        """
        self.sim_data = sim_data
        self.k_boltzmann = sim_data.k_boltzmann  # J/K
        self.k_boltzmann_eV = 8.617333e-5  # eV/K (for eV units)
        
        print("\n--- Activation Energy Analysis ---")
        
    def calculate_activation_energy_from_frequencies(self, effective_jump_freq, 
                                                   attempt_freq, temperature,
                                                   occupancy_factor=None):
        """
        Calculate activation energy using the equation from your image:
        ΔE_i^A = -k_B T ln(Γ_i^eff / ν*)
        
        Args:
            effective_jump_freq (float): Effective jump frequency (Hz)
            attempt_freq (float): Attempt frequency (Hz)  
            temperature (float): Temperature (K)
            occupancy_factor (float, optional): Site occupancy correction factor
            
        Returns:
            dict: Activation energy results in different units
        """
        if effective_jump_freq <= 0 or attempt_freq <= 0:
            raise ValueError("Jump frequencies must be positive")
            
        # Apply occupancy correction if provided
        if occupancy_factor is not None and occupancy_factor > 0:
            corrected_jump_freq = effective_jump_freq / occupancy_factor
            print(f"Applied occupancy correction: {occupancy_factor:.4f}")
        else:
            corrected_jump_freq = effective_jump_freq
            occupancy_factor = None
            
        # Calculate frequency ratio
        freq_ratio = corrected_jump_freq / attempt_freq
        
        if freq_ratio <= 0:
            raise ValueError("Frequency ratio must be positive for ln calculation")
            
        # Calculate activation energy
        activation_energy_J = -self.k_boltzmann * temperature * np.log(freq_ratio)
        activation_energy_eV = activation_energy_J / 1.602176e-19  # Convert to eV
        activation_energy_kJ_mol = activation_energy_J * 6.022140857e23 / 1000  # Convert to kJ/mol
        
        results = {
            'activation_energy_J': activation_energy_J,
            'activation_energy_eV': activation_energy_eV, 
            'activation_energy_kJ_mol': activation_energy_kJ_mol,
            'effective_jump_freq': effective_jump_freq,
            'corrected_jump_freq': corrected_jump_freq,
            'attempt_freq': attempt_freq,
            'temperature': temperature,
            'frequency_ratio': freq_ratio,
            'occupancy_factor': occupancy_factor
        }
        
        print(f"Activation Energy Results:")
        print(f"  Effective jump frequency: {effective_jump_freq:.2e} Hz")
        if occupancy_factor:
            print(f"  Corrected jump frequency: {corrected_jump_freq:.2e} Hz")
            print(f"  Occupancy factor: {occupancy_factor:.4f}")
        print(f"  Attempt frequency: {attempt_freq:.2e} Hz")
        print(f"  Temperature: {temperature:.1f} K")
        print(f"  Frequency ratio: {freq_ratio:.2e}")
        print(f"  Activation Energy: {activation_energy_eV:.3f} eV")
        print(f"                   : {activation_energy_kJ_mol:.1f} kJ/mol")
        
        return results
    
    def calculate_site_specific_activation_energies(self, jump_data, sites_cart,
                                                   temperature, attempt_freq,
                                                   site_occupancies=None):
        """
        Calculate activation energies for different jump types between sites.
        
        Args:
            jump_data (np.array): Jump data with [atom_id, from_site, to_site, time]
            sites_cart (np.array): Site coordinates
            temperature (float): Temperature (K)
            attempt_freq (float): Attempt frequency (Hz)
            site_occupancies (dict, optional): Site occupancy fractions
            
        Returns:
            dict: Activation energies for each jump type
        """
        if jump_data.size == 0:
            print("Warning: No jump data available for site-specific analysis")
            return {}
            
        # Count jumps for each site pair
        jump_counts = {}
        total_time = self.sim_data.total_time
        
        for jump in jump_data:
            from_site = int(jump[1])
            to_site = int(jump[2])
            jump_type = f"Site_{from_site-1}->Site_{to_site-1}"  # Convert to 0-based
            
            if jump_type not in jump_counts:
                jump_counts[jump_type] = 0
            jump_counts[jump_type] += 1
        
        # Calculate activation energies for each jump type
        activation_energies = {}
        
        print(f"\nCalculating site-specific activation energies:")
        print(f"Total simulation time: {total_time:.2e} s")
        print(f"Temperature: {temperature:.1f} K")
        print(f"Attempt frequency: {attempt_freq:.2e} Hz")
        
        for jump_type, count in jump_counts.items():
            # Calculate effective jump frequency
            effective_freq = count / total_time
            
            # Apply site occupancy correction if available
            from_site_idx = int(jump_type.split('->')[0].split('_')[1])
            occupancy = site_occupancies.get(from_site_idx, 1.0) if site_occupancies else None
            
            try:
                results = self.calculate_activation_energy_from_frequencies(
                    effective_freq, attempt_freq, temperature, occupancy
                )
                activation_energies[jump_type] = results
                
            except ValueError as e:
                print(f"Warning: Could not calculate activation energy for {jump_type}: {e}")
                continue
        
        return activation_energies
    
    # def calculate_activation_energies_with_occupancy(self, jump_analysis_results, 
    #                                                sites_cart, temperature=None):
    #     """
    #     Calculate activation energies using occupancy factors from jump analysis.
        
    #     Args:
    #         jump_analysis_results: Results from SiteAnalyzerMDA with occupancy data
    #         sites_cart (np.array): Site coordinates
    #         temperature (float, optional): Temperature in K
            
    #     Returns:
    #         dict: Activation energies with occupancy corrections
    #     """
    #     if temperature is None:
    #         temperature = getattr(self.sim_data, 'temperature', 300.0)
        
    #     attempt_freq = getattr(self.sim_data, 'attempt_freq', 1e12)
        
    #     # Get occupancy factors from jump analysis
    #     # FIX: Handle both jump_analysis object and jump_analysis.results object
    #     if hasattr(jump_analysis_results, 'get_jump_specific_occupancy_factors'):
    #         # jump_analysis_results is the main jump_analysis object
    #         occupancy_factors = jump_analysis_results.get_jump_specific_occupancy_factors()
    #         jump_data = jump_analysis_results.results.all_trans
    #     elif hasattr(jump_analysis_results, 'all_trans'):
    #         # jump_analysis_results is the .results object, try to get parent
    #         print("Warning: Passed results object instead of main analysis object")
    #         occupancy_factors = {}
    #         jump_data = jump_analysis_results.all_trans
    #     else:
    #         occupancy_factors = {}
    #         jump_data = jump_analysis_results.all_trans
        
    #     if not occupancy_factors:
    #         print("Warning: No occupancy factors available from jump analysis")
    #         print("Falling back to standard activation energy calculation")
    #         return self.calculate_site_specific_activation_energies(
    #             jump_analysis_results.all_trans, sites_cart, temperature, attempt_freq
    #         )
        
    #     # Count jumps for each site pair
    #     jump_counts = {}
    #     total_time = self.sim_data.total_time
        
    #     for jump in jump_analysis_results.all_trans:
    #         from_site = int(jump[1])
    #         to_site = int(jump[2])
    #         jump_type = f"Site_{from_site-1}->Site_{to_site-1}"  # Convert to 0-based
            
    #         if jump_type not in jump_counts:
    #             jump_counts[jump_type] = 0
    #         jump_counts[jump_type] += 1
        
    #     # Calculate activation energies with occupancy corrections
    #     activation_energies = {}
        
    #     print(f"\nCalculating activation energies with occupancy corrections:")
    #     print(f"Temperature: {temperature:.1f} K")
    #     print(f"Attempt frequency: {attempt_freq:.2e} Hz")
    #     print(f"Total simulation time: {total_time:.2e} s")
        
    #     for jump_type, count in jump_counts.items():
    #         # Calculate effective jump frequency
    #         effective_freq = count / total_time
            
    #         # Get occupancy factor for this jump type
    #         occupancy_factor = occupancy_factors.get(jump_type, 1.0)
            
    #         try:
    #             results = self.calculate_activation_energy_from_frequencies(
    #                 effective_freq, attempt_freq, temperature, occupancy_factor
    #             )
    #             activation_energies[jump_type] = results
                
    #             print(f"  {jump_type}: {results['activation_energy_eV']:.3f} eV "
    #                   f"(occupancy: {occupancy_factor:.4f})")
                
    #         except ValueError as e:
    #             print(f"Warning: Could not calculate activation energy for {jump_type}: {e}")
    #             continue
        
    #     return activation_energies
    def calculate_activation_energies_with_occupancy(self, jump_analysis_results,
                                               sites_cart, temperature=None):
        """
        Calculate activation energies using occupancy factors from jump analysis.
        
        Args:
            jump_analysis_results: Results from SiteAnalyzerMDA with occupancy data
            sites_cart (np.array): Site coordinates
            temperature (float, optional): Temperature in K
            
        Returns:
            dict: Activation energies with occupancy corrections
        """
        print(f"🔍 DEBUG: Calculating activation energies with occupancy")
        print(f"🔍 jump_analysis_results type: {type(jump_analysis_results)}")
        print(f"🔍 Has results attr: {hasattr(jump_analysis_results, 'results')}")
        print(f"🔍 Has all_trans attr: {hasattr(jump_analysis_results, 'all_trans')}")
        
        if temperature is None:
            temperature = getattr(self.sim_data, 'temperature', 300.0)
        attempt_freq = getattr(self.sim_data, 'attempt_freq', 1e12)
        
        # FIX: Correctly access the jump data and occupancy factors
        if hasattr(jump_analysis_results, 'results'):
            # jump_analysis_results is the main SiteAnalyzerMDA object
            jump_data = jump_analysis_results.results.all_trans
            
            # Get occupancy factors
            if hasattr(jump_analysis_results, 'get_jump_specific_occupancy_factors'):
                occupancy_factors = jump_analysis_results.get_jump_specific_occupancy_factors()
                print(f"🔍 Retrieved occupancy factors: {occupancy_factors}")
                print(f"🔍 Number of occupancy factors: {len(occupancy_factors)}")
            else:
                occupancy_factors = {}
                print(f"⚠️ No get_jump_specific_occupancy_factors method found")
                
        elif hasattr(jump_analysis_results, 'all_trans'):
            # jump_analysis_results is the .results object
            jump_data = jump_analysis_results.all_trans
            occupancy_factors = {}
            print(f"⚠️ Working with results object - no occupancy factors available")
        else:
            print("ERROR: Cannot access jump data from provided object")
            return {}
        
        if jump_data.size == 0:
            print("Warning: No jump data available for site-specific analysis")
            return {}
        
        # Count jumps for each site pair
        jump_counts = {}
        total_time = self.sim_data.total_time
        
        for jump in jump_data:
            from_site = int(jump[1])
            to_site = int(jump[2])
            jump_type = f"Site_{from_site-1}->Site_{to_site-1}"  # Convert to 0-based
            
            if jump_type not in jump_counts:
                jump_counts[jump_type] = 0
            jump_counts[jump_type] += 1
        
        # Calculate activation energies with occupancy corrections
        activation_energies = {}
        
        print(f"\nCalculating activation energies with occupancy corrections:")
        print(f"Temperature: {temperature:.1f} K")
        print(f"Attempt frequency: {attempt_freq:.2e} Hz")
        print(f"Total simulation time: {total_time:.2e} s")
        print(f"Found {len(jump_counts)} different jump types")
        if occupancy_factors:
            print(f"\n✅ NOTE: Occupancy correction enabled. Jump frequencies normalized by site occupancy probabilities.")
        
        for jump_type, count in jump_counts.items():
            # Calculate effective jump frequency
            effective_freq = count / total_time
            
            # Get occupancy factor for this jump type
            occupancy_factor = occupancy_factors.get(jump_type, None)
            
            try:
                # Apply occupancy correction: normalizes jump frequency to the starting site's occupancy probability
                results = self.calculate_activation_energy_from_frequencies(
                    effective_freq, attempt_freq, temperature, occupancy_factor=occupancy_factor
                )
                
                # Store the original occupancy factor for reference in CSV output
                results['occupancy_factor'] = occupancy_factor
                
                activation_energies[jump_type] = results
                
                if occupancy_factor is not None:
                    print(f"  {jump_type}: {results['activation_energy_eV']:.3f} eV "
                        f"(occupancy: {occupancy_factor:.4f})")
                else:
                    print(f"  {jump_type}: {results['activation_energy_eV']:.3f} eV "
                        f"(no occupancy correction)")
                    
            except ValueError as e:
                print(f"Warning: Could not calculate activation energy for {jump_type}: {e}")
                continue
        
        return activation_energies
    
    def analyze_temperature_dependence(self, temperatures, jump_rates,
                                     jump_type="Overall", plot_results=True):
        """
        Analyze temperature dependence of jump rates using Arrhenius relationship.
        
        Args:
            temperatures (list): Temperatures (K)
            jump_rates (list): Jump rates at each temperature (Hz)
            jump_type (str): Description of jump type
            plot_results (bool): Whether to create Arrhenius plot
            
        Returns:
            dict: Fitted activation energy and pre-exponential factor
        """
        temperatures = np.array(temperatures)
        jump_rates = np.array(jump_rates)
        
        if len(temperatures) < 3:
            raise ValueError("Need at least 3 temperature points for fitting")
            
        # Remove zero or negative rates
        valid_idx = jump_rates > 0
        temperatures = temperatures[valid_idx]
        jump_rates = jump_rates[valid_idx]
        
        if len(temperatures) < 3:
            raise ValueError("Need at least 3 valid (positive) jump rates")
        
        # Arrhenius equation: rate = A * exp(-Ea / (k_B * T))
        # Linear form: ln(rate) = ln(A) - Ea/(k_B * T)
        
        x = 1000 / temperatures  # 1000/T for plotting in mK^-1
        y = np.log(jump_rates)
        
        # Linear regression
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        
        # Calculate activation energy
        activation_energy_eV = -slope * self.k_boltzmann_eV * 1000  # Convert from mK to K
        pre_exponential = np.exp(intercept)
        
        # Calculate confidence intervals
        n = len(temperatures)
        t_val = 2.262 if n <= 10 else 1.96  # Rough t-value for 95% confidence
        ea_error = abs(slope * std_err * self.k_boltzmann_eV * 1000 * t_val)
        
        results = {
            'activation_energy_eV': activation_energy_eV,
            'activation_energy_kJ_mol': activation_energy_eV * 96.485,
            'pre_exponential': pre_exponential,
            'r_squared': r_value**2,
            'p_value': p_value,
            'activation_energy_error_eV': ea_error,
            'slope': slope,
            'intercept': intercept,
            'temperatures': temperatures,
            'jump_rates': jump_rates
        }
        
        print(f"\nArrhenius Analysis Results for {jump_type}:")
        print(f"  Activation Energy: {activation_energy_eV:.3f} ± {ea_error:.3f} eV")
        print(f"                   : {activation_energy_eV * 96.485:.1f} ± {ea_error * 96.485:.1f} kJ/mol")
        print(f"  Pre-exponential: {pre_exponential:.2e} Hz")
        print(f"  R²: {r_value**2:.4f}")
        print(f"  p-value: {p_value:.2e}")
        
        if plot_results:
            self._plot_arrhenius(x, y, slope, intercept, r_value**2, 
                               jump_type, activation_energy_eV, ea_error)
        
        return results
    
    def _plot_arrhenius(self, x, y, slope, intercept, r_squared, 
                       jump_type, ea_eV, ea_error):
        """Create Arrhenius plot."""
        plt.figure(figsize=(10, 7))
        
        # Plot data points
        plt.scatter(x, y, s=100, c='red', alpha=0.8, edgecolors='darkred', 
                   linewidth=2, label='Simulation data')
        
        # Plot fitted line
        x_fit = np.linspace(min(x), max(x), 100)
        y_fit = slope * x_fit + intercept
        plt.plot(x_fit, y_fit, 'b--', linewidth=2, 
                label=f'Linear fit (R² = {r_squared:.4f})')
        
        # Labels and formatting
        plt.xlabel('1000/T (K⁻¹)', fontsize=12)
        plt.ylabel('ln(Jump Rate)', fontsize=12)
        plt.title(f'Arrhenius Plot - {jump_type}\n'
                 f'Ea = {ea_eV:.3f} ± {ea_error:.3f} eV', fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Add equation text
        eq_text = f'ln(rate) = {intercept:.2f} - {abs(slope):.0f}/T\n'
        eq_text += f'Ea = {ea_eV:.3f} eV'
        plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
                verticalalignment='top', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def calculate_occupancy_corrected_energies(self, jump_data, sites_cart, 
                                             site_volumes=None, total_volume=None):
        """
        Calculate activation energies with site occupancy corrections.
        
        Args:
            jump_data (np.array): Jump data
            sites_cart (np.array): Site coordinates  
            site_volumes (dict, optional): Volume of each site
            total_volume (float, optional): Total simulation volume
            
        Returns:
            dict: Occupancy-corrected activation energies
        """
        if site_volumes is None or total_volume is None:
            print("Warning: No volume data provided, using uniform occupancy")
            site_occupancies = {i: 1.0 for i in range(len(sites_cart))}
        else:
            # Calculate occupancy fractions
            site_occupancies = {i: vol/total_volume 
                              for i, vol in site_volumes.items()}
        
        # Calculate site-specific activation energies with occupancy correction
        temperature = getattr(self.sim_data, 'temperature', 300.0)
        attempt_freq = getattr(self.sim_data, 'attempt_freq', 1e12)
        
        activation_energies = self.calculate_site_specific_activation_energies(
            jump_data, sites_cart, temperature, attempt_freq, site_occupancies
        )
        
        print(f"\nOccupancy-Corrected Activation Energies:")
        for jump_type, results in activation_energies.items():
            print(f"  {jump_type}: {results['activation_energy_eV']:.3f} eV")
            if results['occupancy_factor']:
                print(f"    Occupancy factor: {results['occupancy_factor']:.4f}")
        
        return activation_energies
    
    def compare_activation_energies(self, activation_energy_dict, 
                                  output_file=None, save_data=True):
        """
        Create comparison plot of activation energies for different jump types.
        
        Args:
            activation_energy_dict (dict): Dictionary of activation energies
            output_file (str, optional): Path to save plot
        """
        import traceback
        print(f"🔍 DEBUG: compare_activation_energies called with {len(activation_energy_dict) if activation_energy_dict else 0} entries")
        print(f"🔍 Called from: {traceback.extract_stack()[-2].filename}:{traceback.extract_stack()[-2].lineno}")
        
        if not activation_energy_dict:
            print("⚠️ No activation energy data to plot - skipping visualization")
            return
            
        # Validate data
        valid_entries = {k: v for k, v in activation_energy_dict.items() 
                        if v and 'activation_energy_eV' in v}
        if not valid_entries:
            print("⚠️ No valid activation energy data found - skipping visualization")
            print(f"🔍 Raw data keys: {list(activation_energy_dict.keys())}")
            print(f"🔍 First entry structure: {next(iter(activation_energy_dict.values())) if activation_energy_dict else 'None'}")
            return
            
        print(f"✅ Creating activation energy comparison plot with {len(valid_entries)} valid entries")
            
        jump_types = list(activation_energy_dict.keys())
        energies_eV = [results['activation_energy_eV'] 
                      for results in activation_energy_dict.values()]
        occupancy_factors = [results.get('occupancy_factor', None) 
                           for results in activation_energy_dict.values()]
        
        # Debug occupancy factors
        print(f"🔍 Occupancy factors: {occupancy_factors}")
        print(f"🔍 Non-None occupancy factors: {[occ for occ in occupancy_factors if occ is not None]}")
        
        plt.figure(figsize=(14, 8))
        
        # Create bar plot with color coding for occupancy factors
        colors = []
        has_occupancy_data = any(occ is not None for occ in occupancy_factors)
        
        if has_occupancy_data:
            # Get min/max occupancy for proper color scaling
            valid_occupancies = [occ for occ in occupancy_factors if occ is not None]
            min_occ = min(valid_occupancies) if valid_occupancies else 0
            max_occ = max(valid_occupancies) if valid_occupancies else 1
            print(f"🔍 Occupancy range: {min_occ:.4f} - {max_occ:.4f}")
            
            for occ in occupancy_factors:
                if occ is None:
                    colors.append('lightgray')  # Gray for missing occupancy data
                else:
                    # Normalize occupancy to [0,1] for colormap
                    norm_occ = (occ - min_occ) / (max_occ - min_occ) if max_occ > min_occ else 0.5
                    colors.append(plt.cm.viridis(norm_occ))
        else:
            # Use uniform color if no occupancy data
            colors = ['skyblue'] * len(jump_types)
            print("🔍 No occupancy data found - using uniform coloring")
        
        bars = plt.bar(range(len(jump_types)), energies_eV, 
                      alpha=0.8, color=colors, edgecolor='navy', linewidth=2)
        
        # Add value labels on bars
        for i, (bar, energy, occ) in enumerate(zip(bars, energies_eV, occupancy_factors)):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{energy:.3f} eV', ha='center', va='bottom', fontweight='bold')
            if occ is not None:
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                        f'occ: {occ:.3f}', ha='center', va='center', 
                        fontsize=9, color='white', fontweight='bold')
        
        plt.xlabel('Jump Type', fontsize=12)
        plt.ylabel('Activation Energy (eV)', fontsize=12)
        
        if has_occupancy_data:
            plt.title('Activation Energies for Different Jump Types\n(Bar colors show occupancy factors)', fontsize=14)
        else:
            plt.title('Activation Energies for Different Jump Types', fontsize=14)
            
        plt.xticks(range(len(jump_types)), jump_types, rotation=45, ha='right')
        plt.grid(True, alpha=0.3, axis='y')
        
        # Add colorbar if occupancy factors are present
        if has_occupancy_data:
            try:
                valid_occupancies = [occ for occ in occupancy_factors if occ is not None]
                min_occ = min(valid_occupancies)
                max_occ = max(valid_occupancies)
                
                sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis, 
                                         norm=plt.Normalize(vmin=min_occ, vmax=max_occ))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=plt.gca())
                cbar.set_label('Occupancy Factor', fontsize=12)
                print(f"✅ Added colorbar for occupancy range: {min_occ:.4f} - {max_occ:.4f}")
            except Exception as e:
                print(f"⚠️ Could not add colorbar: {e}")
                print("   Continuing without colorbar...")
        
        # Only add legend if we have meaningful legend entries
        if has_occupancy_data:
            # Create legend entries for occupancy info
            legend_elements = []
            if any(occ is None for occ in occupancy_factors):
                legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='lightgray', 
                                                   label='No occupancy data'))
            if legend_elements:
                plt.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            print(f"Activation energy comparison saved: {output_file}")
        
        # Save data if requested
        if save_data:
            self._save_activation_energy_data(activation_energy_dict, jump_types, 
                                            energies_eV, occupancy_factors, 
                                            output_file)
        
        plt.show()
    
    def generate_activation_energy_report(self, activation_energies, output_file=None):
        """
        Generate a comprehensive activation energy analysis report.
        
        Args:
            activation_energies (dict): Dictionary of activation energy results
            output_file (str, optional): Path to save report
        """
        if not activation_energies:
            print("No activation energy data for report generation")
            return
        
        report_lines = []
        report_lines.append("="*80)
        report_lines.append("ACTIVATION ENERGY ANALYSIS REPORT")
        report_lines.append("="*80)
        report_lines.append(f"Generated for {len(activation_energies)} jump types")
        report_lines.append(f"Temperature: {getattr(self.sim_data, 'temperature', 300.0):.1f} K")
        report_lines.append(f"Attempt frequency: {getattr(self.sim_data, 'attempt_freq', 1e12):.2e} Hz")
        report_lines.append("")
        
        # Individual jump type results
        report_lines.append("INDIVIDUAL JUMP TYPE RESULTS:")
        report_lines.append("-" * 50)
        
        energies_eV = []
        for jump_type, results in activation_energies.items():
            energy_eV = results['activation_energy_eV']
            energy_kJ = results['activation_energy_kJ_mol']
            effective_freq = results['effective_jump_freq']
            occupancy = results.get('occupancy_factor', None)
            
            energies_eV.append(energy_eV)
            
            report_lines.append(f"\n{jump_type}:")
            report_lines.append(f"  Activation Energy: {energy_eV:.3f} eV ({energy_kJ:.1f} kJ/mol)")
            report_lines.append(f"  Effective Jump Frequency: {effective_freq:.2e} Hz")
            if occupancy is not None:
                report_lines.append(f"  Occupancy Factor: {occupancy:.4f}")
                corrected_freq = results.get('corrected_jump_freq', effective_freq)
                report_lines.append(f"  Corrected Jump Frequency: {corrected_freq:.2e} Hz")
            
        # Statistical summary
        report_lines.append(f"\n\nSTATISTICAL SUMMARY:")
        report_lines.append("-" * 30)
        report_lines.append(f"Average Activation Energy: {np.mean(energies_eV):.3f} ± {np.std(energies_eV):.3f} eV")
        report_lines.append(f"Range: {np.min(energies_eV):.3f} - {np.max(energies_eV):.3f} eV")
        report_lines.append(f"Median: {np.median(energies_eV):.3f} eV")
        
        # Check for occupancy corrections
        occupancy_corrected = sum(1 for results in activation_energies.values() 
                                if results.get('occupancy_factor') is not None)
        report_lines.append(f"Jump types with occupancy corrections: {occupancy_corrected}/{len(activation_energies)}")
        
        report_lines.append("\n" + "="*80)
        
        # Print report
        report_text = "\n".join(report_lines)
        print(report_text)
        
        # Save report if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)
            print(f"\nActivation energy report saved: {output_file}")
        
        return report_text
    
    def _save_activation_energy_data(self, activation_energy_dict, jump_types, 
                                   energies_eV, occupancy_factors, output_file):
        """
        Save comprehensive activation energy data to multiple file formats.
        
        Args:
            activation_energy_dict (dict): Full activation energy data
            jump_types (list): List of jump type names
            energies_eV (list): Activation energies in eV
            occupancy_factors (list): Occupancy factors for each jump type
            output_file (str, optional): Base filename for outputs
        """
        import json
        import pandas as pd
        from datetime import datetime
        
        # Determine base filename
        if output_file:
            base_name = output_file.rsplit('.', 1)[0]  # Remove extension
        else:
            base_name = 'activation_energy_analysis'
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # 1. Save comprehensive JSON data
            json_filename = f"{base_name}_data_{timestamp}.json"
            
            # Prepare comprehensive data structure
            comprehensive_data = {
                'metadata': {
                    'analysis_type': 'activation_energy_analysis',
                    'timestamp': datetime.now().isoformat(),
                    'temperature': getattr(self.sim_data, 'temperature', 300.0),
                    'attempt_frequency': getattr(self.sim_data, 'attempt_freq', 1e12),
                    'total_simulation_time': self.sim_data.total_time,
                    'n_jump_types': len(jump_types)
                },
                'summary_statistics': {
                    'average_activation_energy_eV': float(np.mean(energies_eV)),
                    'std_activation_energy_eV': float(np.std(energies_eV)),
                    'min_activation_energy_eV': float(np.min(energies_eV)),
                    'max_activation_energy_eV': float(np.max(energies_eV)),
                    'median_activation_energy_eV': float(np.median(energies_eV)),
                    'n_with_occupancy_data': sum(1 for occ in occupancy_factors if occ is not None)
                },
                'jump_type_data': {}
            }
            
            # Add individual jump type data
            for i, (jump_type, results) in enumerate(activation_energy_dict.items()):
                # Extract site indices from jump type name
                try:
                    from_site_str, to_site_str = jump_type.split('->')
                    from_site = int(from_site_str.split('_')[1])
                    to_site = int(to_site_str.split('_')[1])
                except:
                    from_site, to_site = None, None
                
                comprehensive_data['jump_type_data'][jump_type] = {
                    'site_info': {
                        'from_site_index': from_site,
                        'to_site_index': to_site,
                        'jump_type_name': jump_type
                    },
                    'energetics': {
                        'activation_energy_eV': float(results['activation_energy_eV']),
                        'activation_energy_kJ_mol': float(results['activation_energy_kJ_mol']),
                        'activation_energy_J': float(results['activation_energy_J'])
                    },
                    'frequencies': {
                        'effective_jump_freq_Hz': float(results['effective_jump_freq']),
                        'corrected_jump_freq_Hz': float(results['corrected_jump_freq']),
                        'attempt_freq_Hz': float(results['attempt_freq']),
                        'frequency_ratio': float(results['frequency_ratio'])
                    },
                    'occupancy': {
                        'occupancy_factor': float(occupancy_factors[i]) if occupancy_factors[i] is not None else None,
                        'has_occupancy_correction': occupancy_factors[i] is not None
                    },
                    'simulation_conditions': {
                        'temperature_K': float(results['temperature'])
                    }
                }
            
            # Save JSON data
            with open(json_filename, 'w') as f:
                json.dump(comprehensive_data, f, indent=2)
            print(f"📊 Comprehensive data saved: {json_filename}")
            
            # 2. Save CSV summary for easy analysis
            csv_filename = f"{base_name}_summary_{timestamp}.csv"
            
            # Prepare CSV data
            csv_data = []
            for i, (jump_type, results) in enumerate(activation_energy_dict.items()):
                # Extract site information
                try:
                    from_site_str, to_site_str = jump_type.split('->')
                    from_site = int(from_site_str.split('_')[1])
                    to_site = int(to_site_str.split('_')[1])
                except:
                    from_site, to_site = -1, -1
                
                csv_row = {
                    'Jump_Type': jump_type,
                    'From_Site': from_site,
                    'To_Site': to_site,
                    'Activation_Energy_eV': results['activation_energy_eV'],
                    'Activation_Energy_kJ_mol': results['activation_energy_kJ_mol'],
                    'Effective_Jump_Freq_Hz': results['effective_jump_freq'],
                    'Corrected_Jump_Freq_Hz': results['corrected_jump_freq'],
                    'Occupancy_Factor': occupancy_factors[i] if occupancy_factors[i] is not None else np.nan,
                    'Has_Occupancy_Data': occupancy_factors[i] is not None,
                    'Temperature_K': results['temperature'],
                    'Attempt_Freq_Hz': results['attempt_freq'],
                    'Frequency_Ratio': results['frequency_ratio']
                }
                csv_data.append(csv_row)
            
            # Save CSV
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False)
            print(f"📈 Summary CSV saved: {csv_filename}")
            
            # 3. Save plotting data for reproduction
            plot_data_filename = f"{base_name}_plot_data_{timestamp}.json"
            
            plot_data = {
                'metadata': {
                    'plot_type': 'activation_energy_comparison',
                    'timestamp': datetime.now().isoformat(),
                    'n_bars': len(jump_types)
                },
                'plot_data': {
                    'jump_types': jump_types,
                    'activation_energies_eV': energies_eV,
                    'occupancy_factors': [float(occ) if occ is not None else None for occ in occupancy_factors],
                    'has_occupancy_data': any(occ is not None for occ in occupancy_factors)
                },
                'color_mapping': {
                    'colormap': 'viridis',
                    'occupancy_range': {
                        'min': float(min([occ for occ in occupancy_factors if occ is not None])) if any(occ is not None for occ in occupancy_factors) else None,
                        'max': float(max([occ for occ in occupancy_factors if occ is not None])) if any(occ is not None for occ in occupancy_factors) else None
                    }
                }
            }
            
            with open(plot_data_filename, 'w') as f:
                json.dump(plot_data, f, indent=2)
            print(f"🎨 Plot data saved: {plot_data_filename}")
            
            # 4. Save site coordinate information if available
            if hasattr(self.sim_data, 'sites_cart'):
                sites_filename = f"{base_name}_sites_{timestamp}.csv"
                
                # Create site information DataFrame
                site_data = []
                for i, coord in enumerate(self.sim_data.sites_cart):
                    site_data.append({
                        'Site_Index': i,
                        'X_coord': coord[0],
                        'Y_coord': coord[1],
                        'Z_coord': coord[2]
                    })
                
                site_df = pd.DataFrame(site_data)
                site_df.to_csv(sites_filename, index=False)
                print(f"🗺️ Site coordinates saved: {sites_filename}")
            
            print(f"\n✅ All activation energy data files saved with timestamp: {timestamp}")
            print(f"   Base filename: {base_name}")
            print(f"   Files saved:")
            print(f"     - {json_filename} (comprehensive JSON data)")
            print(f"     - {csv_filename} (summary CSV for analysis)")
            print(f"     - {plot_data_filename} (plot reproduction data)")
            if hasattr(self.sim_data, 'sites_cart'):
                print(f"     - {sites_filename} (site coordinates)")
                
        except Exception as e:
            print(f"⚠️ Error saving activation energy data: {e}")
            print("   Continuing without data export...")
    
    def save_site_occupancy_data(self, jump_analysis_results, output_file=None):
        """
        Save site occupancy data from jump analysis results.
        
        Args:
            jump_analysis_results: Jump analysis object with occupancy data
                                 Can be:
                                 - Main JumpAnalyzer object (has .results attribute)
                                 - JumpAnalysis results object (has .all_trans attribute)
                                 - Any object with get_jump_specific_occupancy_factors method
            output_file (str, optional): Output filename
        
        Example usage:
            # If your jump analysis is stored in 'jump_analysis'
            activation_analyzer.save_site_occupancy_data(jump_analysis)
            
            # If you have the results object directly
            activation_analyzer.save_site_occupancy_data(jump_analysis.results)
        """
        import json
        import pandas as pd
        from datetime import datetime
        
        if output_file:
            base_name = output_file.rsplit('.', 1)[0]
        else:
            base_name = 'site_occupancy_data'
            
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        try:
            # Get occupancy data
            if hasattr(jump_analysis_results, 'get_jump_specific_occupancy_factors'):
                occupancy_factors = jump_analysis_results.get_jump_specific_occupancy_factors()
            else:
                print("⚠️ No occupancy factor method found in jump analysis results")
                return
                
            if not occupancy_factors:
                print("⚠️ No occupancy factors available to save")
                return
            
            # Save occupancy data
            occupancy_filename = f"{base_name}_occupancy_{timestamp}.json"
            
            occupancy_data = {
                'metadata': {
                    'data_type': 'site_occupancy_factors',
                    'timestamp': datetime.now().isoformat(),
                    'n_jump_types': len(occupancy_factors)
                },
                'occupancy_factors': occupancy_factors
            }
            
            with open(occupancy_filename, 'w') as f:
                json.dump(occupancy_data, f, indent=2)
                
            print(f"📊 Site occupancy data saved: {occupancy_filename}")
            
            # Also save as CSV for easy viewing
            csv_filename = f"{base_name}_occupancy_{timestamp}.csv"
            
            csv_data = []
            for jump_type, occupancy in occupancy_factors.items():
                try:
                    from_site_str, to_site_str = jump_type.split('->')
                    from_site = int(from_site_str.split('_')[1])
                    to_site = int(to_site_str.split('_')[1])
                except:
                    from_site, to_site = -1, -1
                
                csv_data.append({
                    'Jump_Type': jump_type,
                    'From_Site': from_site,
                    'To_Site': to_site,
                    'Occupancy_Factor': occupancy
                })
            
            df = pd.DataFrame(csv_data)
            df.to_csv(csv_filename, index=False)
            print(f"📈 Occupancy CSV saved: {csv_filename}")
            
        except Exception as e:
            print(f"⚠️ Error saving occupancy data: {e}")
    
    @staticmethod
    def find_jump_analysis_variables():
        """
        Diagnostic method to help find jump analysis variables in the current scope.
        
        This method inspects the global namespace to find potential jump analysis objects.
        
        Returns:
            list: Names of potential jump analysis variables
        """
        import sys
        import inspect
        
        # Get the caller's frame to access their local variables
        caller_frame = inspect.currentframe().f_back
        caller_locals = caller_frame.f_locals
        caller_globals = caller_frame.f_globals
        
        potential_variables = []
        
        print("🔍 Searching for jump analysis variables...")
        print("\n--- Checking local variables ---")
        
        # Check local variables first
        for name, obj in caller_locals.items():
            if hasattr(obj, 'results') and hasattr(getattr(obj, 'results', None), 'all_trans'):
                potential_variables.append(f"{name} (JumpAnalyzer object - use: {name})")
                print(f"✅ Found: {name} (JumpAnalyzer object)")
            elif hasattr(obj, 'all_trans'):
                potential_variables.append(f"{name} (Jump results object - use: {name})")
                print(f"✅ Found: {name} (Jump results object)")
            elif hasattr(obj, 'get_jump_specific_occupancy_factors'):
                potential_variables.append(f"{name} (Has occupancy methods - use: {name})")
                print(f"✅ Found: {name} (Has occupancy methods)")
        
        print("\n--- Checking global variables ---")
        
        # Check global variables
        for name, obj in caller_globals.items():
            if name.startswith('_'):  # Skip private variables
                continue
            if hasattr(obj, 'results') and hasattr(getattr(obj, 'results', None), 'all_trans'):
                if f"{name} (JumpAnalyzer object - use: {name})" not in potential_variables:
                    potential_variables.append(f"{name} (JumpAnalyzer object - use: {name})")
                    print(f"✅ Found: {name} (JumpAnalyzer object)")
            elif hasattr(obj, 'all_trans'):
                if f"{name} (Jump results object - use: {name})" not in potential_variables:
                    potential_variables.append(f"{name} (Jump results object - use: {name})")
                    print(f"✅ Found: {name} (Jump results object)")
            elif hasattr(obj, 'get_jump_specific_occupancy_factors'):
                if f"{name} (Has occupancy methods - use: {name})" not in potential_variables:
                    potential_variables.append(f"{name} (Has occupancy methods - use: {name})")
                    print(f"✅ Found: {name} (Has occupancy methods)")
        
        if not potential_variables:
            print("⚠️ No jump analysis variables found in current scope.")
            print("\nCommon variable names to check:")
            print("  - jump_analysis")
            print("  - jump_analyzer")
            print("  - analyzer")
            print("  - results")
            print("  - ja")
            print("\nMake sure you have run jump analysis first:")
            print("  jump_analysis = JumpAnalyzer(...)")
            print("  jump_analysis.run()")
        
        return potential_variables

