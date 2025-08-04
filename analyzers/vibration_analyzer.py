"""Analysis of atomic vibrations and attempt frequency."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import signal
import os
from ..core.utils import smooth

class VibrationAnalyzer:
    """Analyzes atomic vibrations to find attempt frequency and vibration amplitude."""

    def __init__(self):
        """Initialize the VibrationAnalyzer."""
        pass

    def _meanfreq(self, signal_data, fs):
        """Calculate the mean frequency (spectral centroid) of a signal."""
        # Calculate the Power Spectral Density (PSD)
        freqs, psd = signal.periodogram(signal_data, fs)
        
        # Calculate the spectral centroid
        if np.sum(psd) > 0:
            return np.sum(freqs * psd) / np.sum(psd)
        else:
            return 0

    def vibration_properties(self, sim_data, show_pics=False, save_data=False, output_dir=None):
        """
        Calculate attempt frequency and vibration amplitude from simulation data.
        
        Args:
            sim_data (SimData): The populated simulation data object.
            show_pics (bool): If True, display plots of the results.
            save_data (bool): If True, save vibrational data to files.
            output_dir (str, optional): Directory to save data and plots.
            
        Returns:
            tuple: (attempt_freq, vibration_amp, std_attempt_freq)
        """
        # Initialization
        ts = sim_data.time_step  # Sampling period in seconds
        fs = 1 / ts  # Sampling frequency in Hz
        length = sim_data.nr_steps

        if length % 2 != 0:  # Ensure length is even for FFT
            length -= 1

        # Pre-allocate arrays
        one_sided = np.zeros((sim_data.nr_diffusing, length // 2 + 1))
        speed = np.zeros((sim_data.nr_diffusing, sim_data.nr_steps))
        freq_mean = np.zeros(sim_data.nr_diffusing)

        # List to hold all vibration amplitudes from all atoms
        all_amplitudes = []

        # Main Loop over Atoms
        atom_counter = -1
        for i in range(sim_data.nr_atoms):
            if sim_data.diffusing_atom[i]:
                atom_counter += 1
                
                # List to hold amplitudes for the current atom
                current_atom_amplitudes = []

                # Calculate speed (displacement derivative) and vibration amplitudes
                for time in range(1, sim_data.nr_steps):
                    current_speed = sim_data.displacement[i, time] - sim_data.displacement[i, time - 1]
                    speed[atom_counter, time] = current_speed

                    if time == 1:
                        # Start the first amplitude entry
                        current_atom_amplitudes.append(current_speed)
                    else:
                        prev_speed = speed[atom_counter, time - 1]
                        # If the atom continues in the same direction, add to the current amplitude
                        if np.sign(current_speed) == np.sign(prev_speed):
                            current_atom_amplitudes[-1] += current_speed
                        # If the direction changes, start a new amplitude entry
                        else:
                            current_atom_amplitudes.append(current_speed)

                # Add the amplitudes from this atom to the master list
                all_amplitudes.extend(current_atom_amplitudes)

                # Frequency Analysis for the Current Atom
                freq_mean[atom_counter] = self._meanfreq(speed[atom_counter, :], fs)

                # Perform Fourier Transform on the speed
                trans = np.fft.fft(speed[atom_counter, :length])
                two_sided = np.abs(trans / length)
                one_sided[atom_counter, :] = two_sided[:length // 2 + 1]

                # Correct amplitude for single-sided spectrum
                one_sided[atom_counter, 1:-1] *= 2

        # Final Calculations
        if all_amplitudes:
            mean_vib, vibration_amp = norm.fit(all_amplitudes)
        else:
            mean_vib, vibration_amp = 0, 0

        # Calculate mean and standard deviation of attempt frequencies
        attempt_freq = np.mean(freq_mean)
        std_attempt_freq = np.std(freq_mean)

        # Create frequency axis for spectrum
        f = fs * np.arange(length // 2 + 1) / length
        sum_freqs = np.sum(one_sided, axis=0)
        smoothed_spectrum = smooth(sum_freqs, window_len=51)

        # Save data if requested
        if save_data:
            self._save_vibration_data(
                all_amplitudes, freq_mean, f, sum_freqs, smoothed_spectrum,
                attempt_freq, vibration_amp, std_attempt_freq, mean_vib,
                output_dir
            )

        # Plotting
        if show_pics:
            self._generate_plots(
                all_amplitudes, mean_vib, vibration_amp,
                one_sided, fs, length,
                attempt_freq, std_attempt_freq,
                output_dir
            )

        return attempt_freq, vibration_amp, std_attempt_freq

    def get_vibration_data_only(self, sim_data):
        """
        Get vibration analysis data without any plotting or saving.
        
        Args:
            sim_data (SimData): The populated simulation data object.
            
        Returns:
            dict: Dictionary containing all vibration analysis data for custom plotting
        """
        # Same calculation as vibration_properties but no plotting/saving
        ts = sim_data.time_step
        fs = 1 / ts
        length = sim_data.nr_steps

        if length % 2 != 0:
            length -= 1

        one_sided = np.zeros((sim_data.nr_diffusing, length // 2 + 1))
        speed = np.zeros((sim_data.nr_diffusing, sim_data.nr_steps))
        freq_mean = np.zeros(sim_data.nr_diffusing)
        all_amplitudes = []

        atom_counter = -1
        for i in range(sim_data.nr_atoms):
            if sim_data.diffusing_atom[i]:
                atom_counter += 1
                current_atom_amplitudes = []

                for time in range(1, sim_data.nr_steps):
                    current_speed = sim_data.displacement[i, time] - sim_data.displacement[i, time - 1]
                    speed[atom_counter, time] = current_speed

                    if time == 1:
                        current_atom_amplitudes.append(current_speed)
                    else:
                        prev_speed = speed[atom_counter, time - 1]
                        if np.sign(current_speed) == np.sign(prev_speed):
                            current_atom_amplitudes[-1] += current_speed
                        else:
                            current_atom_amplitudes.append(current_speed)

                all_amplitudes.extend(current_atom_amplitudes)
                freq_mean[atom_counter] = self._meanfreq(speed[atom_counter, :], fs)

                trans = np.fft.fft(speed[atom_counter, :length])
                two_sided = np.abs(trans / length)
                one_sided[atom_counter, :] = two_sided[:length // 2 + 1]
                one_sided[atom_counter, 1:-1] *= 2

        if all_amplitudes:
            mean_vib, vibration_amp = norm.fit(all_amplitudes)
        else:
            mean_vib, vibration_amp = 0, 0

        attempt_freq = np.mean(freq_mean)
        std_attempt_freq = np.std(freq_mean)

        # Prepare all data for external plotting
        frequencies = fs * np.arange(length // 2 + 1) / length
        sum_freqs = np.sum(one_sided, axis=0)
        smoothed_spectrum = smooth(sum_freqs, window_len=51)

        return {
            'amplitude_data': all_amplitudes,
            'mean_vib': mean_vib,
            'vibration_amp': vibration_amp,
            'one_sided': one_sided,
            'freq_mean': freq_mean,
            'attempt_freq': attempt_freq,
            'std_attempt_freq': std_attempt_freq,
            'fs': fs,
            'length': length,
            'frequencies': frequencies,
            'sum_freqs': sum_freqs,
            'smoothed_spectrum': smoothed_spectrum
        }

    def plot_amplitude_histogram_only(self, vibration_data, bins=100, figsize=(10, 6),
                                     colors=None, xlim=None, ylim=None, output_file=None):
        """
        Plot only the amplitude histogram with full customization.
        
        Args:
            vibration_data (dict): Data from get_vibration_data_only()
            bins (int): Number of histogram bins
            figsize (tuple): Figure size
            colors (dict): Colors for histogram and fit line
            xlim (tuple): X-axis limits
            ylim (tuple): Y-axis limits
            output_file (str): Path to save plot
        """
        plt.figure(figsize=figsize)
        
        hist_color = colors.get('hist', 'lightblue') if colors else 'lightblue'
        fit_color = colors.get('fit', 'red') if colors else 'red'
        
        plt.hist(vibration_data['amplitude_data'], bins=bins, density=True, 
                alpha=0.7, color=hist_color, label='Amplitude Data')
        
        x_fit = np.linspace(min(vibration_data['amplitude_data']), 
                           max(vibration_data['amplitude_data']), 200)
        y_fit = norm.pdf(x_fit, vibration_data['mean_vib'], vibration_data['vibration_amp'])
        plt.plot(x_fit, y_fit, color=fit_color, linewidth=3, label='Gaussian Fit')
        
        plt.title('Histogram of Vibrational Amplitudes')
        plt.xlabel('Amplitude (Angstrom)')
        plt.ylabel('Density (a.u.)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.show()

    def plot_frequency_spectrum_only(self, vibration_data, xlim=None, ylim=None,
                                   smooth_window=51, figsize=(12, 7), output_file=None):
        """
        Plot only the frequency spectrum with full customization.
        
        Args:
            vibration_data (dict): Data from get_vibration_data_only()
            xlim (tuple): X-axis limits
            ylim (tuple): Y-axis limits
            smooth_window (int): Smoothing window size
            figsize (tuple): Figure size
            output_file (str): Path to save plot
        """
        plt.figure(figsize=figsize)
        
        if smooth_window != 51:
            smoothed = smooth(vibration_data['sum_freqs'], window_len=smooth_window)
        else:
            smoothed = vibration_data['smoothed_spectrum']
        
        plt.plot(vibration_data['frequencies'], smoothed, linewidth=2.5, label='Smoothed Spectrum')
        
        plt.axvline(vibration_data['attempt_freq'], color='r', linestyle='-', linewidth=2,
                   label=f'Attempt Freq: {vibration_data["attempt_freq"]:.2e} Hz')
        plt.axvline(vibration_data['attempt_freq'] - vibration_data['std_attempt_freq'], 
                   color='r', linestyle=':', linewidth=2, label='± 1 Std Dev')
        plt.axvline(vibration_data['attempt_freq'] + vibration_data['std_attempt_freq'], 
                   color='r', linestyle=':', linewidth=2)
        
        plt.title('Frequency Spectrum of Diffusing Element')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Occurrence (a.u.)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if xlim:
            plt.xlim(xlim)
        else:
            plt.xlim([0, 2.5E12])
            
        if ylim:
            plt.ylim(ylim)
        else:
            plt.ylim([0, np.max(smoothed) * 1.1])
        
        if output_file:
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
        
        plt.show()

    def _save_vibration_data(self, all_amplitudes, freq_mean, frequencies, spectrum,
                           smoothed_spectrum, attempt_freq, vibration_amp,
                           std_attempt_freq, mean_vib, output_dir=None):
        """
        Save vibrational analysis data to files.
        
        Args:
            all_amplitudes (list): All vibrational amplitudes
            freq_mean (np.array): Mean frequencies for each atom
            frequencies (np.array): Frequency axis
            spectrum (np.array): Power spectrum
            smoothed_spectrum (np.array): Smoothed power spectrum
            attempt_freq (float): Mean attempt frequency
            vibration_amp (float): Vibration amplitude (std dev)
            std_attempt_freq (float): Standard deviation of attempt frequency
            mean_vib (float): Mean vibrational displacement
            output_dir (str, optional): Output directory
        """
        if output_dir is None:
            output_dir = "."
        
        os.makedirs(output_dir, exist_ok=True)

        # 1. Save amplitude distribution data
        amplitude_file = os.path.join(output_dir, "vibrational_amplitudes.dat")
        np.savetxt(amplitude_file, all_amplitudes,
                  header="Vibrational amplitudes (Angstrom) for all diffusing atoms",
                  fmt="%.8e")
        print(f"Saved vibrational amplitudes: {amplitude_file}")

        # 2. Save per-atom frequency data
        atom_freq_file = os.path.join(output_dir, "per_atom_frequencies.dat")
        atom_indices = np.arange(len(freq_mean))
        atom_freq_data = np.column_stack([atom_indices, freq_mean])
        np.savetxt(atom_freq_file, atom_freq_data,
                  header="Atom_Index Mean_Frequency(Hz)",
                  fmt="%d %.8e")
        print(f"Saved per-atom frequencies: {atom_freq_file}")

        # 3. Save frequency spectrum data
        spectrum_file = os.path.join(output_dir, "frequency_spectrum.dat")
        spectrum_data = np.column_stack([frequencies, spectrum, smoothed_spectrum])
        np.savetxt(spectrum_file, spectrum_data,
                  header="Frequency(Hz) Raw_Spectrum Smoothed_Spectrum",
                  fmt="%.8e %.8e %.8e")
        print(f"Saved frequency spectrum: {spectrum_file}")

        # 4. Save summary statistics
        summary_file = os.path.join(output_dir, "vibration_summary.dat")
        with open(summary_file, 'w') as f:
            f.write("# Vibrational Properties Summary\n")
            f.write("# ================================\n")
            f.write(f"# Analysis timestamp: {np.datetime64('now')}\n")
            f.write("#\n")
            f.write("# FREQUENCY PROPERTIES:\n")
            f.write(f"Mean_Attempt_Frequency(Hz): {attempt_freq:.6e}\n")
            f.write(f"Std_Attempt_Frequency(Hz): {std_attempt_freq:.6e}\n")
            f.write("#\n")
            f.write("# AMPLITUDE PROPERTIES:\n")
            f.write(f"Vibration_Amplitude_StdDev(Angstrom): {vibration_amp:.6f}\n")
            f.write(f"Mean_Vibrational_Displacement(Angstrom): {mean_vib:.6f}\n")
            f.write("#\n")
            f.write("# STATISTICS:\n")
            f.write(f"Number_of_Diffusing_Atoms: {len(freq_mean)}\n")
            f.write(f"Total_Amplitude_Data_Points: {len(all_amplitudes)}\n")
            f.write(f"Min_Frequency(Hz): {np.min(freq_mean):.6e}\n")
            f.write(f"Max_Frequency(Hz): {np.max(freq_mean):.6e}\n")
            f.write(f"Min_Amplitude(Angstrom): {np.min(all_amplitudes):.6f}\n")
            f.write(f"Max_Amplitude(Angstrom): {np.max(all_amplitudes):.6f}\n")
        
        print(f"Saved summary statistics: {summary_file}")

        # 5. Save amplitude histogram bins for easy plotting
        hist_file = os.path.join(output_dir, "amplitude_histogram_data.dat")
        hist_counts, hist_edges = np.histogram(all_amplitudes, bins=100, density=True)
        hist_centers = (hist_edges[:-1] + hist_edges[1:]) / 2
        hist_data = np.column_stack([hist_centers, hist_counts])
        
        # Also save Gaussian fit data
        x_fit = np.linspace(np.min(all_amplitudes), np.max(all_amplitudes), 200)
        y_fit = norm.pdf(x_fit, mean_vib, vibration_amp)
        fit_data = np.column_stack([x_fit, y_fit])
        
        np.savetxt(hist_file, hist_data,
                  header="Histogram: Amplitude_Center(Angstrom) Density",
                  fmt="%.8e %.8e")
        
        fit_file = os.path.join(output_dir, "amplitude_gaussian_fit.dat")
        np.savetxt(fit_file, fit_data,
                  header="Gaussian_Fit: Amplitude(Angstrom) Probability_Density",
                  fmt="%.8e %.8e")
        
        print(f"Saved histogram data: {hist_file}")
        print(f"Saved Gaussian fit: {fit_file}")

    def _generate_plots(self, amplitude_data, mean_vib, vibration_amp,
                       one_sided, fs, length, attempt_freq, std_attempt_freq,
                       output_dir=None):
        """Helper function to generate and display plots."""
        # Plot 1: Histogram of Vibrational Amplitudes
        plt.figure(figsize=(10, 6))
        
        # Plot histogram
        plt.hist(amplitude_data, bins=100, density=True, alpha=0.7, label='Amplitude Data')
        
        # Plot fitted Gaussian curve
        x_fit = np.linspace(min(amplitude_data), max(amplitude_data), 200)
        y_fit = norm.pdf(x_fit, mean_vib, vibration_amp)
        plt.plot(x_fit, y_fit, 'r-', linewidth=3, label='Gaussian Fit')
        
        plt.title('Histogram of Vibrational Amplitudes')
        plt.xlabel('Amplitude (Angstrom)')
        plt.ylabel('Density (a.u.)')
        plt.gca().axes.get_yaxis().set_ticks([])  # Hide y-axis labels
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "vibrational_amplitudes.png"),
                       dpi=300, bbox_inches='tight')
        plt.show()

        # Plot 2: Frequency Spectrum
        plt.figure(figsize=(12, 7))
        f = fs * np.arange(length // 2 + 1) / length  # Frequency axis
        sum_freqs = np.sum(one_sided, axis=0)
        smoothed = smooth(sum_freqs, window_len=51)  # Smooth the spectrum

        plt.plot(f, smoothed, linewidth=2.5, label='Smoothed Spectrum')
        
        # Overlay attempt frequency and standard deviation
        plt.axvline(attempt_freq, color='r', linestyle='-', linewidth=2, 
                   label=f'Attempt Freq: {attempt_freq:.2e} Hz')
        plt.axvline(attempt_freq - std_attempt_freq, color='r', linestyle=':', 
                   linewidth=2, label='± 1 Std Dev')
        plt.axvline(attempt_freq + std_attempt_freq, color='r', linestyle=':', linewidth=2)
        
        plt.title('Frequency Spectrum of Diffusing Element')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Occurrence (a.u.)')
        plt.gca().axes.get_yaxis().set_ticks([])  # Hide y-axis labels
        plt.xlim([0, 2.5E12])
        plt.ylim([0, np.max(smoothed) * 1.1])
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, "frequency_spectrum.png"),
                       dpi=300, bbox_inches='tight')
        plt.show()

