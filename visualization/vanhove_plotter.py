"""Van Hove correlation function plotting utilities."""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class VanHovePlotter:
    """
    A dedicated class for plotting Van Hove correlation functions with full customization control.
    """
    
    def __init__(self):
        """Initialize the Van Hove plotter."""
        self.default_figure_size = (12, 8)
        self.default_colors = plt.cm.viridis
        
    def plot_van_hove_functions(self, r_values, self_results, distinct_results, time_lags, 
                               timestep=None, diffusing_element=None, sigma=None, xlim=None, ylim=None,
                               yscale=None, show_original=False, output_dir=None):
        """
        Plot Van Hove correlation functions with full user control.
        
        Args:
            r_values (np.array): Radial distance values
            self_results (dict): Self Van Hove correlation function results
            distinct_results (dict): Distinct Van Hove correlation function results
            time_lags (list): Time lags used in analysis (in frame units)
            timestep (float, optional): Timestep in seconds to convert frames to real time
            diffusing_element (str, optional): Element name for plot titles
            sigma (float, optional): Standard deviation for Gaussian smoothing
            xlim (tuple, optional): X-axis limits as (xmin, xmax)
            ylim (tuple, optional): Y-axis limits as (ymin, ymax)
            yscale (str, optional): Y-axis scale ('log', 'linear', or None)
            show_original (bool): Whether to show original data alongside smoothed data
            output_dir (str, optional): Directory to save plots
        """
        element_name = diffusing_element or "Diffusing Ions"
        
        # Plot self-part
        self._plot_single_van_hove(
            r_values, self_results, time_lags,
            title=f'Self Van Hove Function - {element_name}',
            ylabel=r'$G_{self}(r,t)$',
            filename=f"{output_dir}/van_hove_self.png" if output_dir else None,
            timestep=timestep, sigma=sigma, xlim=xlim, ylim=ylim, yscale=yscale,
            show_original=show_original, default_log_scale=True
        )
        
        # Plot distinct-part
        self._plot_single_van_hove(
            r_values, distinct_results, time_lags,
            title=f'Distinct Van Hove Function - {element_name}',
            ylabel=r'$G_{distinct}(r,t)$',
            filename=f"{output_dir}/van_hove_distinct.png" if output_dir else None,
            timestep=timestep, sigma=sigma, xlim=xlim, ylim=ylim, yscale=yscale,
            show_original=show_original, default_log_scale=False
        )
    
    def plot_self_van_hove(self, r_values, self_results, time_lags, 
                          timestep=None, diffusing_element=None, sigma=None, xlim=None, ylim=None,
                          yscale=None, show_original=False, output_file=None):
        """
        Plot only the self Van Hove correlation function.
        
        Args:
            r_values (np.array): Radial distance values
            self_results (dict): Self Van Hove correlation function results
            time_lags (list): Time lags used in analysis (in frame units)
            timestep (float, optional): Timestep in seconds to convert frames to real time
            diffusing_element (str, optional): Element name for plot title
            sigma (float, optional): Standard deviation for Gaussian smoothing
            xlim (tuple, optional): X-axis limits as (xmin, xmax)
            ylim (tuple, optional): Y-axis limits as (ymin, ymax)
            yscale (str, optional): Y-axis scale ('log', 'linear', or None)
            show_original (bool): Whether to show original data alongside smoothed data
            output_file (str, optional): Path to save the plot
        """
        element_name = diffusing_element or "Diffusing Ions"
        
        self._plot_single_van_hove(
            r_values, self_results, time_lags,
            title=f'Self Van Hove Function - {element_name}',
            ylabel=r'$G_{self}(r,t)$',
            filename=output_file,
            timestep=timestep, sigma=sigma, xlim=xlim, ylim=ylim, yscale=yscale,
            show_original=show_original, default_log_scale=True
        )
    
    def plot_distinct_van_hove(self, r_values, distinct_results, time_lags,
                              timestep=None, diffusing_element=None, sigma=None, xlim=None, ylim=None,
                              yscale=None, show_original=False, output_file=None):
        """
        Plot only the distinct Van Hove correlation function.
        
        Args:
            r_values (np.array): Radial distance values
            distinct_results (dict): Distinct Van Hove correlation function results
            time_lags (list): Time lags used in analysis (in frame units)
            timestep (float, optional): Timestep in seconds to convert frames to real time
            diffusing_element (str, optional): Element name for plot title
            sigma (float, optional): Standard deviation for Gaussian smoothing
            xlim (tuple, optional): X-axis limits as (xmin, xmax)
            ylim (tuple, optional): Y-axis limits as (ymin, ymax)
            yscale (str, optional): Y-axis scale ('log', 'linear', or None)
            show_original (bool): Whether to show original data alongside smoothed data
            output_file (str, optional): Path to save the plot
        """
        element_name = diffusing_element or "Diffusing Ions"
        
        self._plot_single_van_hove(
            r_values, distinct_results, time_lags,
            title=f'Distinct Van Hove Function - {element_name}',
            ylabel=r'$G_{distinct}(r,t)$',
            filename=output_file,
            timestep=timestep, sigma=sigma, xlim=xlim, ylim=ylim, yscale=yscale,
            show_original=show_original, default_log_scale=False
        )
    
    def _format_time_label(self, lag, timestep):
        """
        Format time label for the legend based on timestep and lag.
        
        Args:
            lag (int): Time lag in frames
            timestep (float): Timestep in seconds
            
        Returns:
            str: Formatted time label
        """
        if timestep is None:
            return f't = {lag}'
        
        actual_time = lag * timestep
        
        # Choose appropriate time unit and formatting
        if actual_time < 1e-12:  # femtoseconds
            return f't = {actual_time*1e15:.1f} fs'
        elif actual_time < 1e-9:  # picoseconds
            return f't = {actual_time*1e12:.1f} ps'
        elif actual_time < 1e-6:  # nanoseconds
            return f't = {actual_time*1e9:.1f} ns'
        elif actual_time < 1e-3:  # microseconds
            return f't = {actual_time*1e6:.1f} μs'
        elif actual_time < 1:  # milliseconds
            return f't = {actual_time*1e3:.1f} ms'
        else:  # seconds
            return f't = {actual_time:.2f} s'
    
    def _plot_single_van_hove(self, r_values, results, time_lags, title, ylabel, 
                             filename=None, timestep=None, sigma=None, xlim=None, ylim=None,
                             yscale=None, show_original=False, default_log_scale=False):
        """
        Plot a single Van Hove correlation function with full user customization.
        
        Args:
            r_values (np.array): Radial distance values
            results (dict): Van Hove results dictionary
            time_lags (list): Time lags in frame units
            title (str): Plot title
            ylabel (str): Y-axis label
            filename (str, optional): Output filename
            timestep (float, optional): Timestep in seconds for time conversion
            sigma (float, optional): Gaussian smoothing parameter
            xlim (tuple, optional): X-axis limits
            ylim (tuple, optional): Y-axis limits
            yscale (str, optional): Y-axis scale ('log', 'linear', or None)
            show_original (bool): Show original data with smoothed
            default_log_scale (bool): Default scale if yscale is None
        """
        plt.figure(figsize=self.default_figure_size)
        colors = self.default_colors(np.linspace(0, 1, len(time_lags)))
        
        for i, lag in enumerate(time_lags):
            data = results[lag]
            
            # Generate time label
            time_label = self._format_time_label(lag, timestep)
            
            if sigma is not None:
                # Apply Gaussian smoothing
                smoothed_data = gaussian_filter1d(data, sigma=sigma)
                
                if show_original:
                    # Plot original data as thin, transparent line
                    plt.plot(r_values, data, color=colors[i], alpha=0.3, linewidth=1,
                            linestyle='--', label=f'{time_label} (original)')
                    # Plot smoothed data as main line
                    plt.plot(r_values, smoothed_data, color=colors[i], linewidth=2.5,
                            label=f'{time_label}') #  (σ={sigma})
                else:
                    # Plot only smoothed data
                    plt.plot(r_values, smoothed_data, color=colors[i], linewidth=2.5,
                            label=f'{time_label}') #  (σ={sigma})
            else:
                # Plot original data only
                plt.plot(r_values, data, color=colors[i], linewidth=2, label=time_label)
        
        # Set labels and title
        plt.xlabel(r'$r$ (Å)', fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(title, fontsize=14)
        plt.legend(fontsize=10, bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        if xlim is not None:
            plt.xlim(xlim)
        else:
            plt.xlim(0, min(10, max(r_values)))
            
        if ylim is not None:
            plt.ylim(ylim)
        
        # Set y-axis scale based on user input or default
        if yscale == 'log':
            plt.yscale('log')
        elif yscale == 'linear':
            plt.yscale('linear')
        elif yscale is None and default_log_scale:
            plt.yscale('log')
            
        plt.tight_layout()
        
        if filename:
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Plot saved: {filename}")
        
        plt.show()
    
    def set_plot_style(self, figure_size=None, colormap=None):
        """
        Customize the default plot styling.
        
        Args:
            figure_size (tuple, optional): Figure size as (width, height)
            colormap (str or matplotlib colormap, optional): Colormap for time series
        """
        if figure_size is not None:
            self.default_figure_size = figure_size
            
        if colormap is not None:
            if isinstance(colormap, str):
                self.default_colors = plt.cm.get_cmap(colormap)
            else:
                self.default_colors = colormap
                
        print(f"Plot style updated: figure_size={self.default_figure_size}")

