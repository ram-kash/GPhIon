import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from ..core import AnalysisBase
from ..core.plot_io import save_plot_data

class NonGaussianCalculator(AnalysisBase):
    """
    Calculate the non-Gaussian parameter (NGP) from displacement data,
    assuming sim_data.displacement is (n_atoms, n_timesteps) with magnitudes.
    """

    def diagnose_displacement(self, sim_data):
        disp = sim_data.displacement
        print("Displacement shape:", disp.shape)
        print("Min displacement:", np.nanmin(disp))
        print("Max displacement:", np.nanmax(disp))
        print("Typical final displacement:", np.mean(disp[:, -1]))

    def calculate_ngp(self, sim_data, skip_steps=1, smoothing_window=1):
        """
        Calculate NGP α2(t) = (3⟨r⁴⟩)/(5⟨r²⟩²) - 1 for 3D.
        """
        disp = sim_data.displacement  # shape (n_atoms, n_timesteps)
        n_steps = disp.shape[1]
        time_indices = np.arange(0, n_steps, skip_steps)
        if time_indices[-1] != n_steps - 1:
            time_indices = np.append(time_indices, n_steps-1)

        ngp = np.zeros(len(time_indices))
        msd = np.zeros(len(time_indices))
        for i, t in enumerate(time_indices):
            r = disp[:, t]
            msd[i] = np.mean(r ** 2)
            # print(i,t,r,msd[i])
            mean_r4 = np.mean(r ** 4)
            if msd[i] > 1e-10:
                ngp[i] = (3 / 5) * (mean_r4 / msd[i] ** 2) - 1
            else:
                ngp[i] = 0.0

        if smoothing_window > 1:
            ngp_smoothed = uniform_filter1d(ngp, size=smoothing_window)
        else:
            ngp_smoothed = ngp.copy()

        # Store or return as desired
        sim_data.ngp_time = time_indices * sim_data.time_step
        sim_data.ngp = ngp
        sim_data.ngp_smoothed = ngp_smoothed
        sim_data.msd = msd
        return sim_data

    from typing import Optional

    def plot_ngp(self, sim_data, use_smoothed=True, save_payload_dir: Optional[str] = None):
        time = sim_data.ngp_time
        ngp = sim_data.ngp_smoothed if use_smoothed else sim_data.ngp
        # Save payload
        try:
            base_dir = save_payload_dir or self.save_payload_dir or self.output_dir or "plot_data"
            save_plot_data(
                name="ngp",
                payload={
                    "time": time,
                    "ngp": ngp,
                    "raw_ngp": getattr(sim_data, 'ngp', None),
                    "msd": getattr(sim_data, 'msd', None),
                    "params": {"use_smoothed": bool(use_smoothed)},
                },
                base_dir=base_dir,
            )
        except Exception as e:
            print(f"WARNING: Failed to save NGP payload: {e}")

        eff_show = self.show_plot
        if not eff_show:
            return
        plt.figure(figsize=(7,4))
        plt.plot(time, ngp, label='NGP (smoothed)' if use_smoothed else 'NGP (raw)')
        # plt.axhline(y=0, color='r', ls='--', label='Gaussian (NGP=0)')
        plt.xlabel('Time step')
        plt.ylabel('Non-Gaussian Parameter $α_2$(t)')
        plt.legend()
        plt.tight_layout()
        plt.show()

