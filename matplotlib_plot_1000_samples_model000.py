# Program 5
import matplotlib.pyplot as plt
import numpy as np

# This program loads the previously saved validation results and creates
# three simple scatter plots for inspecting the model errors across
# all validation frames as well as a signed energy-error plot.
# Please see Matplotlib scatter and NumPy loadtxt
# documentation for more information.

data = np.loadtxt(
    "results/validation_all_model000.csv",
    delimiter=",",
    skiprows=1
)
frame = data[:, 0]
energy_error = data[:, 3]
force_rmse = data[:, 4]
# PLOT 1. Signed (in the sense of positive and/or negative) energy error plot
plt.scatter(frame, energy_error, s=10, alpha=0.4)
plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlabel("Frame")
plt.ylabel("Energy error (meV/atom)")
plt.title("Energy error across validation frames")
plt.savefig("results/matplotlib_plot_1000_samples_model000_figure_energy_1.png",
            dpi=200, bbox_inches="tight")
plt.show()

# PLOT 2. Force RMSE vs. frame plot
plt.scatter(frame, force_rmse, s=10, alpha=0.4)
plt.xlabel("Frame")
plt.ylabel("Force RMSE (eV/Å)")
plt.title("Force RMSE across validation frames")
plt.savefig("results/matplotlib_plot_1000_samples_model000_figure_force_RMSE_1.png",
            dpi=200, bbox_inches="tight")
plt.show()

# PLOT 3. Energy parity plot: predicted energy vs. reference energy
e_ref = data[:, 1]
e_pred = data[:, 2]

plt.scatter(e_ref, e_pred, s=10, alpha=0.4)

# Perfect-agreement line: predicted = reference
limits = [
    min(e_ref.min(), e_pred.min()),
    max(e_ref.max(), e_pred.max())
]
plt.plot(limits, limits, "--")

plt.xlabel("Reference energy (eV)")
plt.ylabel("Predicted energy (eV)")
plt.title("Energy parity plot")
plt.savefig("results/matplotlib_plot_1000_samples_model000_figure_energy_parity_plot_1.png",
            dpi=200, bbox_inches="tight")
plt.show()

print("Frame / energy error:", np.corrcoef(frame, energy_error)[0, 1])
print("Frame / force RMSE:", np.corrcoef(frame, force_rmse)[0, 1])

r_energy_error = np.corrcoef(e_ref, energy_error)[0, 1]
r_force_error = np.corrcoef(e_ref, force_rmse)[0, 1]
r_frame_energy = np.corrcoef(frame, e_ref)[0, 1]


print(f"Reference energy / energy error: {r_energy_error:+.6f}")
print(f"Reference energy / force RMSE: {r_force_error:+.6f}")
print(f"Frame / reference energy: {r_frame_energy:+.6f}")
