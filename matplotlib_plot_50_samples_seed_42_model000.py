# Program 4
import matplotlib.pyplot as plt
import numpy as np

# This program loads the previously saved validation results and creates
# two simple scatter plots for inspecting the model errors across
# the sampled frames. Please see Matplotlib scatter and NumPy loadtxt
# documentation for more information.

data = np.loadtxt(
    "results/validation_50_seed42_model000.csv",
    delimiter=",",
    skiprows=1
)
frame = data[:, 0]
energy_error = data[:, 3]
force_rmse = data[:, 4]

plt.scatter(frame, energy_error)
plt.xlabel("Frame")
plt.ylabel("Energy error (meV/atom)")
plt.title("Energy error across sampled frames")
plt.show()


plt.scatter(frame, force_rmse)
plt.xlabel("Frame")
plt.ylabel("Force RMSE (eV/Å)")
plt.title("Force RMSE across sampled frames")
plt.show()

e_ref = data[:, 1]
e_pred = data[:, 2]

plt.scatter(e_ref, e_pred)

# Perfect-agreement line: predicted = reference
limits = [
    min(e_ref.min(), e_pred.min()),
    max(e_ref.max(), e_pred.max())
]
plt.plot(limits, limits, "--")

plt.xlabel("Reference energy (eV)")
plt.ylabel("Predicted energy (eV)")
plt.title("Energy parity plot")
plt.show()
