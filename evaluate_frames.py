# Program 3
# This section imports two tools that will be used for running and timing
# the DeepMD model. DeepPot is DeepMD-kit's inference class for loading
# a trained potential model and evaluating quantities such as energies,
# forces, and virials for atomic configurations. perf_counter is Python's
# high-resolution timer, intended for measuring elapsed execution time
# between two points in the program. This will let us evaluate the trained
# deep potential and measure how long that evaluation takes. Please see
# the DeepMD-kit Python inference doc:
# https://docs.deepmodeling.com/projects/deepmd/en/master/api_py/deepmd.infer.html
# and the Python time.perf_counter() documentation for more information.
from time import perf_counter
from pathlib import Path
from deepmd.infer import DeepPot
import numpy as np


# Locate the data in relation to this program.
# Here Path(__file__).resolve().parent determines the directory where the current Python
# file is located, and the / operator is then used to construct paths relative to that location.
# More info: https://docs.python.org/3/library/pathlib.html
project = Path(__file__).resolve().parent
system = project / "data/raw/data/valid"
dataset = system / "set.000"

# Read the element names and each atom's integer type. np.loadtxt(..., dtype=int) loads these
# assignments as integers, .reshape(-1) ensures they form a one-dimensional array. atom_types.size
# gives the total number of atoms in the system. See the NumPy `loadtxt` documentation
# https://numpy.org/doc/stable/reference/generated/numpy.loadtxt.html for more information.
type_names = (system / "type_map.raw").read_text().split()
atom_types = np.loadtxt(system / "type.raw", dtype=int).reshape(-1)
n_atoms = atom_types.size

# We want to open the numerical array in read-only mode.
# np.load reads Numpy .npy files while mmap_mode="r" opens them in read-only memory-mapped mode,
# which means that the complete arrays do not need to be loaded into RAM at once
# https://numpy.org/doc/stable/reference/generated/numpy.load.html (see 'load' documentation)
coords = np.load(dataset / "coord.npy", mmap_mode="r")
boxes = np.load(dataset / "box.npy", mmap_mode="r")
energies = np.load(dataset / "energy.npy", mmap_mode="r")
forces = np.load(dataset / "force.npy", mmap_mode="r")

# This section selects the first stored configuration, or frame, from the dataset.
# The coordinates and forces are reshaped into (n_atoms, 3), giving one row per atom
# and three columns corresponding to the Cartesian components/coordinates, while the
# simulation cell is represented as a 3x3 matrix. The energy belongs to the configuration
# as a whole rather than to individual atoms, and .item() extracts it as a scalar value from
# the NumPy array. See the "reshape" documentation
# (https://numpy.org/doc/stable/reference/generated/numpy.reshape.html) for more information.
frame = 0
positions = coords[frame].reshape(n_atoms, 3)
cell = boxes[frame].reshape(3, 3)
energy = energies[frame].item()
atom_forces = forces[frame].reshape(n_atoms, 3)

# This section prints a compact summary of the loaded dataset and the selected configuration.
# np.set_printoptions() controls how NumPy arrays are displayed. The following lines report, among
# others, stored frames (snapshots), number of atoms per frame, selected frame, and its reference
# total energy. Lastly, simulation cell is printed as a 3x3 matrix, where each row represents one of
# the 3 cell vectors in Å. See https://numpy.org/doc/stable/reference/generated/numpy.set_printoptions.html
np.set_printoptions(precision=6, suppress=True)
print(f"NumPy version: {np.__version__}")
print(f"Frames in this set: {coords.shape[0]}")
print(f"Atoms per frame: {n_atoms}")
print(f"Selected frame: {frame}")
print(f"Reference total energy: {energy: .8f} eV")
print("\nSimulation cell vectors (rows, in Å):")
print(cell)

# This section prints the positions and forces of the first 5 atoms in the selected configuration,
# providing a small and readable sample of the atomic data. The for loop iterates over at most five
# atoms, while "type_names[atom_types[atom]]" converts each atom's numerical type into its
# chemical symbol. For each atom, the output reports its index and element, followed by its 3D
# position in Å and the corresponding force vector in eV/Å.
print("\nFirst five atoms: positions in Å; forces in eV/Å")
for atom in range(min(5, n_atoms)):
    symbol = type_names[atom_types[atom]]
    print(
        f"Atom {atom:3d} ({symbol:2s}) | "
        f"position {positions[atom]} | force {atom_forces[atom]}"
    )

# This section aims to tell us how many atoms of each element
# listed in type_names we have. For example, for 3 water molecules,
# type.raw contains nine integer atom types; their corresponding
# element names come from type_map.raw, giving H:6 and O:3.
# See Python enumerate() and NumPy count_nonzero() for more info.
print("\nElement counts")
for type_index, symbol in enumerate(type_names):
    count = np.count_nonzero(atom_types == type_index)
    print(f"{symbol}: {count}")

# This section will increase our understanding of
# the cell geometry: are all stored cells identical
# relative to frame 0? See the NumPy max and absolute
# documentation for more information.
max_cell_difference = np.max(np.abs(boxes - boxes[0]))
print(
    "\nLargest absolute cell-component difference from frame 0: "
    f"{max_cell_difference:.10g} Å"
)

# Load one trained model and inspect its stored metadata.
# See DeepMD DeepPOT documentation for more information:
# https://docs.deepmodeling.com/projects/deepmd/en/master/api_py/deepmd.infer.html
model_path = project / "data/raw/models/model.000.pb"
model = DeepPot(str(model_path))

model_types = model.get_type_map()
print(f"\nModel: {model_path.name}")
print(f"Model element ordering: {model_types}")
print(f"Model cutoff: {model.get_rcut()} Å")

# Integer atom labels must mean the same thing to the model.
if model_types != type_names:
    raise ValueError("Model and dataset element orderings differ.")

# This section evaluates the trained model for the currently selected atomic configuration and
# measures how long that prediction takes. positions.reshape(1, -1) converts the coordinates
# into a two-dimensional array where the first dimension has size 1, meaning one frame is
# being supplied, while the second dimension contains all coordinate values for that frame.
# In other words, instead of passing an (n_atoms, 3) coordinate array, we pass something shaped
# conceptually like (number_of_frames, coordinate_values_per_frame), which here becomes
# (1, 3 × n_atoms). The cell is treated similarly with cell.reshape(1, 9), meaning one frame
# containing the nine components of the 3x3 simulation-cell matrix. model.eval(...)
# then returns the predicted energy, forces, and an additional output for that single frame,
# while perf_counter() measures the elapsed evaluation time. See the DeepMD-kit DeepPot.eval
# documentation: https://docs.deepmodeling.com/projects/deepmd/en/master/api_py/deepmd.infer.html
# and the NumPy reshape documentation for more information.
start = perf_counter()
predicted_energies, predicted_forces, _ = model.eval(
    positions.reshape(1, -1),
    cell.reshape(1, 9),
    atom_types.tolist(),
)
elapsed = perf_counter() - start

# Extract this frame's predictions and compare with the references.
predicted_energy = predicted_energies.item()
predicted_force = predicted_forces[0]

energy_error = predicted_energy - energy
force_error = predicted_force - atom_forces
force_rmse = np.sqrt(np.mean(force_error**2))

print(f"\nFrame: {frame}")
print(f"Reference energy: {energy:.8f} eV")
print(f"Predicted energy: {predicted_energy:.8f} eV")
print(f"Energy error (ML - reference): {energy_error:+.8f} eV")
print(
    "Absolute energy error per atom: "
    f"{1000 * abs(energy_error) / n_atoms:.6f} meV/atom"
)
print(f"Force-component RMSE: {force_rmse:.6f} eV/Å")
print(f"First evaluation time: {elapsed:.3f} seconds")


# This section randomly selects 50 different frames from the loaded validation
# set (see 'dataset' and 'coords' above) while making the selection
# reproducible. np.random.default_rng(seed) creates a NumPy random-number
# generator initialized with the chosen seed (42), and rng.choice(...)
# uses that generator to select frame indices from the coords.shape[0]
# available frames. size=sample_size requests 50 indices, and replace=False
# means that once a frame was selected it cannot be selected again, so all
# 50 frames are unique. See NumPy random generator documentation for more information.
sample_size = 50
seed = 42
rng = np.random.default_rng(seed)
frame_ids = np.sort(
    rng.choice(coords.shape[0], size=sample_size, replace=False)
)

# This section reuses the already loaded DeepMD model
# to evaluate the 50 frames one by one, for each we shape them
# into the format expected by model.eval() which returns model-predicted
# energies and forces, the ref energies are then retrieved for that frame,
# so we can compare. We then convert the predicted energy into a scalar, compute
# the signed energy error per atom in the right unit (meV/atom), calculate the force
# RMSE for each sampled frame. We collected them in 'row'.
# We raise error in case of non-finite values (e.g., NaN, inf). We append
# to 'rows', print a progress every 10 frames. Afterwards, elapsed stores
# the elapsed time for the evaluation loop and np.array(rows) converts the list into a NumPy array.
# See NumPy isfinite and array documentations for more information.
print(f"frame ids: {frame_ids}")
rows = []
start = perf_counter()

for count, frame_id in enumerate(frame_ids, start=1):
    e_pred, f_pred, _ = model.eval(
        coords[frame_id].reshape(1, -1),
        boxes[frame_id].reshape(1, 9),
        atom_types.tolist(),
    )

    e_ref = energies[frame_id].item()
    f_ref = forces[frame_id].reshape(n_atoms, 3)
    e_pred = e_pred.item()

    energy_error_per_atom = 1000 * (e_pred - e_ref) / n_atoms
    force_rmse = np.sqrt(np.mean((f_pred[0]-f_ref)**2))

    row = [frame_id, e_ref, e_pred, energy_error_per_atom, force_rmse]
    if not np.isfinite(row).all():
        raise ValueError(
            f"Non-finite or abnormal result for frame {frame_id}.")

    rows.append(row)

    if count % 10 == 0:
        print(f"Evaluated {count}/{sample_size} sampled frames")

elapsed = perf_counter() - start
results = np.array(rows)

# Save the selected indices and results together
output_folder = project / "results"
output_folder.mkdir(exist_ok=True)
output_file = output_folder / "validation_50_seed42_model000.csv"

np.savetxt(
    output_file,
    results,
    delimiter=",",
    header=(
        "frame,reference_energy_ev,predicted_energy_ev,"
        "energy_error_meV_per_atom,force_rmse_eV_per_A"
    ),
    comments="",
    fmt=["%d", "%.10f", "%.10f", "%.8f", "%.8f"],
)
# Finally we summarize the errors across the samples!
# This gives a compact summary of how the trained model
# reproduces the reference data over the 50 randomly selected configurations.
energy_rmse = np.sqrt(np.mean(results[:, 3]**2))
overall_force_rmse = np.sqrt(np.mean(results[:, 4]**2))

print(f"\nSample size: {sample_size}; random seed: {seed}")
print(f"Energy RMSE per atom: {energy_rmse:.6f} meV/atom")
print(f"Overall force-component RMSE: {overall_force_rmse:.6f} eV/Å")
print(f"Subset evaluation time: {elapsed:.3f} seconds")
print(f"Saved: {output_file}")
mean_energy_error = np.mean(results[:, 3])
print(f"Mean signed energy error: {mean_energy_error:+.6f}")


# We are going to check the translational invariance which was described
# in section 2.1. We translate every atom by the same vector in Å and evaluate
# the translated configuration with the same periodic cell.

shift = np.array([0.3, -0.2, 0.4])
shifted_positions = positions + shift
shifted_energies, shifted_forces, _ = model.eval(
    shifted_positions.reshape(1, -1),
    cell.reshape(1, 9),
    atom_types.tolist(),
)

# Compare with the original model predictions for frame 0.
energy_change = shifted_energies.item() - predicted_energy
force_max_change = np.max(
    np.abs(shifted_forces[0] - predicted_force)
)

print(f"Translation energy change: {energy_change:+.3e} eV")
print(f"Largest force-component change: {force_max_change:.3e} eV/Å")

# We are going to reorder the atoms reproducibly, keeping coordinates
# and types together. We are going to check permutation invariance.
# It may seem unnecessary and/or tedious but it is a good practice
# to convince ourselves that the energy stays unchanged.
# This section checks permutation invariance by randomly reordering
# the atom records while keeping each atom’s coordinates and element type
# together. np.random.default_rng(42).permutation(n_atoms) creates a
# reproducible shuffled ordering of the atom indices, and that same
# ordering is then applied to both 'positions' and 'atom_types',
# so the physical configuration itself is unchanged.
# The permuted configuration is passed through the same model,
# which should therefore predict the same total energy and the
# same force for each physical atom. Because the atoms now
# appear in a different row order, 'predicted_force[permutation]'
# simply rearranges the original force output into that same
# new order so that corresponding atoms are compared correctly.
# The code then reports the change in total energy and the largest
# force-component difference, both of which should be essentially
# zero if the model is permutation invariant.
# See the NumPy Generator.permutation documentation for more information.
permutation = np.random.default_rng(42).permutation(n_atoms)
permuted_positions = positions[permutation]
permuted_types = atom_types[permutation]

permuted_energies, permuted_forces, _ = model.eval(
    permuted_positions.reshape(1, -1),
    cell.reshape(1, 9),
    permuted_types.tolist(),
)

permutation_energy_change = permuted_energies.item() - predicted_energy
permutation_force_max_change = np.max(
    np.abs(permuted_forces[0] - predicted_force[permutation])
)

print(f"Permutation energy change: {permutation_energy_change:+.3e} eV")
print(
    "Largest force-component change after reordering: "
    f"{permutation_force_max_change: .3e} eV/Å"
)

# Lastly, we are going to check rotational invariance
# (energy invariance and force equivariance) by rotating
# the entire configuration and simulation cell by 37 degrees
# around the z-axis. np.deg2rad(angle) converts the angle to
# radians, while the matrix built from cos(theta) and sin(theta)
# is the standard 3D rotation about z.  Multiplication by rotation.T
# rotates both consistently without changing the arrangement relative
# to the cell. The E remain the same but the force needs
# to be rotated (so also multiplied by rotation.T), each force vector
# keeps its magnitude and rotates with the configuration.
# Convince yourself of that simply by thinking about a vector changing
# direction not magnitude. See NumPy matrix multiplication and NumPy
# trigonometric functions documentations for more information.
theta = np.deg2rad(37.0)
c, s = np.cos(theta), np.sin(theta)

rotation = np.array([
    [c, -s, 0.0],
    [s, c, 0.0],
    [0.0, 0.0, 1.0],
])

# This section checks that the matrix we constructed is actually
# a valid rotation matrix before using it. rotation.T @ rotation should
# equal the identity matrix, because the transpose of a proper rotation
# matrix is also its inverse; np.allclose(...) verifies that this is true
# within a very small numerical tolerance. np.linalg.det(rotation) computes
# the determinant, which should be +1 for a proper rotation, and
# np.isclose(...) checks that condition. The assert statements stop the
# program immediately if either test fails, which is useful for catching
# an incorrectly constructed matrix. See the NumPy allclose and determinant
# documentation for more information.
assert np.allclose(rotation.T @ rotation, np.eye(3), rtol=0, atol=1e-12)
assert np.isclose(np.linalg.det(rotation), 1.0, rtol=0, atol=1e-12)

# Coordinates and cell vectors are stored as rows.
rotated_positions = positions @ rotation.T
rotated_cell = cell @ rotation.T

rotated_energies, rotated_forces, _ = model.eval(
    rotated_positions.reshape(1, -1),
    rotated_cell.reshape(1, 9),
    atom_types.tolist(),
)

expected_rotated_forces = predicted_force @ rotation.T

rotation_energy_change = rotated_energies.item() - predicted_energy
rotation_force_max_change = np.max(
    np.abs(rotated_forces[0] - expected_rotated_forces)
)

print(f"Rotation energy change: {rotation_energy_change:+.3e} eV")
print(
    "Largest force-component difference after rotation: "
    f"{rotation_force_max_change:+.3e} eV/Å"
)


# Evaluate all 1000 samples! (see 50 frame example above for comments)
frame_ids = np.arange(coords.shape[0])
sample_size = frame_ids.size
rows = []
start = perf_counter()

for count, frame_id in enumerate(frame_ids, start=1):
    e_pred, f_pred, _ = model.eval(
        coords[frame_id].reshape(1, -1),
        boxes[frame_id].reshape(1, 9),
        atom_types.tolist(),
    )

    e_ref = energies[frame_id].item()
    f_ref = forces[frame_id].reshape(n_atoms, 3)
    e_pred = e_pred.item()

    energy_error_per_atom = 1000 * (e_pred - e_ref) / n_atoms
    force_rmse = np.sqrt(np.mean((f_pred[0]-f_ref)**2))

    row = [frame_id, e_ref, e_pred, energy_error_per_atom, force_rmse]
    if not np.isfinite(row).all():
        raise ValueError(
            f"Non-finite or abnormal result for frame {frame_id}.")

    rows.append(row)

    if count % 100 == 0:
        print(f"Evaluated {count}/{sample_size} sampled frames")

elapsed = perf_counter() - start
results = np.array(rows)


output_folder = project / "results"
output_folder.mkdir(exist_ok=True)
output_file = output_folder / "validation_all_model000.csv"

np.savetxt(
    output_file,
    results,
    delimiter=",",
    header=(
        "frame,reference_energy_ev,predicted_energy_ev,"
        "energy_error_meV_per_atom,force_rmse_eV_per_A"
    ),
    comments="",
    fmt=["%d", "%.10f", "%.10f", "%.8f", "%.8f"],
)

energy_rmse = np.sqrt(np.mean(results[:, 3]**2))
overall_force_rmse = np.sqrt(np.mean(results[:, 4]**2))

print(f"\nValidation frames evaluated: {sample_size}")
print(f"Energy RMSE per atom: {energy_rmse:.6f} meV/atom")
print(f"Overall force-component RMSE: {overall_force_rmse:.6f} eV/Å")
print(f"Subset evaluation time: {elapsed:.3f} seconds")
print(f"Saved: {output_file}")
mean_energy_error = np.mean(results[:, 3])
print(f"Mean signed energy error: {mean_energy_error:+.6f} meV/atom")
