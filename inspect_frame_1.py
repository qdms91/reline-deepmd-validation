# Program 1
from pathlib import Path
import numpy as np

# Locate the data in relation to this program.
# Here Here, Path(__file__).resolve().parent determines the directory where the current Python
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
# listed in type_names we have. For example, if we have 3 water
# molecules in type_raw which relate to the assigned integers
# in type_map.raw, then we expect to print H:6 and O:3.
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
