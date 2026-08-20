"""
simulate_tem_pattern.py

Simulates a kinematic TEM electron diffraction pattern for an arbitrary
crystal structure (given as a .cif file) and an arbitrary user-defined
beam direction (zone axis).

USAGE:
    python3 simulate_tem_pattern.py path/to/structure.cif --zone 0 0 1

METHOD OVERVIEW (see accompanying README.md for full derivation):
    1. Parse the CIF and expand the space-group symmetry to get the full
       list of atoms in the unit cell (done by pymatgen).
    2. Build the reciprocal lattice vectors a*, b*, c* from the real
       lattice vectors (general formula, works for any crystal system).
    3. For a range of candidate Miller indices (h,k,l):
         a. Apply the zone law h*u + k*v + l*w = 0 to keep only
            reflections visible for the chosen beam direction [uvw].
         b. Compute the structure factor
                F_hkl = sum_i f_i * exp(2*pi*i*(h*x_i + k*y_i + l*z_i))
            directly from the atom basis, using electron atomic form
            factors f_i from scikit-ued (Kirkland 2010 parameterization).
         c. Discard reflections with |F_hkl| ~ 0 (systematic absences).
    4. Project each surviving reciprocal vector g_hkl onto the 2D plane
       perpendicular to the beam direction to get spot (x, y) positions.
       Spot intensity ~ |F_hkl|^2.
    5. Plot the result as a spot pattern.

DEPENDENCIES:
    pip install pymatgen scikit-ued matplotlib numpy

DOCUMENTED SOURCE FOR ELECTRON SCATTERING FACTORS:
    scikit-ued's `affe()` function (skued.simulation.form_factors), which
    implements the parameterization from:
        Kirkland, E. J. (2010). Advanced Computing in Electron Microscopy
        (2nd ed.). Springer.
    This is standard in the electron microscopy community and is distinct
    from X-ray form factor tables (e.g. the ones bundled with pymatgen by
    default), which would give the wrong f_i for electron diffraction.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pymatgen.core import Structure

# Electron atomic scattering factor parameterization (Kirkland, E. J. (2010).
# Advanced Computing in Electron Microscopy, 2nd ed., Springer). These are
# the same numeric parameters scikit-ued's `affe()` uses internally -- pulled
# in directly here to avoid scikit-ued's dependency chain (scikit-ued ->
# crystals -> PyCifRW), which requires compiling a C extension on Windows.
# Add more elements to this dict as needed. Format is scikit-ued's raw table
# row: [placeholder, a1,b1,a2,b2,a3,b3,c1,d1,c2,d2,c3,d3] -- the first entry
# is unused (discarded below), matching scikit-ued's own unpacking convention.
_KIRKLAND_PARAMS = {
    "Ti": [3.52980000e-02, 3.62383267e-01, 7.54707114e-02, 9.84232966e-01,
           4.97757309e-01, 7.41715642e-01, 8.17659391e+00, 3.62555269e-01,
           9.55524906e-01, 1.49159390e+00, 1.62221677e+01, 1.61659509e-02,
           7.33140839e-02],
}


def affe(element, nG_array):
    """Electron atomic form factor for a neutral atom (Kirkland parameterization).
    nG_array: scattering vector norm |G| = 4*pi*s, in Angstrom^-1.
    """
    _, a1, b1, a2, b2, a3, b3, c1, d1, c2, d2, c3, d3 = _KIRKLAND_PARAMS[element]
    q = np.asarray(nG_array) / (2 * np.pi)
    q2 = q ** 2
    sum1 = a1 / (q2 + b1) + a2 / (q2 + b2) + a3 / (q2 + b3)
    sum2 = c1 * np.exp(-d1 * q2) + c2 * np.exp(-d2 * q2) + c3 * np.exp(-d3 * q2)
    return sum1 + sum2


def load_structure(cif_path):
    """Parse CIF and expand symmetry -> full atom basis in the unit cell."""
    structure = Structure.from_file(cif_path)
    atoms = [(str(site.specie), site.frac_coords) for site in structure]
    return structure, atoms


def reciprocal_lattice(structure):
    """General reciprocal lattice vectors a*, b*, c* (A^-1), any crystal system."""
    a_vec, b_vec, c_vec = structure.lattice.matrix
    V = np.dot(a_vec, np.cross(b_vec, c_vec))
    a_star = np.cross(b_vec, c_vec) / V
    b_star = np.cross(c_vec, a_vec) / V
    c_star = np.cross(a_vec, b_vec) / V
    return a_star, b_star, c_star


def structure_factor(h, k, l, atoms, a_star, b_star, c_star):
    """F_hkl = sum_i f_i * exp(2*pi*i*(h*x_i + k*y_i + l*z_i)), computed
    directly from the atomic basis -- no crystal-specific shortcuts."""
    total = 0j
    g_hkl = h * a_star + k * b_star + l * c_star
    nG = 2 * np.pi * np.linalg.norm(g_hkl)
    for element, (x, y, z) in atoms:
        f_i = affe(element, np.array([nG if nG > 0 else 1e-6]))[0]
        phase = 2 * np.pi * (h * x + k * y + l * z)
        total += f_i * np.exp(1j * phase)
    return total


def zone_axis_basis(uvw, structure):
    """Two orthonormal vectors spanning the plane perpendicular to the
    beam direction [uvw] (given in real-space lattice coordinates)."""
    a_vec, b_vec, c_vec = structure.lattice.matrix
    beam = uvw[0] * a_vec + uvw[1] * b_vec + uvw[2] * c_vec
    beam = beam / np.linalg.norm(beam)

    helper = np.array([1.0, 0.0, 0.0])
    if abs(np.dot(helper, beam)) > 0.9:
        helper = np.array([0.0, 1.0, 0.0])
    e1 = helper - np.dot(helper, beam) * beam
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.cross(beam, e1)
    return e1, e2


def simulate_pattern(cif_path, uvw, hkl_range=4, tol=1e-2):
    """Full pipeline: returns list of (h, k, l, x, y, intensity)."""
    structure, atoms = load_structure(cif_path)
    a_star, b_star, c_star = reciprocal_lattice(structure)
    e1, e2 = zone_axis_basis(uvw, structure)

    u, v, w = uvw
    spots = []
    for h in range(-hkl_range, hkl_range + 1):
        for k in range(-hkl_range, hkl_range + 1):
            for l in range(-hkl_range, hkl_range + 1):
                if h == 0 and k == 0 and l == 0:
                    continue
                if h * u + k * v + l * w != 0:          # zone law
                    continue
                F = structure_factor(h, k, l, atoms, a_star, b_star, c_star)
                if abs(F) <= tol:                        # systematic absence
                    continue
                g_hkl = h * a_star + k * b_star + l * c_star
                x = np.dot(g_hkl, e1)
                y = np.dot(g_hkl, e2)
                spots.append((h, k, l, x, y, abs(F) ** 2))
    return spots


def plot_pattern(spots, uvw, out_path=None, label_spots=True):
    xs = [s[3] for s in spots]
    ys = [s[4] for s in spots]
    intensities = np.array([s[5] for s in spots])
    sizes = 40 + 260 * (intensities / intensities.max())

    fig, ax = plt.subplots(figsize=(6, 6), facecolor="black")
    ax.set_facecolor("black")
    ax.scatter(xs, ys, s=sizes, c="white", alpha=0.95, edgecolors="none")
    ax.scatter([0], [0], s=90, c="white", marker="x")

    if label_spots:
        for h, k, l, x, y, I in spots:
            if I > 0.15 * intensities.max():
                ax.annotate(f"{h}{k}{l}", (x, y), textcoords="offset points",
                            xytext=(6, 6), fontsize=8, color="yellow")

    ax.set_aspect("equal")
    lim = max(max(map(abs, xs)), max(map(abs, ys))) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_title(f"Simulated electron diffraction pattern, zone axis {uvw}", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    ax.set_xlabel("A$^{-1}$", color="white")
    ax.set_ylabel("A$^{-1}$", color="white")

    if out_path:
        fig.savefig(out_path, dpi=150, facecolor="black", bbox_inches="tight")
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cif_path", help="Path to the .cif structure file")
    parser.add_argument("--zone", nargs=3, type=int, default=[0, 0, 1],
                         metavar=("U", "V", "W"), help="Zone axis [uvw], e.g. --zone 0 0 1")
    parser.add_argument("--range", type=int, default=4, help="Max |h|,|k|,|l| to scan")
    parser.add_argument("--out", default="diffraction_pattern.png", help="Output image path")
    args = parser.parse_args()

    spots = simulate_pattern(args.cif_path, tuple(args.zone), hkl_range=args.range)
    print(f"Found {len(spots)} allowed reflections for zone axis {tuple(args.zone)}")
    plot_pattern(spots, tuple(args.zone), out_path=args.out)
    print(f"Saved pattern to {args.out}")