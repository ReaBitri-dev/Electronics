# directional_stark_qd.py
# Directional Stark engineering in a quantum dot (Python-only)
# Model: electron in a 3D anisotropic infinite box (Lx, Ly, Lz)
# Method: basis expansion + Hamiltonian diagonalization
#
# Outputs:
# 1) E(F) for 3 field directions
# 2) Dipole moment vs field
# 3) Absorption spectrum vs photon energy (from dipole matrix elements)

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.constants import hbar, m_e, e, electron_volt, pi
from scipy.linalg import eigh

# -----------------------------
# Utilities: units
# -----------------------------
def nm(x):  # nm -> m
    return x * 1e-9

def meV(x):  # eV -> meV
    return x * 1e3

def J_to_eV(x):
    return x / electron_volt

def eV_to_J(x):
    return x * electron_volt

# -----------------------------
# Basis and matrix elements
# -----------------------------
def energy_infinite_box_1d(n, L, m_eff):
    """1D infinite well energy in Joules."""
    return (n**2 * pi**2 * hbar**2) / (2 * m_eff * L**2)

def x_matrix_1d(n, np_, L):
    """
    Matrix element <n|x|n'> for 1D infinite well on [0,L].
    Known analytic form:
      if n == n': L/2
      if n != n':
          <n|x|n'> = ( (-1)^(n-n') - 1 ) * (2 L / (pi^2)) * (n n') / ( (n^2 - n'^2)^2 ) * L?  (careful)
    We'll use a safe exact integral formula derived from sine basis.
    Basis: phi_n(x)=sqrt(2/L)*sin(n*pi*x/L).
    """
    if n == np_:
        return L / 2.0

    # Exact integral:
    # <n|x|m> = (2/L) * ∫_0^L x sin(nπx/L) sin(mπx/L) dx
    # Closed form:
    # = (2L/(π^2)) * [ ((-1)^(n-m) - 1)/(n-m)^2 - ((-1)^(n+m) - 1)/(n+m)^2 ] / 2
    # Let's implement carefully.
    n = int(n); m = int(np_)
    term1 = ((-1)**(n - m) - 1.0) / ((n - m)**2)
    term2 = ((-1)**(n + m) - 1.0) / ((n + m)**2)
    return (L / (pi**2)) * (term1 - term2)

def build_3d_basis(nmax_x, nmax_y, nmax_z):
    """List of (nx, ny, nz)."""
    basis = []
    for nx in range(1, nmax_x + 1):
        for ny in range(1, nmax_y + 1):
            for nz in range(1, nmax_z + 1):
                basis.append((nx, ny, nz))
    return basis

@dataclass
class QDBoxModel:
    Lx_m: float
    Ly_m: float
    Lz_m: float
    m_eff: float  # effective mass in kg
    nmax_x: int = 6
    nmax_y: int = 6
    nmax_z: int = 6

    def __post_init__(self):
        self.basis = build_3d_basis(self.nmax_x, self.nmax_y, self.nmax_z)
        self.N = len(self.basis)
        self._build_H0()
        self._build_xz_operators()

    def _build_H0(self):
        """Diagonal H0 in basis (infinite box)."""
        H0 = np.zeros((self.N, self.N), dtype=float)
        for i, (nx, ny, nz) in enumerate(self.basis):
            Ex = energy_infinite_box_1d(nx, self.Lx_m, self.m_eff)
            Ey = energy_infinite_box_1d(ny, self.Ly_m, self.m_eff)
            Ez = energy_infinite_box_1d(nz, self.Lz_m, self.m_eff)
            H0[i, i] = Ex + Ey + Ez
        self.H0 = H0

    def _build_xz_operators(self):
        """Build X and Z operators in 3D basis."""
        X = np.zeros((self.N, self.N), dtype=float)
        Z = np.zeros((self.N, self.N), dtype=float)

        # Precompute 1D matrices
        nx_list = range(1, self.nmax_x + 1)
        ny_list = range(1, self.nmax_y + 1)
        nz_list = range(1, self.nmax_z + 1)

        X1 = np.zeros((self.nmax_x, self.nmax_x))
        Z1 = np.zeros((self.nmax_z, self.nmax_z))
        for a, n in enumerate(nx_list):
            for b, m in enumerate(nx_list):
                X1[a, b] = x_matrix_1d(n, m, self.Lx_m)
        for a, n in enumerate(nz_list):
            for b, m in enumerate(nz_list):
                Z1[a, b] = x_matrix_1d(n, m, self.Lz_m)  # same formula for coordinate on [0,Lz]

        # Fill 3D operators using separability
        # <nx,ny,nz | x | nx',ny',nz'> = <nx|x|nx'> * δ(ny,ny') * δ(nz,nz')
        # similarly for z
        for i, (nx, ny, nz) in enumerate(self.basis):
            for j, (nxp, nyp, nzp) in enumerate(self.basis):
                if ny == nyp and nz == nzp:
                    X[i, j] = X1[nx - 1, nxp - 1]
                if ny == nyp and nx == nxp:
                    Z[i, j] = Z1[nz - 1, nzp - 1]

        self.X = X
        self.Z = Z

    def solve_for_field(self, Fx_Vpm, Fz_Vpm, n_states=6):
        """
        Hamiltonian: H = H0 + e*(Fx*x + Fz*z) for electron?
        Potential energy: U = -e E·r, so add (-e)*(E·r).
        Here E = (Fx, Fz), r=(x,z) => U = -e(Fx*x + Fz*z).
        """
        # Field term in Joules: -e * E·r
        H = self.H0 + (-e) * (Fx_Vpm * self.X + Fz_Vpm * self.Z)

        # Solve
        evals, evecs = eigh(H)
        evals = evals[:n_states]
        evecs = evecs[:, :n_states]

        # Expectation values for dipole / position
        # <x>_k = v^T X v  (real basis; if complex, use v* X v)
        x_exp = np.einsum("ik,ij,jk->k", evecs, self.X, evecs)
        z_exp = np.einsum("ik,ij,jk->k", evecs, self.Z, evecs)

        return evals, evecs, x_exp, z_exp

def absorption_spectrum(evals, evecs, Xop, Zop, Ehat, broadening_meV=3.0,
                        Emin_meV=0.0, Emax_meV=120.0, npts=2000, n_trans=8):
    """
    Compute toy absorption from ground state -> excited states using dipole matrix element:
      M = <0| r·e_pol |j>
    Here we take polarization along field direction Ehat (can be separate if you want).
    Intensity ~ |M|^2 and delta(Ej-E0 - hν) broadened with Gaussian.
    """
    # energies in Joules
    E0 = evals[0]
    # direction operator
    R = Ehat[0] * Xop + Ehat[1] * Zop

    # Build spectrum axis in meV (photon energy)
    hw_meV = np.linspace(Emin_meV, Emax_meV, npts)
    hw_J = eV_to_J(hw_meV / 1000.0)

    sigma_J = eV_to_J(broadening_meV / 1000.0)
    spec = np.zeros_like(hw_J)

    # Dipole transitions
    v0 = evecs[:, 0]
    for j in range(1, min(n_trans, evecs.shape[1])):
        vj = evecs[:, j]
        M = v0 @ (R @ vj)  # <0|R|j>
        I = (M**2)  # real here
        dE = (evals[j] - E0)
        # Gaussian broadening around photon energy = dE
        spec += I * np.exp(-0.5 * ((hw_J - dE) / sigma_J)**2)

    # Normalize for plotting
    if spec.max() > 0:
        spec /= spec.max()
    return hw_meV, spec

# -----------------------------
# Main run
# -----------------------------
def main():
    # ====== Dot geometry (edit these) ======
    # Anisotropic "heterostructure-like" dot: make Lz different so orientation matters
    Lx, Ly, Lz = nm(12.0), nm(12.0), nm(6.0)

    # Effective mass (example: InAs electron ~0.023 m0; GaAs ~0.067 m0)
    m_eff = 0.05 * m_e

    # Basis size: increase for accuracy; keep moderate for speed
    model = QDBoxModel(Lx, Ly, Lz, m_eff, nmax_x=7, nmax_y=5, nmax_z=6)

    # ====== Field sweep (edit these) ======
    # Use V/m. For reference: 50 kV/cm = 5e6 V/m
    Fmax = 8e6
    nF = 41
    F_list = np.linspace(-Fmax, Fmax, nF)

    # Directions: (Fx,Fz) via theta in x-z plane
    dirs = {
        "Fx (lateral)": 0.0,
        "F45°": np.pi / 4.0,
        "Fz (vertical)": np.pi / 2.0,
    }

    n_states = 6  # how many levels to track

    # Store results
    results = {}

    for name, theta in dirs.items():
        Ehat = np.array([np.cos(theta), np.sin(theta)])  # (x,z)
        E_levels = np.zeros((nF, n_states))
        x0 = np.zeros(nF); z0 = np.zeros(nF)
        ppar0 = np.zeros(nF)

        for iF, Fmag in enumerate(F_list):
            Fx = Fmag * np.cos(theta)
            Fz = Fmag * np.sin(theta)
            evals, evecs, x_exp, z_exp = model.solve_for_field(Fx, Fz, n_states=n_states)

            E_levels[iF, :] = J_to_eV(evals)  # eV
            x0[iF] = x_exp[0]
            z0[iF] = z_exp[0]
            ppar0[iF] = -e * (x_exp[0] * Ehat[0] + z_exp[0] * Ehat[1])  # C·m (dipole along field)

        results[name] = dict(
            theta=theta,
            Ehat=Ehat,
            E_levels=E_levels,
            x0=x0, z0=z0,
            ppar0=ppar0
        )

    # -----------------------------
    # Plot 1: Stark curves E(F) ground state for 3 directions
    # -----------------------------
    plt.figure()
    for name in dirs.keys():
        E0 = results[name]["E_levels"][:, 0]
        plt.plot(F_list, meV(E0 - E0[nF//2]), label=name)  # shift to center value
    plt.xlabel("Field magnitude F (V/m)")
    plt.ylabel("Ground-state shift ΔE0 (meV) [center-shifted]")
    plt.title("Directional Stark shift (ground state)")
    plt.legend()
    plt.tight_layout()

    # -----------------------------
    # Plot 2: Dipole moment vs field (along field) for 3 directions
    # -----------------------------
    plt.figure()
    for name in dirs.keys():
        p = results[name]["ppar0"]
        # plot in e·nm units for intuition: p / (e * 1 nm)
        p_enm = p / (e * 1e-9)
        plt.plot(F_list, p_enm, label=name)
    plt.xlabel("Field magnitude F (V/m)")
    plt.ylabel("Dipole (along field)  p∥  [e·nm]")
    plt.title("Dipole moment vs field (ground state)")
    plt.legend()
    plt.tight_layout()

    # -----------------------------
    # Plot 3: Absorption spectrum at a chosen field magnitude
    # -----------------------------
    # Choose one field point (e.g., positive max)
    idx = -1  # last point = +Fmax
    plt.figure()
    for name, theta in dirs.items():
        Fmag = F_list[idx]
        Fx = Fmag * np.cos(theta)
        Fz = Fmag * np.sin(theta)
        evals, evecs, _, _ = model.solve_for_field(Fx, Fz, n_states=10)
        hw_meV, spec = absorption_spectrum(
            evals, evecs, model.X, model.Z, results[name]["Ehat"],
            broadening_meV=4.0, Emin_meV=0.0, Emax_meV=180.0, npts=2500, n_trans=10
        )
        plt.plot(hw_meV, spec, label=name)

    plt.xlabel("Photon energy (meV)")
    plt.ylabel("Normalized absorption (a.u.)")
    plt.title(f"Toy absorption spectra at F = {F_list[idx]:.2e} V/m (polarization along field)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()