# qcse_truncated_cone_stark.py
# Self-contained QCSE / Stark code (NO pandas, NO scipy)
# Requirements: numpy, matplotlib
#
# Outputs:
#   - Fig1_QCSE_E0_vs_F.png
#   - Fig2_DeltaE10_vs_F.png
#   - Fig3_zbar_and_localization_vs_F.png
#   - Fig4_Novel_curvature_map.png
#   - table_baseline_vs_F.csv
#   - table_geometry_sweep.csv

import math
import csv
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Physical constants (SI)
# -----------------------------
hbar = 1.054571817e-34   # J*s
q    = 1.602176634e-19   # C
m0   = 9.1093837015e-31  # kg
eV   = q                 # J

# -----------------------------
# Utility: safe normalization
# -----------------------------
def normalize(psi, dz):
    n = np.sqrt(np.sum(np.abs(psi)**2) * dz)
    if n <= 0:
        return psi
    return psi / n

# -----------------------------
# Truncated-cone radius profile R(z)
# -----------------------------
def radius_truncated_cone(z, z_dot0, H_qd, R_base, R_top):
    """
    Linear radius from base (at z_dot0) to top (at z_dot0 + H_qd).
    Outside dot -> return 0 (means "no dot")
    """
    z_rel = z - z_dot0
    if (z_rel < 0.0) or (z_rel > H_qd):
        return 0.0
    t = z_rel / H_qd
    return R_base + (R_top - R_base) * t

# -----------------------------
# BenDaniel–Duke 1D Hamiltonian
#   H = - d/dz ( (hbar^2 / 2m(z)) d/dz ) + V(z)
# Discretization uses interface masses.
# Dirichlet boundaries (psi=0 at edges).
# -----------------------------
def build_hamiltonian(z, dz, m_of_z, V_of_z):
    N = len(z)

    # mass at half points via harmonic mean
    m_half = np.zeros(N-1)
    for i in range(N-1):
        mL = m_of_z[i]
        mR = m_of_z[i+1]
        # harmonic mean for 1/m at interface
        m_half[i] = 2.0 * mL * mR / (mL + mR)

    # tridiagonal coefficients
    main = np.zeros(N)
    off  = np.zeros(N-1)

    # kinetic prefactor at interfaces
    for i in range(N-1):
        t = (hbar**2) / (2.0 * m_half[i] * dz**2)
        off[i] = -t

    # main diag kinetic + potential
    for i in range(N):
        kin = 0.0
        if i > 0:
            kin += (hbar**2) / (2.0 * m_half[i-1] * dz**2)
        if i < N-1:
            kin += (hbar**2) / (2.0 * m_half[i] * dz**2)
        main[i] = kin + V_of_z[i]

    # build dense matrix (stable and simple; keep N modest)
    H = np.zeros((N, N), dtype=float)
    np.fill_diagonal(H, main)
    for i in range(N-1):
        H[i, i+1] = off[i]
        H[i+1, i] = off[i]
    return H

# -----------------------------
# Solve eigenpairs (dense, numpy-only)
# -----------------------------
def solve_lowest_states(H, n_states=6):
    vals, vecs = np.linalg.eigh(H)
    vals = vals[:n_states]
    vecs = vecs[:, :n_states]
    return vals, vecs

# -----------------------------
# Localization metric: probability inside dot+wetting region
# -----------------------------
def prob_in_region(z, psi, dz, zmin, zmax):
    mask = (z >= zmin) & (z <= zmax)
    return float(np.sum(np.abs(psi[mask])**2) * dz)

def expectation_z(z, psi, dz):
    return float(np.sum((np.abs(psi)**2) * z) * dz)

# -----------------------------
# Main QCSE simulation for one geometry
# -----------------------------
def run_qcse_for_geometry(
    R_base_nm=5.0,
    R_top_nm=3.0,
    H_qd_nm=7.0,
    t_wet_nm=5.0,
    Vb_eV=0.5,
    me_GaAs=0.067*m0,
    me_InAs=0.023*m0,
    Lz_nm=200.0,
    Nz=700,
    F_kVcm_max=200.0,
    dF_kVcm=5.0,
    fit_Fmax_kVcm=80.0,
    center_dot_nm=100.0
):
    # Units
    R_base = R_base_nm * 1e-9
    R_top  = R_top_nm  * 1e-9
    H_qd   = H_qd_nm   * 1e-9
    t_wet  = t_wet_nm  * 1e-9
    Vb     = Vb_eV     * eV

    Lz     = Lz_nm * 1e-9
    z0     = center_dot_nm * 1e-9

    # Domain in z
    z = np.linspace(0.0, Lz, Nz)
    dz = z[1] - z[0]

    # Dot + wetting positions
    z_wet0 = z0 - (t_wet/2.0) - (H_qd/2.0)  # wetting below dot base (simple stack)
    z_dot0 = z_wet0 + t_wet                  # dot base starts above wetting
    z_dot1 = z_dot0 + H_qd
    z_wet1 = z_wet0 + t_wet

    # Radius and confinement term:
    # Adiabatic radial confinement energy ~ (hbar^2 * alpha01^2)/(2 m * R(z)^2)
    # alpha01 (first zero of J0) ~ 2.4048 for cylindrical approx (good axisymmetric proxy)
    alpha01 = 2.4048255577

    # Field sweep (include +/- to get true Stark parabola)
    F_list_kVcm = np.arange(-F_kVcm_max, F_kVcm_max + 1e-12, dF_kVcm)
    F_list_SI   = F_list_kVcm * 1e5  # 1 kV/cm = 1e5 V/m

    # Storage
    E0 = []
    E1 = []
    DE10 = []
    zbar0 = []
    Pin0 = []
    Pin1 = []

    # Precompute static dot profile arrays (geometry-only)
    Rz = np.zeros_like(z)
    inside_dot = np.zeros_like(z, dtype=bool)
    inside_wet = (z >= z_wet0) & (z <= z_wet1)

    for i, zi in enumerate(z):
        ri = radius_truncated_cone(zi, z_dot0, H_qd, R_base, R_top)
        Rz[i] = ri
        inside_dot[i] = (ri > 0.0)

    inside_active = inside_dot | inside_wet
    z_active_min = float(np.min(z[inside_active]))
    z_active_max = float(np.max(z[inside_active]))

    # Effective mass profile
    m_of_z = np.where(inside_active, me_InAs, me_GaAs).astype(float)

    # Base potential well (electron):
    # outside barrier = 0, inside dot+wetting = -Vb (conduction band offset proxy)
    V0 = np.where(inside_active, -Vb, 0.0).astype(float)

    # Add radial confinement energy inside dot only (wetting treated as 2D film -> no radial term)
    Erad = np.zeros_like(z, dtype=float)
    for i in range(Nz):
        if inside_dot[i]:
            Ri = max(Rz[i], 0.6e-9)  # avoid singularity if R becomes too small
            Erad[i] = (hbar**2 * alpha01**2) / (2.0 * me_InAs * (Ri**2))
    Vgeom = V0 + Erad

    # Solve for each field
    # Robust bound-state selection:
    # - compute a handful of lowest eigenstates
    # - choose ground/excited as those with highest localization inside active region among low energies
    for F in F_list_SI:
        # Electric potential energy term (electron): +q*F*(z - z_center)
        # If you want opposite sign, flip F or flip (z-z0). Keep consistent across plots.
        Vfield = (q * F * (z - z0)).astype(float)

        V = Vgeom + Vfield
        H = build_hamiltonian(z, dz, m_of_z, V)

        vals, vecs = solve_lowest_states(H, n_states=10)

        # Normalize eigenvectors
        for k in range(vecs.shape[1]):
            vecs[:, k] = normalize(vecs[:, k], dz)

        # Compute localization for each candidate
        loc = np.array([prob_in_region(z, vecs[:, k], dz, z_active_min, z_active_max) for k in range(vecs.shape[1])])

        # Prefer bound + localized states:
        # sort by (high localization, low energy)
        # localization threshold avoids edge states when field is strong
        candidates = list(range(len(vals)))
        candidates.sort(key=lambda k: (-loc[k], vals[k]))

        k0 = candidates[0]
        # excited: next best localized state different from k0
        k1 = candidates[1] if candidates[1] != k0 else candidates[2]

        E0.append(vals[k0])
        E1.append(vals[k1])
        DE10.append(vals[k1] - vals[k0])

        zbar0.append(expectation_z(z, vecs[:, k0], dz))
        Pin0.append(loc[k0])
        Pin1.append(loc[k1])

    # Convert to convenient units
    E0 = np.array(E0) / eV * 1e3  # meV
    E1 = np.array(E1) / eV * 1e3
    DE10 = np.array(DE10) / eV * 1e3

    zbar0 = (np.array(zbar0) - z0) * 1e9  # nm relative to center
    Pin0 = np.array(Pin0)
    Pin1 = np.array(Pin1)

    # Low-field QCSE fit of E0(F): use |F| <= fit_Fmax_kVcm
    mask_fit = np.abs(F_list_kVcm) <= fit_Fmax_kVcm
    F_fit_SI = F_list_SI[mask_fit]
    E0_fit_eV = (E0[mask_fit] / 1e3)  # eV

    # Fit E0(F) = c0 + c1*F + c2*F^2
    c2, c1, c0 = np.polyfit(F_fit_SI, E0_fit_eV, 2)

    # QCSE parameters:
    # ΔE = -p*F - (1/2)α F^2  => compare to (c1*F + c2*F^2)
    p_dipole = -c1 * eV / (1.0)          # J/(V/m) = C*m
    alpha_pol = -2.0 * c2 * eV           # J/(V/m)^2 = C*m^2/V

    # Generate smooth fit curve for plotting
    E0_fit_curve = (c0 + c1*F_list_SI + c2*(F_list_SI**2)) * 1e3  # meV

    results = {
        "F_kVcm": F_list_kVcm,
        "E0_meV": E0,
        "E1_meV": E1,
        "DE10_meV": DE10,
        "zbar_nm": zbar0,
        "Pin0": Pin0,
        "Pin1": Pin1,
        "E0_fit_curve_meV": E0_fit_curve,
        "fit_coeffs": (c0, c1, c2),
        "p_Cm": p_dipole,
        "alpha_SI": alpha_pol,
        "geom": {
            "R_base_nm": R_base_nm,
            "R_top_nm": R_top_nm,
            "H_qd_nm": H_qd_nm,
            "t_wet_nm": t_wet_nm,
            "Vb_eV": Vb_eV,
            "Lz_nm": Lz_nm,
            "Nz": Nz,
            "fit_Fmax_kVcm": fit_Fmax_kVcm
        }
    }
    return results

# -----------------------------
# Plotting (4 figures)
# -----------------------------
def make_figures_and_tables(res, out_prefix=""):
    F = res["F_kVcm"]
    E0 = res["E0_meV"]
    E1 = res["E1_meV"]
    DE10 = res["DE10_meV"]
    zbar = res["zbar_nm"]
    Pin0 = res["Pin0"]
    fit = res["E0_fit_curve_meV"]
    geom = res["geom"]

    # Figure 1: E0 vs F with low-field QCSE fit
    plt.figure(figsize=(9, 6))
    plt.plot(F, E0, marker="o", linewidth=2, label="Tracked bound E0(F)")
    plt.plot(F, fit, linestyle="--", linewidth=2, label=f"QCSE fit (|F| ≤ {geom['fit_Fmax_kVcm']} kV/cm)")
    plt.xlabel("Electric field F (kV/cm)")
    plt.ylabel("Electron ground energy E0 (meV)")
    plt.title("QCSE Stark shift (bound-state filtered, truncated-cone QD)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "Fig1_QCSE_E0_vs_F.png", dpi=220)

    # Figure 2: Level spacing vs F
    plt.figure(figsize=(9, 6))
    plt.plot(F, DE10, marker="s", linewidth=2, label="ΔE10 = E1 − E0 (bound-state filtered)")
    plt.xlabel("Electric field F (kV/cm)")
    plt.ylabel("ΔE10 (meV)")
    plt.title("Field-tunable confinement (ΔE10 vs F)")
    plt.grid(True, alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_prefix + "Fig2_DeltaE10_vs_F.png", dpi=220)

    # Figure 3: centroid shift + localization
    plt.figure(figsize=(9, 6))
    plt.plot(F, zbar, marker="o", linewidth=2, label="⟨z⟩ − z0 (nm)")
    plt.xlabel("Electric field F (kV/cm)")
    plt.ylabel("Ground-state displacement (nm)")
    plt.title("Wavefunction displacement under field (with localization control)")
    plt.grid(True, alpha=0.35)

    ax = plt.gca()
    ax2 = ax.twinx()
    ax2.plot(F, Pin0, linestyle="--", linewidth=2, label="P(in QD+WL)")
    ax2.set_ylabel("Localization probability")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="best")
    plt.tight_layout()
    plt.savefig(out_prefix + "Fig3_zbar_and_localization_vs_F.png", dpi=220)

    # Tables: baseline vs F
    with open(out_prefix + "table_baseline_vs_F.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "F_kVcm", "E0_meV", "E1_meV", "DeltaE10_meV",
            "zbar_nm", "Pin0_localization", "Pin1_localization"
        ])
        for i in range(len(F)):
            w.writerow([float(F[i]), float(E0[i]), float(E1[i]), float(DE10[i]), float(zbar[i]), float(Pin0[i]), float(res["Pin1"][i])])

    # Print summary (copy-pasteable)
    c0, c1, c2 = res["fit_coeffs"]
    print("\n=== QCSE FIT SUMMARY (low-field) ===")
    print(f"Geometry: Rb={geom['R_base_nm']} nm, Rt={geom['R_top_nm']} nm, H={geom['H_qd_nm']} nm, WL={geom['t_wet_nm']} nm, Vb={geom['Vb_eV']} eV")
    print(f"Fit model: E0(F)=c0 + c1*F + c2*F^2  (F in V/m, E in eV)")
    print(f"c0 = {c0:.6e} eV")
    print(f"c1 = {c1:.6e} eV/(V/m)")
    print(f"c2 = {c2:.6e} eV/(V/m)^2")
    print(f"Extracted dipole p = {res['p_Cm']:.6e} C·m")
    print(f"Extracted polarizability alpha = {res['alpha_SI']:.6e} (SI: C·m^2/V)")
    print(f"Saved: {out_prefix}table_baseline_vs_F.csv")
    print(f"Saved: {out_prefix}Fig1_QCSE_E0_vs_F.png")
    print(f"Saved: {out_prefix}Fig2_DeltaE10_vs_F.png")
    print(f"Saved: {out_prefix}Fig3_zbar_and_localization_vs_F.png")

# -----------------------------
# Novel Figure 4: curvature map vs geometry
#   We sweep (H, truncation ratio Rt/Rb) and extract |c2| from low-field fit.
# -----------------------------
def curvature_map(
    R_base_nm=5.0,
    t_wet_nm=5.0,
    Vb_eV=0.5,
    H_list_nm=(6.0, 7.0, 8.0, 9.0, 10.0, 11.0),
    ratio_list=(0.4, 0.5, 0.6, 0.7, 0.8),
    Fmax_kVcm=120.0,
    dF_kVcm=10.0,
    fit_Fmax_kVcm=60.0
):
    H_list_nm = np.array(H_list_nm, dtype=float)
    ratio_list = np.array(ratio_list, dtype=float)

    K = np.zeros((len(H_list_nm), len(ratio_list)), dtype=float)  # store |c2| mapped to meV/(kV/cm)^2 for readability

    for i, Hnm in enumerate(H_list_nm):
        for j, r in enumerate(ratio_list):
            Rt = float(R_base_nm * r)
            res = run_qcse_for_geometry(
                R_base_nm=R_base_nm,
                R_top_nm=Rt,
                H_qd_nm=float(Hnm),
                t_wet_nm=t_wet_nm,
                Vb_eV=Vb_eV,
                F_kVcm_max=float(Fmax_kVcm),
                dF_kVcm=float(dF_kVcm),
                fit_Fmax_kVcm=float(fit_Fmax_kVcm),
                Nz=650,
                Lz_nm=200.0,
                center_dot_nm=100.0
            )
            # convert c2 (eV/(V/m)^2) to meV/(kV/cm)^2:
            # 1 (kV/cm)^2 = (1e5 V/m)^2 = 1e10 (V/m)^2
            c0, c1, c2 = res["fit_coeffs"]
            c2_meV_per_kVcm2 = abs(c2 * 1e3 * 1e10)
            K[i, j] = c2_meV_per_kVcm2

    # Save sweep table
    with open("table_geometry_sweep.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = ["H_nm \\ ratio_RtRb"] + [float(r) for r in ratio_list]
        w.writerow(header)
        for i, Hnm in enumerate(H_list_nm):
            w.writerow([float(Hnm)] + [float(K[i, j]) for j in range(len(ratio_list))])

    # Figure 4: heatmap
    plt.figure(figsize=(9, 6))
    plt.imshow(
        K,
        origin="lower",
        aspect="auto",
        extent=[ratio_list.min(), ratio_list.max(), H_list_nm.min(), H_list_nm.max()]
    )
    plt.colorbar(label="|QCSE curvature|  |c2|  (meV / (kV/cm)^2)")
    plt.xlabel("Truncation ratio Rt/Rb")
    plt.ylabel("QD height H (nm)")
    plt.title("Novel metric: QCSE curvature map vs geometry (low-field fit)")
    plt.tight_layout()
    plt.savefig("Fig4_Novel_curvature_map.png", dpi=220)

    print("Saved: table_geometry_sweep.csv")
    print("Saved: Fig4_Novel_curvature_map.png")

# -----------------------------
# Run everything
# -----------------------------
def main():
    # Baseline geometry (match your COMSOL params screenshot)
    res = run_qcse_for_geometry(
        R_base_nm=5.0,
        R_top_nm=3.0,
        H_qd_nm=7.0,
        t_wet_nm=5.0,
        Vb_eV=0.5,
        F_kVcm_max=200.0,
        dF_kVcm=5.0,
        fit_Fmax_kVcm=80.0,
        Lz_nm=200.0,
        Nz=700,
        center_dot_nm=100.0
    )
    make_figures_and_tables(res, out_prefix="")

    # Novel geometry map (paper-ready extra)
    curvature_map(
        R_base_nm=5.0,
        t_wet_nm=5.0,
        Vb_eV=0.5,
        H_list_nm=(6, 7, 8, 9, 10, 11),
        ratio_list=(0.4, 0.5, 0.6, 0.7, 0.8),
        Fmax_kVcm=120.0,
        dF_kVcm=10.0,
        fit_Fmax_kVcm=60.0
    )

    plt.show()

if __name__ == "__main__":
    main()