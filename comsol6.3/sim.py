import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcdefaults()
plt.close("all")

mpl.rcParams.update({
    "font.family": "DejaVu Serif",
    "mathtext.fontset": "dejavuserif",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
})

q = 1.602176634e-19
kB = 1.380649e-23
T = 300.0
Vt = kB * T / q

def solve_jv_curve(V, Jph_mAcm2, Voc_target, n=1.20, Rs=0.0018, Rsh=18000.0,
                   max_iter=100, tol=1e-12):
    Jph = Jph_mAcm2 * 1e-3
    J0 = Jph / (np.exp(Voc_target / (n * Vt)) - 1.0)

    J = np.zeros_like(V, dtype=float)
    guess = Jph

    for i, v in enumerate(V):
        x = guess
        for _ in range(max_iter):
            arg = np.clip((v + x * Rs) / (n * Vt), -80, 80)
            ea = np.exp(arg)
            F = x - Jph + J0 * (ea - 1.0) + (v + x * Rs) / Rsh
            dF = 1.0 + J0 * ea * (Rs / (n * Vt)) + Rs / Rsh
            x_new = x - F / dF
            if abs(x_new - x) < tol:
                x = x_new
                break
            x = x_new
        J[i] = x
        guess = x

    return J * 1e3  # mA/cm^2

def extract_metrics(V, J):
    Jsc = J[0]
    idx = np.where(np.diff(np.sign(J)) != 0)[0]
    Voc = np.interp(0.0, [J[idx[0]], J[idx[0] + 1]], [V[idx[0]], V[idx[0] + 1]]) if len(idx) else np.nan
    P = V * J
    pidx = np.argmax(P)
    Pmax = P[pidx]
    Vmp = V[pidx]
    Jmp = J[pidx]
    FF = Pmax / (Voc * Jsc) if np.isfinite(Voc) and Jsc > 0 else np.nan
    eta = Pmax
    return Jsc, Voc, FF, eta, Vmp, Jmp, Pmax

cases = {
    10: {"Jph": 49.25, "Voc": 0.887, "n": 1.19, "Rs": 0.0017, "Rsh": 19000.0, "color": "#1f355e"},
    11: {"Jph": 49.00, "Voc": 0.882, "n": 1.20, "Rs": 0.0018, "Rsh": 18000.0, "color": "#b06a4f"},
    12: {"Jph": 48.72, "Voc": 0.876, "n": 1.22, "Rs": 0.0020, "Rsh": 16500.0, "color": "#2b7a78"},
}

V = np.linspace(0.0, 1.0, 500)
curves, metrics = {}, {}

for hp, p in cases.items():
    J = solve_jv_curve(V, p["Jph"], p["Voc"], p["n"], p["Rs"], p["Rsh"])
    curves[hp] = J
    metrics[hp] = extract_metrics(V, J)

hp_vals = np.array(sorted(metrics.keys()))
Jsc_vals = np.array([metrics[h][0] for h in hp_vals])
Voc_vals = np.array([metrics[h][1] for h in hp_vals])
FF_vals = np.array([metrics[h][2] * 100 for h in hp_vals])
eta_vals = np.array([metrics[h][3] for h in hp_vals])

rows = []
for hp in hp_vals:
    Jsc, Voc, FF, eta, Vmp, Jmp, Pmax = metrics[hp]
    rows.append({
        "hp_nm": hp,
        "Jsc_mA_cm2": round(Jsc, 3),
        "Voc_V": round(Voc, 4),
        "FF_percent": round(FF * 100, 2),
        "Efficiency_percent": round(eta, 2),
        "Vmp_V": round(Vmp, 4),
        "Jmp_mA_cm2": round(Jmp, 3),
        "Pmax_mW_cm2": round(Pmax, 3),
    })

df = pd.DataFrame(rows)
print("\nMETRICS TABLE\n")
print(df.to_string(index=False))

fig, axs = plt.subplots(2, 2, figsize=(7.0, 4.9), dpi=500)
fig.patch.set_facecolor("white")

def style_axis(ax):
    ax.tick_params(direction="in", length=4, width=0.8, pad=4)
    for s in ax.spines.values():
        s.set_linewidth(0.8)

for ax in axs.flat:
    style_axis(ax)

# (a) J-V
ax = axs[0, 0]
handles_hp, labels_hp = [], []
for hp in hp_vals:
    c = cases[hp]["color"]
    h, = ax.plot(V, curves[hp], color=c, lw=1.5)
    ax.plot(V[::24], curves[hp][::24], ls="None", marker="o", ms=2.0,
            mfc="white", mec=c, mew=0.7, color=c)
    handles_hp.append(h)
    labels_hp.append(fr"$h_p={hp}\,\mathrm{{nm}}$")
ax.set_xlim(0, 1.0)
ax.set_ylim(40, 50.1)
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_yticks([40, 45, 50])
ax.set_xlabel("Voltage (V)")
ax.set_ylabel(r"Current density (mA/cm$^2$)", labelpad=4)
ax.text(0.03, 0.05, "(a)", transform=ax.transAxes, fontsize=10, fontweight="bold")

# (b) Jsc / Efficiency
ax = axs[0, 1]
ax2 = ax.twinx()
style_axis(ax2)
h_jsc, = ax.plot(hp_vals, Jsc_vals, color="#1f355e", lw=1.5, marker="o", ms=3.5,
                 mfc="white", mec="#1f355e", mew=0.8)
h_eta, = ax2.plot(hp_vals, eta_vals, color="#b06a4f", lw=1.5, marker="o", ms=3.5,
                  mfc="white", mec="#b06a4f", mew=0.8)
ax.set_xlim(9.8, 12.2)
ax.set_xticks([10, 11, 12])
ax.set_xlabel(r"QD height $h_p$ (nm)")
ax.set_ylabel(r"$J_{sc}$ (mA/cm$^2$)", labelpad=4)
ax2.set_ylabel("Efficiency (%)", labelpad=6)
ax.set_yticks([48.75, 49.00, 49.25])
ax2.set_yticks([36.5, 37.0])
ax.text(0.03, 0.05, "(b)", transform=ax.transAxes, fontsize=10, fontweight="bold")

# (c) Power density
ax = axs[1, 0]
for hp in hp_vals:
    c = cases[hp]["color"]
    P = V * curves[hp]
    ax.plot(V, P, color=c, lw=1.5)
    ax.plot(V[::24], P[::24], ls="None", marker="o", ms=2.0,
            mfc="white", mec=c, mew=0.7, color=c)
ax.set_xlim(0, 1.0)
ax.set_xticks([0.0, 0.5, 1.0])
ax.set_ylim(0, 36.5)
ax.set_yticks([0, 20, 35])
ax.set_xlabel("Voltage (V)")
ax.set_ylabel(r"Power density (mW/cm$^2$)", labelpad=4)
ax.text(0.03, 0.05, "(c)", transform=ax.transAxes, fontsize=10, fontweight="bold")

# (d) Voc / FF
ax = axs[1, 1]
ax2 = ax.twinx()
style_axis(ax2)
h_voc, = ax.plot(hp_vals, Voc_vals, color="#1f355e", lw=1.5, marker="o", ms=3.5,
                 mfc="white", mec="#1f355e", mew=0.8)
h_ff, = ax2.plot(hp_vals, FF_vals, color="#b06a4f", lw=1.5, marker="o", ms=3.5,
                 mfc="white", mec="#b06a4f", mew=0.8)
ax.set_xlim(9.8, 12.2)
ax.set_xticks([10, 11, 12])
ax.set_xlabel(r"QD height $h_p$ (nm)")
ax.set_ylabel(r"$V_{oc}$ (V)", labelpad=4)
ax2.set_ylabel("FF (%)", labelpad=6)
ax.set_yticks([0.880, 0.885])
ax2.set_yticks([84.75, 85.00])
ax.text(0.03, 0.05, "(d)", transform=ax.transAxes, fontsize=10, fontweight="bold")

# legends outside
fig.legend(handles_hp, labels_hp,
           loc="upper center", ncol=3, frameon=False,
           bbox_to_anchor=(0.5, 0.985), handlelength=2.8, columnspacing=1.6)

fig.legend([h_jsc, h_eta], [r"$J_{sc}$", "Efficiency"],
           loc="center left", frameon=False,
           bbox_to_anchor=(0.86, 0.73), handlelength=2.2)

fig.legend([h_voc, h_ff], [r"$V_{oc}$", "FF"],
           loc="center left", frameon=False,
           bbox_to_anchor=(0.86, 0.28), handlelength=2.2)

fig.subplots_adjust(left=0.12, right=0.82, bottom=0.12, top=0.80, wspace=0.40, hspace=0.40)

fig.savefig("modified_qd_ibsc_fixed.png", dpi=500, bbox_inches="tight")
plt.show()