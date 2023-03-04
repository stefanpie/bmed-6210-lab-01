import sympy as sp
import numpy as np
import matplotlib.pyplot as plt


T1_values={
    "fat": 250-3,
    "white_matter": 680-3,
    "gray_matter": 810-3,
    "csf": 2500-3,
}

equations = {}

for material, T1_val in T1_values.items():
    M0 = sp.Symbol("M0")
    T1 = sp.Symbol("T1")
    Mz = sp.Symbol("Mz")
    t = sp.Symbol("t")
    right = -M0*sp.exp(-t/T1) + M0*(1-sp.exp(-t/T1))
    eq = right.subs(T1, T1_val).subs(M0, 1)
    equations[material] = eq

fig, ax = plt.subplots(1, 2, figsize=(10, 6))
for material, eq in equations.items():
    print(f"{material}: {eq}")
    eq_plot = sp.lambdify("t", eq, "numpy")
    t = np.linspace(0, 4000, 1000)
    Mz = eq_plot(t)
    line = ax[0].plot(t, Mz, label=material)
    color = line[0].get_color()
    ax[1].plot(t, np.abs(Mz), color=color, ls="--", label=f"{material} (abs)")
    t_eq = sp.solve(eq, "t")[0]
    print(f"t_eq: {t_eq.evalf()}")

ax[0].set_title("Magnetization vs. Time after Inversion Pulse")
ax[0].set_xlim(0, 4000)
ax[0].set_ylim(-1, 1)
ax[0].set_xlabel("t")
ax[0].set_ylabel("Mz")

ax[1].set_title("abs(Magnetization) vs. Time after Inversion Pulse")
ax[1].set_xlim(0, 4000)
ax[1].set_ylim(0, 1)
ax[1].set_xlabel("t")
ax[1].set_ylabel("abs(Mz)")

ax[0].legend()
ax[1].legend()

plt.tight_layout()
plt.savefig("pre_lab_stuff.png", dpi=300)