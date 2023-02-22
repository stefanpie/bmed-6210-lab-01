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

fig, ax = plt.subplots(1, 1, figsize=(10, 10))
for material, eq in equations.items():
    print(material, eq)
    eq_plot = sp.lambdify("t", eq, "numpy")
    t = np.linspace(0, 3000, 100)
    Mz = eq_plot(t)
    ax.plot(t, Mz, label=material)
    # solve for expression = 0
    t_eq = sp.solve(eq, "t")[0]
    print("t_eq", t_eq.evalf())

ax.set_xlabel("t")
ax.set_ylabel("Mz")
ax.legend()
plt.show()