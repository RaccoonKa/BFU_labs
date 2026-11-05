import numpy as np
import matplotlib.pyplot as plt


def check_area(x, y):
    return (y <= 4 - x ** 2) & (y >= x ** 2 - 4) & (x <= 4 - y ** 2 / 2) & (x >= y ** 2 / 2 - 4) & (y >= x ** 2 / 2 - 2)


def f(x, y):
    return x + y ** 2


x_min, x_max, y_min, y_max = -4, 4, -4, 4
S_R = (x_max - x_min) * (y_max - y_min)
N_list = [100, 1000, 5000, 10000, 50000, 100000]

errors = []
areas = []
integrals = []

fig_domains, axes = plt.subplots(2, 3, figsize=(15, 9))
axes = axes.flatten()

for idx, N in enumerate(N_list):
    x_pts = np.random.uniform(x_min, x_max, N)
    y_pts = np.random.uniform(y_min, y_max, N)

    in_domain = check_area(x_pts, y_pts)
    N_in = np.sum(in_domain)

    S_D = S_R * N_in / N
    areas.append(S_D)

    g_vals = np.where(in_domain, f(x_pts, y_pts), 0)
    I_val = S_R * np.mean(g_vals)
    integrals.append(I_val)

    variance = np.var(g_vals, ddof=1) if N > 1 else 0
    error = S_R * np.sqrt(variance / N)
    errors.append(error)

    ax = axes[idx]
    ax.scatter(x_pts[~in_domain], y_pts[~in_domain], c='lightgray', s=1, alpha=0.5)
    ax.scatter(x_pts[in_domain], y_pts[in_domain], c='crimson', s=1)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"N = {N}\nS = {S_D:.2f}, I = {I_val:.2f}")
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()

fig_err, ax_err = plt.subplots(figsize=(8, 5))
ax_err.plot(N_list, errors, marker='o', color='indigo', lw=2)
ax_err.set_xscale('log')
ax_err.set_yscale('log')
ax_err.set_title("Убывание статистической погрешности")
ax_err.set_xlabel("Количество точек N")
ax_err.set_ylabel("Погрешность")
ax_err.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()