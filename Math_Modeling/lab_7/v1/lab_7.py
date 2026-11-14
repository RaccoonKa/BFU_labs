import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


def lotka_volterra(N, t, r1, r2, K1, K2, a12, a21):
    N1, N2 = N
    dN1_dt = r1 * N1 * (K1 - N1 - a12 * N2) / K1
    dN2_dt = r2 * N2 * (K2 - N2 - a21 * N1) / K2
    return [dN1_dt, dN2_dt]


r1, r2 = 1.0, 1.0
K1, K2 = 100.0, 100.0

scenarios = [
    {'a12': 0.5, 'a21': 0.5, 'title': '1. Устойчивое сосуществование (a12<1, a21<1)'},
    {'a12': 0.5, 'a21': 1.5, 'title': '2. Вытеснение второго вида первым (a12<1, a21>1)'},
    {'a12': 1.5, 'a21': 0.5, 'title': '3. Вытеснение первого вида вторым (a12>1, a21<1)'},
    {'a12': 1.5, 'a21': 1.5, 'title': '4. Неустойчивое сосуществование (a12>1, a21>1)'}
]

t = np.linspace(0, 50, 500)
initial_conditions = [[10, 10], [10, 80], [80, 10], [90, 90]]

N1_grid, N2_grid = np.meshgrid(np.linspace(0, 120, 20), np.linspace(0, 120, 20))

for scenario in scenarios:
    a12, a21 = scenario['a12'], scenario['a21']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(scenario['title'], fontsize=16, fontweight='bold')

    for ic in initial_conditions:
        solution = odeint(lotka_volterra, ic, t, args=(r1, r2, K1, K2, a12, a21))
        ax1.plot(t, solution[:, 0], 'b-', alpha=0.7)
        ax1.plot(t, solution[:, 1], 'r-', alpha=0.7)

    ax1.plot([], [], 'b-', label='Вид 1 (N1)')
    ax1.plot([], [], 'r-', label='Вид 2 (N2)')
    ax1.set_xlabel('Время (t)')
    ax1.set_ylabel('Численность')
    ax1.set_title('Зависимость численности от времени')
    ax1.grid(True, linestyle='--', alpha=0.6)
    ax1.legend()

    u, v = np.zeros(N1_grid.shape), np.zeros(N2_grid.shape)
    for i in range(N1_grid.shape[0]):
        for j in range(N1_grid.shape[1]):
            rates = lotka_volterra([N1_grid[i, j], N2_grid[i, j]], 0, r1, r2, K1, K2, a12, a21)
            u[i, j] = rates[0]
            v[i, j] = rates[1]

    norm = np.hypot(u, v)
    norm[norm == 0] = 1
    u = u / norm
    v = v / norm

    ax2.quiver(N1_grid, N2_grid, u, v, color='gray', alpha=0.5, scale=30)

    n1_iso = np.linspace(0, 120, 100)
    n2_iso1 = (K1 - n1_iso) / a12
    n2_iso2 = K2 - a21 * n1_iso

    ax2.plot(n1_iso, n2_iso1, 'b--', linewidth=2, label='Изоклина dN1/dt = 0')
    ax2.plot(n1_iso, n2_iso2, 'r--', linewidth=2, label='Изоклина dN2/dt = 0')

    for ic in initial_conditions:
        solution = odeint(lotka_volterra, ic, t, args=(r1, r2, K1, K2, a12, a21))
        ax2.plot(solution[:, 0], solution[:, 1], 'k-', linewidth=1.5)
        ax2.plot(solution[0, 0], solution[0, 1], 'go', markersize=6)

    singular_points = [(0, 0), (K1, 0), (0, K2)]
    det = 1 - a12 * a21
    if det != 0:
        n1_star = (K1 - a12 * K2) / det
        n2_star = (K2 - a21 * K1) / det
        if n1_star > 0 and n2_star > 0:
            singular_points.append((n1_star, n2_star))

    for sp in singular_points:
        ax2.plot(sp[0], sp[1], 'ro', markersize=8, markeredgecolor='black')

    ax2.set_xlim(0, 120)
    ax2.set_ylim(0, 120)
    ax2.set_xlabel('Численность Вида 1 (N1)')
    ax2.set_ylabel('Численность Вида 2 (N2)')
    ax2.set_title('Фазовый портрет')
    ax2.grid(True, linestyle='--', alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plt.show()