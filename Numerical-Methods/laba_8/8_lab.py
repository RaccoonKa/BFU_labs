import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


os.makedirs("Results_8_lab", exist_ok = True)
A_bound, B_bound = 0.0, 1.0
T0, TMAX = 0.0, 10.0
D = 1.0


def phi(x):
    return x


def psi(x):
    return 2.0


def phi_second_deriv(x):
    return 0.0


def g1(t):
    return 0.0


def g2(t):
    return 1.0


def f_source(x, t):
    return 0.0


NX = 20
h = (B_bound - A_bound) / NX
tau = h / D
NT = int(np.ceil((TMAX - T0) / tau))
tau = (TMAX - T0) / NT

x_grid = np.linspace(A_bound, B_bound, NX + 1)
t_grid = np.linspace(T0, TMAX, NT + 1)
gamma2 = (D * tau / h) ** 2

U_expl = np.zeros((NT + 1, NX + 1))
U_impl1 = np.zeros((NT + 1, NX + 1))
U_impl2 = np.zeros((NT + 1, NX + 1))

for i in range(NX + 1):
    val = phi(x_grid[i])
    U_expl[0, i] = U_impl1[0, i] = U_impl2[0, i] = val

for i in range(1, NX):
    val_1 = U_expl[0, i] + tau * psi(x_grid[i]) + 0.5 * (tau ** 2) * D ** 2 * phi_second_deriv(x_grid[i])
    U_expl[1, i] = U_impl1[1, i] = U_impl2[1, i] = val_1

for mat in [U_expl, U_impl1, U_impl2]:
    mat[1, 0] = g1(t_grid[1])
    mat[1, NX] = g2(t_grid[1])

# явный метод
for n in range(1, NT):
    for i in range(1, NX):
        U_expl[n + 1, i] = (2 * U_expl[n, i] - U_expl[n - 1, i]
                            + gamma2 * (U_expl[n, i + 1] - 2 * U_expl[n, i] + U_expl[n, i - 1])
                            + (tau ** 2) * f_source(x_grid[i], t_grid[n]))
    U_expl[n + 1, 0] = g1(t_grid[n + 1])
    U_expl[n + 1, NX] = g2(t_grid[n + 1])


def solve_progonka(n, U_matrix, sigma):
    A_coeff = sigma * gamma2
    B_coeff = sigma * gamma2
    C_coeff = 1.0 + 2.0 * sigma * gamma2

    alpha = np.zeros(NX + 1)
    beta = np.zeros(NX + 1)

    alpha[1] = 0.0
    beta[1] = g1(t_grid[n + 1])

    for i in range(1, NX):
        if sigma == 1.0:
            F_i = 2 * U_matrix[n, i] - U_matrix[n - 1, i]
        else:
            L_n = gamma2 * (U_matrix[n, i + 1] - 2 * U_matrix[n, i] + U_matrix[n, i - 1])
            L_prev = gamma2 * (U_matrix[n - 1, i + 1] - 2 * U_matrix[n - 1, i] + U_matrix[n - 1, i - 1])
            F_i = 2 * U_matrix[n, i] - U_matrix[n - 1, i] + (1 - 2 * sigma) * L_n + sigma * L_prev

        F_i += (tau ** 2) * f_source(x_grid[i], t_grid[n + 1])

        denom = C_coeff - A_coeff * alpha[i]
        alpha[i + 1] = B_coeff / denom
        beta[i + 1] = (F_i + A_coeff * beta[i]) / denom

    U_matrix[n + 1, NX] = g2(t_grid[n + 1])
    for i in range(NX - 1, 0, -1):
        U_matrix[n + 1, i] = alpha[i + 1] * U_matrix[n + 1, i + 1] + beta[i + 1]
    U_matrix[n + 1, 0] = g1(t_grid[n + 1])


# неявный метод 1
for n in range(1, NT):
    solve_progonka(n, U_impl1, 1.0)

# неявный метод 2
for n in range(1, NT):
    solve_progonka(n, U_impl2, 0.25)


def plot_results(u, title):
    times_to_plot = [0, 2, 4, 6, 8, 10]
    indices = [int(np.argmin(np.abs(t_grid - t))) for t in times_to_plot]
    plt.figure(figsize = (10, 6))
    for t, idx in zip(times_to_plot, indices):
        plt.plot(x_grid, u[idx], label = f"t = {t}")
    plt.title(f"{title} (2D)")
    plt.xlabel("x")
    plt.ylabel("U(x,t)")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join("Results_8_lab", f"{title.replace(' ', '_')}_2d.png"))
    plt.close()
    X, T = np.meshgrid(x_grid, t_grid)
    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(111, projection = '3d')
    surf = ax.plot_surface(X, T, u, rstride = 1, cstride = 1, cmap = 'magma', edgecolor = 'none')
    fig.colorbar(surf, ax = ax, shrink = 0.5, aspect = 5)
    ax.set_title(f"{title} (3D)")
    plt.savefig(os.path.join("Results_8_lab", f"{title.replace(' ', '_')}_3d.png"))
    plt.close()


plot_results(U_expl, "Явный метод")
plot_results(U_impl1, "Неявный метод 1")
plot_results(U_impl2, "Неявный метод 2")

print("Графики сохранены в папку Results_8_lab")
