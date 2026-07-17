import numpy as np
import os
import matplotlib.pyplot as plt

output_dir = 'Results_10_lab'
os.makedirs(output_dir, exist_ok = True)
a, b = 0, 10
c, d = 0, 10
eps = 0.01


def f(x, y):
    return 2 * x * y


def initialize_grid(N):
    x = np.linspace(a, b, N + 1)
    y = np.linspace(c, d, N + 1)
    U = np.zeros((N + 1, N + 1))
    for i in range(N + 1):
        U[i, 0] = x[i] + c
        U[i, -1] = x[i] + d
    for j in range(N + 1):
        U[0, j] = a + y[j]
        U[-1, j] = b + y[j]

    return x, y, U


def solve_elliptic(N, method='simple'):
    x, y, U = initialize_grid(N)
    h = (b - a) / N
    F = np.zeros((N + 1, N + 1))
    for i in range(1, N):
        for j in range(1, N):
            F[i, j] = f(x[i], y[j]) * h ** 2

    iterations = 0
    while True:
        max_diff = 0

        if method == 'simple':
            U_new = U.copy()
            for i in range(1, N):
                for j in range(1, N):
                    U_new[i, j] = 0.25 * (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1] - F[i, j])

            max_diff = np.max(np.abs(U_new - U))
            U = U_new.copy()

        elif method == 'seidel':
            for i in range(1, N):
                for j in range(1, N):
                    old_val = U[i, j]
                    U[i, j] = 0.25 * (U[i + 1, j] + U[i - 1, j] + U[i, j + 1] + U[i, j - 1] - F[i, j])
                    max_diff = max(max_diff, abs(U[i, j] - old_val))

        iterations += 1

        if max_diff < eps:
            break

    return x, y, U, iterations


def run_and_save():
    grids = [5, 10]
    methods = ['simple', 'seidel']

    report_path = os.path.join(output_dir, 'report.txt')
    with open(report_path, 'w', encoding = 'utf-8') as f_out:
        f_out.write("Результаты\n")

        for N in grids:
            f_out.write(f"--- Сетка {N}x{N} ячеек ---\n")
            fig3d = plt.figure(figsize = (12, 5))
            fig2d = plt.figure(figsize = (12, 5))

            for idx, method in enumerate(methods):
                x, y, U, iters = solve_elliptic(N, method = method)
                method_name = "Простая итерация" if method == 'simple' else "Метод Зейделя"
                f_out.write(f"Метод: {method_name}\n")
                f_out.write(f"Количество итераций до достижения погрешности {eps}: {iters}\n")
                f_out.write("Центральное сечение матрицы U (строка N/2):\n")
                f_out.write(str(np.round(U[:, N // 2], 3)) + "\n\n")
                X, Y = np.meshgrid(x, y)

                ax3d = fig3d.add_subplot(1, 2, idx + 1, projection = '3d')
                surf = ax3d.plot_surface(X, Y, U.T, cmap = 'viridis', edgecolor = 'none')
                ax3d.set_title(f"{method_name} 3D ({N}x{N})")
                ax3d.set_xlabel('X')
                ax3d.set_ylabel('Y')
                ax3d.set_zlabel('U(x,y)')

                ax2d = fig2d.add_subplot(1, 2, idx + 1)
                contour = ax2d.contourf(X, Y, U.T, levels = 20, cmap = 'viridis')
                fig2d.colorbar(contour, ax = ax2d)
                ax2d.set_title(f"{method_name} 2D ({N}x{N})")
                ax2d.set_xlabel('X')
                ax2d.set_ylabel('Y')

            fig3d.tight_layout()
            plot_path_3d = os.path.join(output_dir, f'plot_3d_{N}x{N}.png')
            fig3d.savefig(plot_path_3d)
            plt.close(fig3d)

            fig2d.tight_layout()
            plot_path_2d = os.path.join(output_dir, f'plot_2d_{N}x{N}.png')
            fig2d.savefig(plot_path_2d)
            plt.close(fig2d)

            f_out.write("\n")

    print(f"Все результаты и графики сохранены в папку: {output_dir}")


if __name__ == '__main__':
    run_and_save()
