import numpy as np
import matplotlib.pyplot as plt


def d(x): return 12.0 - x


def f(x): return 2.0 * x


def ic(x): return x * (10.0 - x) ** 2


t_max = 100.0 / np.sqrt(12.0 * 2.0)
L = 10.0
h_vars = [1.0, 0.5, 0.25, 0.125, 0.0625]

results_exp = []
results_imp = []

for idx, h in enumerate(h_vars):
    tau = 0.8 * h ** 2 / 24.0

    Nx = int(L / h) + 1
    x = np.linspace(0, L, Nx)

    Nt = int(np.ceil(t_max / tau)) + 1
    t = np.linspace(0, t_max, Nt)

    U_exp = ic(x)
    U_imp = ic(x)

    D = d(x)
    F_arr = f(x)

    for n in range(1, Nt):
        dt = t[n] - t[n - 1]

        U_new = np.zeros(Nx)
        for i in range(1, Nx - 1):
            U_new[i] = (1.0 - 2.0 * dt * D[i] / h ** 2 - dt * F_arr[i]) * U_exp[i] + \
                       (dt * D[i] / h ** 2) * U_exp[i + 1] + \
                       (dt * D[i] / h ** 2) * U_exp[i - 1] + 5.0 * dt
        U_exp = U_new.copy()

        A = -dt * D / h ** 2
        B = 1.0 + 2.0 * dt * D / h ** 2 + dt * F_arr
        C = -dt * D / h ** 2

        alpha = np.zeros(Nx)
        beta = np.zeros(Nx)

        for i in range(1, Nx - 1):
            rhs = U_imp[i] + 5.0 * dt
            denom = B[i] + A[i] * alpha[i - 1]
            alpha[i] = -C[i] / denom
            beta[i] = (rhs - A[i] * beta[i - 1]) / denom

        U_new_imp = np.zeros(Nx)
        for i in range(Nx - 2, 0, -1):
            U_new_imp[i] = alpha[i] * U_new_imp[i + 1] + beta[i]
        U_imp = U_new_imp.copy()

    step_size = int(h_vars[0] / h)
    results_exp.append(U_exp[::step_size])
    results_imp.append(U_imp[::step_size])

    plt.figure(figsize=(10, 6))
    plt.plot(x, U_exp, 'r-', linewidth=2, label='Явная схема')
    plt.plot(x, U_imp, 'b--', linewidth=2, label='Неявная схема')
    plt.title(f'Диффузия при t = {t_max:.3f} (Шаг h = {h})')
    plt.xlabel('x')
    plt.ylabel('U(t, x)')
    plt.grid(True)
    plt.legend()
    plt.show()

print(f"Проверка порядка аппроксимации методом Рунге (сгущения сеток):")
for i in range(len(h_vars) - 2):
    err1_exp = np.max(np.abs(results_exp[i] - results_exp[i + 1]))
    err2_exp = np.max(np.abs(results_exp[i + 1] - results_exp[i + 2]))
    order_exp = np.log2(err1_exp / err2_exp)

    err1_imp = np.max(np.abs(results_imp[i] - results_imp[i + 1]))
    err2_imp = np.max(np.abs(results_imp[i + 1] - results_imp[i + 2]))
    order_imp = np.log2(err1_imp / err2_imp)

    print(f"\nПереход h = {h_vars[i]} -> {h_vars[i + 1]} -> {h_vars[i + 2]}:")
    print(f"  Экспериментальный порядок (Явная):   {order_exp:.2f}")
    print(f"  Экспериментальный порядок (Неявная): {order_imp:.2f}")