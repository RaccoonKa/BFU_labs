import numpy as np
import matplotlib.pyplot as plt


def ic(x):
    return np.maximum(0, (x - 5) * (10 - x))


def bc(t):
    return t * (t - 20) * (t - 50) / 20000.0


def v(x):
    return 11.0 - x


def analytical_solution(t, x):
    z = (11.0 - x) * np.exp(t)
    res = np.zeros_like(x)

    mask1 = z <= 11.0
    if np.any(mask1):
        res[mask1] = 6.5 * t + np.maximum(0, (6.0 - z[mask1]) * (z[mask1] - 1.0))

    mask2 = z > 11.0
    if np.any(mask2):
        ln_part = np.log((11.0 - x[mask2]) / 11.0)
        t_shift = t + ln_part
        res[mask2] = bc(t_shift) - 6.5 * ln_part

    return res


t_max = np.log(11.0)
L = 10.0
h_variants = [1.0, 0.5, 0.25, 0.1, 0.05]

for idx, h in enumerate(h_variants):
    tau = 0.9 * h / 11.0

    x = np.arange(0, L + h, h)
    t = np.arange(0, t_max + tau, tau)
    if t[-1] < t_max:
        t = np.append(t, t_max)

    Nx = len(x)
    Nt = len(t)

    V = v(x)
    f = 6.5

    U_exp = np.zeros(Nx)
    U_exp[:] = ic(x)

    U_imp = np.zeros(Nx)
    U_imp[:] = ic(x)

    for n in range(1, Nt):
        dt = t[n] - t[n - 1]

        U_exp_new = np.zeros(Nx)
        U_exp_new[0] = bc(t[n])
        for i in range(1, Nx):
            U_exp_new[i] = U_exp[i] - dt / h * V[i] * (U_exp[i] - U_exp[i - 1]) + dt * f
        U_exp = U_exp_new.copy()

        U_imp_new = np.zeros(Nx)
        U_imp_new[0] = bc(t[n])
        for i in range(1, Nx):
            U_imp_new[i] = (U_imp[i] + dt * f + dt / h * V[i] * U_imp_new[i - 1]) / (1.0 + dt / h * V[i])
        U_imp = U_imp_new.copy()

    U_exact = analytical_solution(t[-1], x)

    mse_exp = np.sqrt(np.mean((U_exp - U_exact) ** 2))
    mse_imp = np.sqrt(np.mean((U_imp - U_exact) ** 2))

    print(f"Вариант {idx + 1}: h = {h:.3f}, tau = {tau:.4f}")
    print(f"  СКО Явная схема:   {mse_exp:.6f}")
    print(f"  СКО Неявная схема: {mse_imp:.6f}\n")

    plt.figure(figsize=(10, 6))
    plt.plot(x, U_exact, 'k-', linewidth=2, label='Аналитическое')
    plt.plot(x, U_exp, 'r--', linewidth=2, label='Явная схема')
    plt.plot(x, U_imp, 'b:', linewidth=2, label='Неявная схема')
    plt.title(f'Сравнение решений при t = {t_max:.3f} (Вариант {idx + 1}, Шаг h = {h})')
    plt.xlabel('x')
    plt.ylabel('U(t, x)')
    plt.grid(True)
    plt.legend()
    plt.show()