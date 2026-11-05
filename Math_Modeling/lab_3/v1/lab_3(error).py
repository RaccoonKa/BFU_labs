import numpy as np
import matplotlib.pyplot as plt


def check_area(x, y):
    return (y <= 4 - x**2) & (y >= x**2 - 4)


def f(x, y):
    return x + y ** 2


exact_area = 64 / 3
exact_integral = 8192 / 105

x_min, x_max, y_min, y_max = -4, 4, -4, 4
S_R = (x_max - x_min) * (y_max - y_min)
N_list = [100, 1000, 5000, 10000, 50000, 100000, 500000, 1000000]

rel_errors_area = []
rel_errors_integral = []

print(f"{'N':<10} | {'S (MC)':<10} | {'S (Ошибка %)':<12} | {'I (MC)':<10} | {'I (Ошибка %)'}")
print("-" * 65)

for N in N_list:
    x_pts = np.random.uniform(x_min, x_max, N)
    y_pts = np.random.uniform(y_min, y_max, N)

    in_domain = check_area(x_pts, y_pts)
    N_in = np.sum(in_domain)

    S_MC = S_R * N_in / N

    g_vals = np.where(in_domain, f(x_pts, y_pts), 0)
    I_MC = S_R * np.mean(g_vals)

    rel_err_S = (abs(exact_area - S_MC) / exact_area) * 100
    rel_err_I = (abs(exact_integral - I_MC) / exact_integral) * 100

    rel_errors_area.append(rel_err_S)
    rel_errors_integral.append(rel_err_I)

    print(f"{N:<10} | {S_MC:<10.4f} | {rel_err_S:<12.4f} | {I_MC:<10.4f} | {rel_err_I:.4f}")

fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(N_list, rel_errors_area, marker='o', color='teal', lw=2, label='Погрешность площади S')
ax.plot(N_list, rel_errors_integral, marker='s', color='darkorange', lw=2, label='Погрешность интеграла I')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_title("Падение относительной (истинной) погрешности")
ax.set_xlabel("Количество точек N")
ax.set_ylabel("Относительная погрешность (%)")
ax.grid(True, linestyle='--', alpha=0.6)
ax.legend()

plt.tight_layout()
plt.show()