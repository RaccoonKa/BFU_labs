import numpy as np
import pandas as pd
from pathlib import Path

EPS = 0.001
a, b = 0.0, 0.5

WORKDIR = Path.cwd()
RESULTS_DIR = WORKDIR / "Results_5_lab"
RESULTS_DIR.mkdir(exist_ok = True)


def f1(x, y):
    return np.cos(x**2 - y**2) + 0.2 * y


def f2_system(x, Y):
    y1, y2 = Y
    dy1 = y2
    dy2 = 1 - np.sin(0.75 * x + y1**2)
    return np.array([dy1, dy2])


def euler_cauchy(f, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)

    for i in range(n):
        k1 = f(x[i], y[i])
        y_pred = y[i] + h * k1
        k2 = f(x[i] + h, y_pred)
        y[i + 1] = y[i] + h * (k1 + k2) / 2

    return x, y


def runge_kutta4(f, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = np.zeros(n + 1)

    for i in range(n):
        k1 = h * f(x[i], y[i])
        k2 = h * f(x[i] + h/2, y[i] + k1/2)
        k3 = h * f(x[i] + h/2, y[i] + k2/2)
        k4 = h * f(x[i] + h, y[i] + k3)

        y[i + 1] = y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, y


def rk4_system(f, n):
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    Y = np.zeros((n + 1, 2))
    Y[0] = [0, 1]

    for i in range(n):
        k1 = h * f(x[i], Y[i])
        k2 = h * f(x[i] + h/2, Y[i] + k1/2)
        k3 = h * f(x[i] + h/2, Y[i] + k2/2)
        k4 = h * f(x[i] + h, Y[i] + k3)

        Y[i + 1] = Y[i] + (k1 + 2*k2 + 2*k3 + k4) / 6

    return x, Y


def adams3(f, n):
    h = (b - a) / n
    x, Y = rk4_system(f, n)

    for i in range(2, n):
        f_i  = f(x[i], Y[i])
        f_i1 = f(x[i-1], Y[i-1])
        f_i2 = f(x[i-2], Y[i-2])

        Y[i+1] = Y[i] + h * (23*f_i - 16*f_i1 + 5*f_i2) / 12

    return x, Y[:, 0]


def adams4(f, n):
    h = (b - a) / n
    x, Y = rk4_system(f, n)

    for i in range(3, n):
        f_i  = f(x[i], Y[i])
        f_i1 = f(x[i-1], Y[i-1])
        f_i2 = f(x[i-2], Y[i-2])
        f_i3 = f(x[i-3], Y[i-3])

        Y[i+1] = Y[i] + h * (55*f_i - 59*f_i1 + 37*f_i2 - 9*f_i3) / 24

    return x, Y[:, 0]


def double_recalculation(method, f):
    n = 2

    while True:
        x_prev, y_prev = method(f, n-1)
        x_last, y_last = method(f, 2 * n-1)
        y_last_match = y_last[::2]
        diff = np.max(np.abs(y_prev - y_last_match))

        if diff < EPS:
            print(diff)
            print(f"n:{2*n}")
            return n, x_prev, y_prev, 2*n, x_last, y_last

        n = 2*n


def save_table(method_name, n_prev, x_prev, y_prev, n_last, x_last, y_last):
    x_match = x_last[::2]
    y_last_match = y_last[::2]

    x_prev_sel = x_match[-8:]
    y_prev_sel = y_prev[-8:]
    y_last_sel = y_last_match[-8:]

    diff = np.abs(y_last_sel - y_prev_sel)

    df_prev = pd.DataFrame({
        "x_k": x_prev_sel,
        "y_prev": y_prev_sel,
        "y_last": y_last_sel,
        "difference": diff
    })
    df_last = pd.DataFrame({
        "x_k": x_last[-16:],
        "y_last": y_last[-16:]
    })

    df_prev = df_prev.round(4)
    df_last = df_last.round(4)

    df_prev.to_csv(RESULTS_DIR / f"{method_name}_prev.csv", index = False)
    df_last.to_csv(RESULTS_DIR / f"{method_name}_last.csv", index = False)

    print("\n")
    print("Метод:", method_name)
    print("Число точек разбиения последней итерации:", n_last)

n_prev, x_prev, y_prev, n_last, x_last, y_last = double_recalculation(euler_cauchy, f1)
save_table("Euler_Cauchy_eq1", n_prev, x_prev, y_prev, n_last, x_last, y_last)

n_prev, x_prev, y_prev, n_last, x_last, y_last = double_recalculation(runge_kutta4, f1)
save_table("Runge_Kutta4_eq1", n_prev, x_prev, y_prev, n_last, x_last, y_last)

n_prev, x_prev, y_prev, n_last, x_last, y_last = double_recalculation(adams3, f2_system)
save_table("Adams3_eq2", n_prev, x_prev, y_prev, n_last, x_last, y_last)

n_prev, x_prev, y_prev, n_last, x_last, y_last = double_recalculation(adams4, f2_system)
save_table("Adams4_eq2", n_prev, x_prev, y_prev, n_last, x_last, y_last)
