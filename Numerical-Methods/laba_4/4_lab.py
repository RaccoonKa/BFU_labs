import numpy as np
from pathlib import Path
from math import sin, cos, factorial
from typing import Callable
import pandas as pd


WORKDIR = Path(__file__).parent
RESULTS_DIR = WORKDIR / "Results_4_lab"
RESULTS_DIR.mkdir(exist_ok = True)


def interpolate(y: Callable, y4: Callable, x: float, xk: list[float]):

    yk = [round(y(xi), 4) for xi in xk]
    n = len(xk)
    li_polynomials = []
    L_poly = np.poly1d([0])

    for i in range(n):

        numerator = np.poly1d([1])
        denominator = 1

        for j in range(n):
            if i != j:
                numerator *= np.poly1d([1, -xk[j]])  # (x - xj)
                denominator *= (xk[i] - xk[j])

        li = numerator / denominator
        li_polynomials.append(li)
        L_poly += yk[i] * li

    y_interp = L_poly(x)
    y_exact = y(x)
    abs_error = abs(y_interp - y_exact)
    xs = np.linspace(xk[0], xk[-1], 1000)
    M = max(abs(y4(xi)) for xi in xs)

    omega = 1
    for xi in xk:
        omega *= (x - xi)

    R = (M / factorial(n)) * abs(omega)

    with open(RESULTS_DIR / "Interpolation.txt", "w", encoding = "utf-8") as f:

        f.write("Интерполяция\n\n")
        f.write("Узлы:\n")
        for i in range(n):
            f.write(f"x{i} = {xk[i]:.2f}, y{i} = {yk[i]:.4f}\n")
        f.write("\nБазисные многочлены l_i(x):\n\n")
        for i, li in enumerate(li_polynomials):
            f.write(f"l_{i}(x) = {li}\n\n")
        f.write("\nИтоговый многочлен Лагранжа:\n")
        f.write(f"L_{n-1}(x) = {L_poly}\n\n")
        f.write(f"L({x}) = {y_interp:.7f}\n")
        f.write(f"Exact value = {y_exact:.7f}\n")
        f.write(f"Absolute error = {abs_error:.7f}\n")
        f.write(f"Theoretical estimate = {R:.7e}\n")

    return L_poly


def differentiate(f, f1, f2, a, b, n):
    h = (b - a) / (n - 1)
    x_vals = [a + i*h for i in range(n)]
    y_vals = [f(xi) for xi in x_vals]
    left, right, central, second = [], [], [], []

    for i in range(n):
        if i == 0:
            left.append(None)
            right.append((y_vals[i+1] - y_vals[i]) / h)
            central.append(None)
            second.append(None)
        elif i == n - 1:
            left.append((y_vals[i] - y_vals[i-1]) / h)
            right.append(None)
            central.append(None)
            second.append(None)
        else:
            left.append((y_vals[i] - y_vals[i-1]) / h)
            right.append((y_vals[i+1] - y_vals[i]) / h)
            central.append((y_vals[i+1] - y_vals[i-1]) / (2*h))
            second.append((y_vals[i-1] - 2*y_vals[i] + y_vals[i+1]) / h**2)

    df = pd.DataFrame({
        "x": np.round(x_vals, 4),
        "Left": left,
        "Right": right,
        "Central": central,
        "Second": second,
        "Exact f'(x)": [f1(xi) for xi in x_vals],
        "Exact f''(x)": [f2(xi) for xi in x_vals]
    })

    df.to_csv(
        RESULTS_DIR / "Differentiation.csv",
        index=False,
        float_format="%.4f"
    )

    errors_data = []
    for i in range(n):
        row = {
            "x_k, m": f"{x_vals[i]:.3f}, {i}",
            "f'(x) left": abs(left[i] - f1(x_vals[i])) if left[i] is not None else None,
            "f'(x) right": abs(right[i] - f1(x_vals[i])) if right[i] is not None else None,
            "f'(x) mid": abs(central[i] - f1(x_vals[i])) if central[i] is not None else None,
            "f''(x) numeric": abs(second[i] - f2(x_vals[i])) if second[i] is not None else None,
        }
        errors_data.append(row)
    errors_df = pd.DataFrame(errors_data)


    def format_float(x):
        if pd.isna(x):
            return "-"
        return f"{x:.5f}".replace('.', ',')
    for col in errors_df.columns[1:]:
        errors_df[col] = errors_df[col].apply(format_float)
    errors_df.to_csv(
        RESULTS_DIR / "Errors_Differentiation.csv",
        sep=';',
        index=False,
        encoding='utf-8'
    )

    return df


if __name__ == "__main__":
    y = lambda x: sin(x)
    y4 = lambda x: sin(x)
    x_point = 1.04
    xk = [1.00, 1.05, 1.10, 1.15]
    interpolate(y, y4, x_point, xk)
    f = lambda x: sin(x)
    f1 = lambda x: cos(x)
    f2 = lambda x: -sin(x)
    differentiate(f, f1, f2, 0, 1, 5)
    print("Вычисления завершены. Результаты сохранены в папке Results_4_lab.")