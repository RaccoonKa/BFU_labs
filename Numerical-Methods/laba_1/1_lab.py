import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WORKDIR = Path(__file__).parent
RESULT_DIR = WORKDIR / "Results_1_lab"
RESULT_DIR.mkdir(parents = True, exist_ok = True)
pd.options.display.float_format = '{:.8f}'.format
FUNCTION = 'exp(1-x) + x^2 - 5'


def f(x):
    return np.exp(1 - x) + x ** 2 - 5


def df(x):
    return -np.exp(1 - x) + 2 * x


def ddf(x):
    return np.exp(1 - x) + 2


def create_beautiful_axes(ax, x_min, x_max, y_min, y_max):
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    for spine in ['top', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['left', 'bottom']:
        ax.spines[spine].set_linewidth(1.5)
        ax.spines[spine].set_color('black')

    if x_min <= 0 <= x_max:
        ax.spines['bottom'].set_position('zero')

    if y_min <= 0 <= y_max:
        ax.spines['left'].set_position('zero')

    ax.minorticks_on()
    ax.grid(True, which = 'major', linestyle = '--', linewidth = 0.5, alpha = 0.7)
    ax.grid(True, which = 'minor', linestyle = ':', linewidth = 0.3, alpha = 0.4)

    ax.set_xlabel('x', fontsize=12)
    ax.set_ylabel('f(x)', fontsize=12)

    return ax


def logger(file_name, data, interval):
    pd.DataFrame({'x': data}).to_csv(
        RESULT_DIR / f"{file_name}.csv",
        sep = '\t',
        index = False
    )

    return pd.DataFrame({
        'Метод решения': [file_name],
        'Интервал': [interval],
        'Корень': [data[-1]],
        'Итераций': [len(data)]
    })


a0, b0 = -2, 3
step = 0.25
n = 21
eps = 1e-7
h = 1e-1
x_tab = np.linspace(a0, b0, n)
y_tab = f(x_tab)
print("Таблица значений:")
print(pd.DataFrame({'x': x_tab, 'f(x)': y_tab}))

intervals = []
for i in range(len(x_tab) - 1):
    if f(x_tab[i]) * f(x_tab[i + 1]) <= 0:
        intervals.append((x_tab[i], x_tab[i + 1]))
print("\nИнтервалы с корнями:", intervals)

x_plot = np.linspace(a0, b0, 500)
y_plot = f(x_plot)
fig, ax = plt.subplots(figsize = (12, 8))
ax.plot(x_plot, y_plot,
        linewidth = 2.5,
        label = 'f(x) = e^(1-x) + x^2 - 5',
        alpha = 0.85)
y_min, y_max = y_plot.min(), y_plot.max()
x_margin = 0.05 * (b0 - a0)
y_margin = 0.1 * (y_max - y_min)
ax = create_beautiful_axes(
    ax,
    a0 - x_margin,
    b0 + x_margin,
    y_min - y_margin,
    y_max + y_margin
)
ax.set_title(
    f'f(x) = exp(1-x) + x^2 - 5\n[{a0}, {b0}], n={n}, step={step}',
    fontsize = 15,
    pad = 15
)

ax.axhline(0, linestyle = '--', alpha = 0.6)
ax.legend()
plt.tight_layout()
plt.savefig(RESULT_DIR / "function_plot.png", dpi = 300, bbox_inches = 'tight')
plt.close()

full_result = pd.DataFrame()
for idx, (a, b) in enumerate(intervals, start=1):
    print(f"\nОбработка корня {idx} на интервале [{a:.4f}, {b:.4f}]")

    # ньютон
    x = a if f(a) * ddf(a) > 0 else b
    result = [f'{x:.8f}']
    while True:
        x_new = x - f(x) / df(x)
        result.append(f'{x_new:.8f}')
        if abs(x_new - x) < eps:
            break
        x = x_new
    full_result = pd.concat([full_result,
                             logger(f'Newton_root{idx}', result, (a, b))])
    print(f"Ньютон: корень = {float(result[-1]):.8f}, итераций = {len(result)}")

    # метод хорд
    x = b
    result = [f'{x:.8f}']
    while True:
        x_new = x - f(x) * (x - a) / (f(x) - f(a))
        result.append(f'{x_new:.8f}')
        if abs(x_new - x) < eps:
            break
        x = x_new
    full_result = pd.concat([full_result,
                             logger(f'Chords_root{idx}', result, (a, b))])
    print(f"Хорды: корень = {float(result[-1]):.8f}, итераций = {len(result)}")

    # метод секущих
    x0, x1 = a, b
    result = [f'{x0:.8f}']
    while True:
        x2 = x1 - f(x1) * (x1 - x0) / (f(x1) - f(x0))
        result.append(f'{x2:.8f}')
        if abs(x2 - x1) < eps:
            break
        x0, x1 = x1, x2
    full_result = pd.concat([full_result,
                             logger(f'Secants_root{idx}', result, (a, b))])
    print(f"Секущие: корень = {float(result[-1]):.8f}, итераций = {len(result)}")

    # конечно-разностный
    x = a
    result = [f'{x:.8f}']
    while True:
        x_new = x - h * f(x) / (f(x + h) - f(x))
        result.append(f'{x_new:.8f}')
        if abs(x_new - x) < eps:
            break
        x = x_new
    full_result = pd.concat([full_result,
                             logger(f'FiniteNewton_root{idx}', result, (a, b))])
    print(f"Конечно-разностный Ньютон: корень = {float(result[-1]):.8f}, итераций = {len(result)}")

    # стеффенсен
    x = (a + b) / 2
    result = [f'{x:.8f}']
    while True:
        x_new = x - f(x) ** 2 / (f(x + f(x)) - f(x))
        result.append(f'{x_new:.8f}')
        if abs(x_new - x) < eps:
            break
        x = x_new
    full_result = pd.concat([full_result,
                             logger(f'Steffensen_root{idx}', result, (a, b))])
    print(f"Стеффенсен: корень = {float(result[-1]):.8f}, итераций = {len(result)}")

    # релакс
    x_vals = np.linspace(a, b, 100)
    df_vals = df(x_vals)
    m = min(abs(df_vals))
    M = max(abs(df_vals))

    if m > 0:
        if df((a + b) / 2) > 0:
            tau = 2 / (M + m)
        else:
            tau = -2 / (M + m)
    else:
        tau = 0.1 * np.sign(df((a + b) / 2))

    x = (a + b) / 2
    result = [f'{x:.8f}']
    max_iter = 10000
    iter_count = 0

    while iter_count < max_iter:
        x_new = x - tau * f(x)
        result.append(f'{x_new:.8f}')
        if abs(x_new - x) < eps:
            break

        if x_new < a or x_new > b:
            x = (a + b) / 2
            tau = tau * 0.5
            iter_count += 1
            continue

        x = x_new
        iter_count += 1

    full_result = pd.concat([full_result,
                             logger(f'Relax_root{idx}', result, (a, b))])
    print(f"Релаксация: корень = {float(result[-1]):.8f}, итераций = {len(result)}, tau = {tau:.6f}")

full_result.to_csv(RESULT_DIR / "Results.csv", index = False)

print("\n" + "=" * 60)
print("Итог:")
print("=" * 60)
print(full_result.to_string())
print("\nФайлы сохранены в:", RESULT_DIR)
