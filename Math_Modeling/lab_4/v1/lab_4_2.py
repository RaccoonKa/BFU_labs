"""
Суть алгоритма (Трехлистный аттрактор через IFS и его размерность):
1. Заданы 3 матрицы сжатия/поворота A и 3 вектора сдвига B.
2. Начиная с (0, 0), на каждом шаге равновероятно выбираем
   одно из 3 преобразований: X_(n+1) = A_i * X_n + B_i.
3. Первые 100 точек отбрасываем как переходный процесс.
4. Находим bounding box аттрактора.
5. Box-counting: Разбиваем пространство сеткой с размерами 2^k ячеек (от 2^3 до 2^8).
   Считаем число непустых ячеек через 2D-гистограмму.
6. Регрессия МНК: Строим зависимость ln(счетчики) от ln(1 / размер).
   Угловой коэффициент прямой — метрическая размерность аттрактора.
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_attractor(c):
    # Матрицы и векторы по условию задания
    A1 = np.array([[-0.346, -0.760], [-0.196, -0.688]])
    B1 = np.array([[44.021], [69.862]])

    A2 = np.array([[0.374, 0.180], [0.208, 0.816]])
    B2 = np.array([[-2.256], [-27.600]])

    A3 = np.array([[-0.397, 0.760], [0.716, 0.456]])
    B3 = np.array([[-41.704], [-51.329]])

    M = [A1, A2, A3]
    V = [B1, B2, B3]

    pts = np.zeros((2, c))
    curr = np.array([[0.0], [0.0]])

    for i in range(c):
        idx = np.random.randint(0, 3)
        curr = M[idx] @ curr + V[idx]
        pts[:, i] = curr[:, 0]

    # Отбрасываем первые 100 точек (переходный процесс)
    return pts[0, 100:], pts[1, 100:]

def calc_dimension_and_regression_pts(xv, yv):
    # Находим границы аттрактора
    x_min, x_max = xv.min(), xv.max()
    y_min, y_max = yv.min(), yv.max()
    span = max(x_max - x_min, y_max - y_min)

    # Степени двойки для разбиения (от 2^3=8 до 2^8=256 корзин)
    powers = np.arange(3, 9)
    sizes = []
    counts = []

    for p in powers:
        b = 2**p
        eps = span / b
        # Считаем количество непустых ячеек с помощью двумерной гистограммы
        H, _, _ = np.histogram2d(xv, yv, bins=b, range=[[x_min, x_min + span], [y_min, y_min + span]])
        counts.append(np.count_nonzero(H))
        sizes.append(eps)

    sizes = np.array(sizes)
    counts = np.array(counts)

    # Логарифмические шкалы ln(1 / размер) и ln(счетчики)
    x = np.log(1.0 / sizes)
    y = np.log(counts)

    # Поиск углового коэффициента по МНК
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept, x, y

# Разное количество точек
point_counts = [10000, 50000, 150000]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=120)

for idx, c in enumerate(point_counts):
    xv, yv = generate_attractor(c)
    slope, intercept, x, y = calc_dimension_and_regression_pts(xv, yv)

    # Верхний ряд: Аттрактор высокого качества (мелкие точки для четкости)
    axes[0, idx].scatter(xv, yv, s=0.08, color='darkcyan', alpha=0.7)
    axes[0, idx].set_title(f"Аттрактор: {c} точек")
    axes[0, idx].axis('equal')
    axes[0, idx].axis('off')

    # Нижний ряд: График регрессии МНК
    axes[1, idx].scatter(x, y, color='teal', s=40, zorder=3, label='Точки разбиения')
    axes[1, idx].plot(x, slope * x + intercept, color='darkred', linestyle='--', linewidth=1.5,
                       label=f'МНК: $D = {slope:.3f}$')
    axes[1, idx].set_title(f"Линейная регрессия (N={c})")
    axes[1, idx].set_xlabel(r"$\ln(1 / \varepsilon)$")
    axes[1, idx].set_ylabel(r"$\ln(N(\varepsilon))$")
    axes[1, idx].grid(True, linestyle=':', alpha=0.6)
    axes[1, idx].legend()

plt.tight_layout()
plt.show()