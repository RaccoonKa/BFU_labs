"""
Суть алгоритма нахождения метрической размерности DLA:
1. Случайное блуждание частиц по сетке с прилипанием к кластеру.
2. Находим минимальные и максимальные координаты (bounding box).
3. Обрезанный кластер помещаем в центр квадратного массива
   размером 2^k x 2^k, где 2^k — ближайшая сверху степень двойки.
4. Box-counting: Разбиваем сетку на квадратные блоки размером от 2^(k-1) до 2^2 (4 пикселя) включительно.
   Считаем количество непустых блоков (где есть хотя бы 1 частица).
5. Переходим к координатам x = ln(1 / размер_ячейки), y = ln(число_блоков).
6. Линейная регрессия (МНК): Методом np.polyfit строим прямую y = D * x + b.
   Угловой коэффициент D и есть искомая фрактальная (метрическая) размерность.
"""

import numpy as np
import matplotlib.pyplot as plt


def dla_generate(n_particles):
    # Увеличил размер поля, так как 3000 точек раскидывают очень длинные ветки!
    g_size = 800
    grid = np.zeros((g_size, g_size), dtype=bool)
    c = g_size // 2
    grid[c, c] = True  # Ставим затравку в центр

    # 8 направлений шага (окрестность Мура)
    moves = np.array([[-1, 0], [1, 0], [0, -1], [0, 1],
                      [-1, -1], [-1, 1], [1, -1], [1, 1]])

    r_max = 2.0
    stuck = 1

    while stuck < n_particles:
        # Спавним частицу чуть дальше текущего радиуса кластера
        angle = np.random.uniform(0, 2 * np.pi)
        r_spawn = r_max + 3
        x = int(c + r_spawn * np.cos(angle))
        y = int(c + r_spawn * np.sin(angle))

        # Даем частице больше попыток найти кластер
        for _ in range(1000):
            # ЖЕЛЕЗОБЕТОННАЯ ПРОВЕРКА: если подошли к краю поля - убиваем частицу,
            # чтобы не словить ошибку IndexError
            if x <= 1 or x >= g_size - 2 or y <= 1 or y >= g_size - 2:
                break

            # Если рядом есть хоть одна зафиксированная точка — прилипаем
            if grid[x - 1:x + 2, y - 1:y + 2].any():
                grid[x, y] = True
                stuck += 1
                dist = np.hypot(x - c, y - c)
                if dist > r_max:
                    r_max = dist
                break

            # Случайный шаг
            step = moves[np.random.randint(0, 8)]
            x += step[0]
            y += step[1]

            # Если частица улетела слишком далеко в пустоту — удаляем и спавним новую
            if np.hypot(x - c, y - c) > r_max + 15:
                break

    return grid


def prepare_cluster(grid):
    # Пункт 1: Определить границы кластера
    rows, cols = np.where(grid)
    min_r, max_r = rows.min(), rows.max()
    min_c, max_c = cols.min(), cols.max()
    cropped = grid[min_r:max_r + 1, min_c:max_c + 1]

    # Пункт 2: Дополнить нулями до квадрата со стороной 2^k
    h, w = cropped.shape
    max_dim = max(h, w)
    k = int(np.ceil(np.log2(max_dim)))
    k = max(k, 3)  # Защита, чтобы k было не меньше 3
    p = 2 ** k

    padded = np.zeros((p, p), dtype=int)
    # Центрируем фрактал в новом окне
    sr = (p - h) // 2
    sc = (p - w) // 2
    padded[sr:sr + h, sc:sc + w] = cropped.astype(int)

    return padded, k


def calc_dimension_and_regression(Z, k):
    # Пункт 3: Размеры ячеек — степени двойки на убывание до 2^2 (4) включительно
    powers = np.arange(k - 1, 1, -1)
    sizes = 2 ** powers
    counts = []

    for size in sizes:
        # Суммируем блоки размера size x size
        reduced = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], size), axis=0),
            np.arange(0, Z.shape[1], size), axis=1)
        # Считаем непустые блоки
        counts.append(np.count_nonzero(reduced))

    counts = np.array(counts)

    # Пункт 4: Логарифмические шкалы ln(1 / размер) и ln(счетчики)
    x = np.log(1.0 / sizes)
    y = np.log(counts)

    # Пункт 5: Линейная регрессия методом МНК
    slope, intercept = np.polyfit(x, y, 1)
    return slope, intercept, x, y


# Тестовые значения из тетрадки одногруппницы
test_points = [100, 1000, 3000]

fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=120)

for idx, n in enumerate(test_points):
    raw_grid = dla_generate(n)
    padded_grid, power_k = prepare_cluster(raw_grid)
    slope, intercept, x, y = calc_dimension_and_regression(padded_grid, power_k)

    # Верхний ряд: Рисунки высокого качества
    axes[0, idx].imshow(padded_grid, cmap='magma', origin='lower')
    axes[0, idx].set_title(f"Кластер DLA: {n} частиц\nСетка: {2 ** power_k}x{2 ** power_k} ($2^{{{power_k}}}$)")
    axes[0, idx].axis('off')

    # Нижний ряд: Прямая МНК
    axes[1, idx].scatter(x, y, color='red', s=40, zorder=3, label='Точки разбиения')
    axes[1, idx].plot(x, slope * x + intercept, color='navy', linestyle='--', linewidth=1.5,
                      label=f'МНК: $D = {slope:.3f}$')
    axes[1, idx].set_title(f"Линейная регрессия (N={n})")
    axes[1, idx].set_xlabel(r"$\ln(1 / \varepsilon)$")
    axes[1, idx].set_ylabel(r"$\ln(N(\varepsilon))$")
    axes[1, idx].grid(True, linestyle=':', alpha=0.6)
    axes[1, idx].legend()

plt.tight_layout()
plt.show()