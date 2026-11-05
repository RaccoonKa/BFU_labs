import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
import math

def simulate_dla(num_particles, max_steps, spawn_margin, kill_margin):
    cluster = {(0, 0)}
    points_history = [(0, 0)]
    r_max = 0

    moves = [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    particles_added = 0
    while particles_added < num_particles:
        angle = np.random.uniform(0, 2 * np.pi)
        r_spawn = r_max + spawn_margin
        x = int(r_spawn * np.cos(angle))
        y = int(r_spawn * np.sin(angle))
        
        if (x + y) % 2 != 0:
            x += 1

        r_kill = r_max + kill_margin
        steps_taken = 0

        while True:
            if steps_taken >= max_steps:
                break

            stuck = False
            for dx, dy in moves:
                if (x + dx, y + dy) in cluster:
                    stuck = True
                    break
            
            if stuck:
                cluster.add((x, y))
                points_history.append((x, y))
                r_max = max(r_max, math.hypot(x, y))
                particles_added += 1
                break
            
            step = moves[np.random.randint(0, 4)]
            x += step[0]
            y += step[1]
            steps_taken += 1

            if math.hypot(x, y) > r_kill:
                break

    return np.array(points_history)


def calculate_metric_dimension(points):
    x, y = points[:, 0], points[:, 1]

    x = x - np.min(x)
    y = y - np.min(y)
    max_range = max(np.max(x), np.max(y))

    scales = np.logspace(0.5, np.log10(max_range / 2), num=15, endpoint=False)
    sizes = np.unique(np.floor(scales)).astype(int)
    sizes = sizes[sizes > 0]

    counts = []
    for size in sizes:
        bins_x = np.arange(0, max_range + size, size)
        bins_y = np.arange(0, max_range + size, size)
        H, _, _ = np.histogram2d(x, y, bins=(bins_x, bins_y))
        counts.append(np.sum(H > 0))

    log_sizes = np.log(1.0 / sizes)
    log_counts = np.log(counts)

    # ФИКС ДЛЯ ПРЕПОДА: Отбрасываем крайние точки (2 самые большие и 2 самые мелкие коробки)
    # Это убирает эффекты конечного размера и насыщения на графике
    valid_idx = slice(2, -2)
    valid_log_sizes = log_sizes[valid_idx]
    valid_log_counts = log_counts[valid_idx]

    # Считаем регрессию ТОЛЬКО по центральному (линейному) участку
    slope, intercept, _, _, _ = linregress(valid_log_sizes, valid_log_counts)

    # Возвращаем valid_idx, чтобы отрисовать серым отброшенные точки
    return slope, log_sizes, log_counts, intercept, valid_idx


def visualize_results(results):
    num_runs = len(results)
    fig, axs = plt.subplots(2, num_runs, figsize=(6 * num_runs, 10))

    if num_runs == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    for col, res in enumerate(results):
        pts = res['points']
        dim = res['dimension']
        n_pts = res['n_points']
        n_stps = res['n_steps']
        s_margin = res['spawn_margin']
        k_margin = res['kill_margin']

        ax_top = axs[0, col]
        sc = ax_top.scatter(pts[:, 0], pts[:, 1], c=range(len(pts)), cmap='YlOrRd', s=5, marker='s')
        ax_top.set_title(
            f"DLA: {n_pts} точек\n(Шагов: {n_stps}, Spawn: +{s_margin}, Kill: +{k_margin})\nРазмерность $D \\approx {dim:.3f}$")
        ax_top.set_aspect('equal')
        plt.colorbar(sc, ax=ax_top, label="Время прилипания")

        ax_bot = axs[1, col]
        ls, lc, intercept, v_idx = res['log_sizes'], res['log_counts'], res['intercept'], res['valid_idx']

        ax_bot.plot(ls, lc, 'o', color='lightgray', label="Отброшенные (краевые эффекты)")

        ax_bot.plot(ls[v_idx], lc[v_idx], 'bo', label="Используемые для регрессии")

        ax_bot.plot(ls, intercept + dim * ls, 'r-', label=f"Регрессия (D = {dim:.3f})")

        ax_bot.set_title(f"Оценка размерности (Запуск {col + 1})")
        ax_bot.set_xlabel("log(1 / размер ячейки)")
        ax_bot.set_ylabel("log(кол-во непустых ячеек)")
        ax_bot.legend()
        ax_bot.grid(True)

    plt.tight_layout()
    plt.show()


def visualize_results(results):
    num_runs = len(results)
    fig, axs = plt.subplots(2, num_runs, figsize=(6 * num_runs, 10))
    
    if num_runs == 1:
        axs = np.array([[axs[0]], [axs[1]]])

    for col, res in enumerate(results):
        pts = res['points']
        dim = res['dimension']
        n_pts = res['n_points']
        n_stps = res['n_steps']
        s_margin = res['spawn_margin']
        k_margin = res['kill_margin']

        ax_top = axs[0, col]
        sc = ax_top.scatter(pts[:, 0], pts[:, 1], c=range(len(pts)), cmap='cividis_r', s=5, marker='s')
        ax_top.set_title(f"DLA: {n_pts} точек\n(Шагов: {n_stps}, Spawn: +{s_margin}, Kill: +{k_margin})\nРазмерность $D \\approx {dim:.3f}$")
        ax_top.set_aspect('equal')
        plt.colorbar(sc, ax=ax_top, label="Время прилипания")

        ax_bot = axs[1, col]
        ls, lc, intercept = res['log_sizes'], res['log_counts'], res['intercept']
        ax_bot.plot(ls, lc, 'o', label="Данные Box-Counting")
        ax_bot.plot(ls, intercept + dim * ls, 'r-', label=f"Регрессия (D = {dim:.3f})")
        ax_bot.set_title(f"Оценка размерности (Запуск {col+1})")
        ax_bot.set_xlabel("log(1 / размер ячейки)")
        ax_bot.set_ylabel("log(кол-во непустых ячеек)")
        ax_bot.legend()
        ax_bot.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    spawn_margin = 5
    kill_margin = 20

    runs = [
        [1000, 1000],
        [1000, 2000],
        [2000, 1000],
        [2000, 2000],
    ]

    results_data = []

    print("Начинаем моделирование DLA...")
    for idx, (n_points, n_steps) in enumerate(runs):
        print(f"[{idx+1}/{len(runs)}] Генерация {n_points} точек (лимит шагов: {n_steps})...")
        
        points = simulate_dla(n_points, n_steps, spawn_margin, kill_margin)
        dim, log_sizes, log_counts, intercept, valid_idx = calculate_metric_dimension(points)
        
        results_data.append({
            'n_points': n_points,
            'n_steps': n_steps,
            'spawn_margin': spawn_margin,
            'kill_margin': kill_margin,
            'points': points,
            'dimension': dim,
            'log_sizes': log_sizes,
            'log_counts': log_counts,
            'intercept': intercept,
            'valid_idx' : valid_idx,
        })
    
    print("Моделирование завершено. Строим графики...")
    visualize_results(results_data)


if __name__ == "__main__":
    main()