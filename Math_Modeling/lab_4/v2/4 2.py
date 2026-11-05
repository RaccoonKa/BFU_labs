import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from typing import Tuple, Dict, Any

def generate_attractor(matrices: np.ndarray, vectors: np.ndarray, num_points: int, drop_first: int = 100) -> np.ndarray:
    """
    Генерирует точки аттрактора с помощью метода Игры в хаос.
    """
    points = np.zeros((num_points, 2))
    current_point = np.array([0.0, 0.0])
    
    num_transforms = len(matrices)
    choices = np.random.choice(num_transforms, size=num_points + drop_first)
    
    for i in range(num_points + drop_first):
        choice = choices[i]
        current_point = matrices[choice] @ current_point + vectors[choice]
        
        if i >= drop_first:
            points[i - drop_first] = current_point
            
    return points


def calc_box_counting_dimension(points: np.ndarray, num_scales: int = 15,
                                trim_start: int = 2, trim_end: int = 3) -> Tuple[float, np.ndarray, np.ndarray, float]:
    """
    Вычисляет размерность, отбрасывая крайние точки (макро- и микромасштабы),
    чтобы избежать эффектов насыщения конечной выборки.
    """
    min_p, max_p = np.min(points, axis=0), np.max(points, axis=0)
    max_range = np.max(max_p - min_p)

    epsilons = max_range / np.logspace(np.log10(2), np.log10(200), num_scales)

    N_boxes = []
    for eps in epsilons:
        boxes = np.floor(points / eps)
        unique_boxes = np.unique(boxes, axis=0)
        N_boxes.append(len(unique_boxes))

    epsilons = np.array(epsilons)
    N_boxes = np.array(N_boxes)

    x = np.log(1 / epsilons)
    y = np.log(N_boxes)

    x_fit = x[trim_start: -trim_end]
    y_fit = y[trim_start: -trim_end]

    slope, intercept, r_value, p_value, std_err = linregress(x_fit, y_fit)

    return slope, x, y, r_value ** 2

def main():
    """
    Точка входа. Инициализирует параметры СИФ, формирует массив экспериментов
    и рендерит результаты всех прогонов на одном Figure (в сетке).
    """
    MATRICES = np.array([
        [[-0.231, -0.092], [0.196, 0.800]],
        [[0.857, -0.196], [0.348, -0.144]],
        [[-0.331, -0.020], [-0.276, 0.768]]
    ])
    
    VECTORS = np.array([
        [35.723, -68.830],
        [11.678, 35.113],
        [-47.715, 45.252]
    ])
    
    DROP_FIRST = 500
    NUM_SCALES = 15
    
    RUN_CONFIGS = [
        {"num_points": 10_000,  "marker_size": 1.5,  "alpha": 0.8, "color": "#1f77b4"},
        {"num_points": 50_000,  "marker_size": 0.5,  "alpha": 0.6, "color": "#2ca02c"},
        {"num_points": 250_000, "marker_size": 0.05, "alpha": 0.4, "color": "#9467bd"}
    ]
    
    fig, axes = plt.subplots(len(RUN_CONFIGS), 2, figsize=(14, 5 * len(RUN_CONFIGS)))
    fig.suptitle("Исследование аттрактора и его размерности", fontsize=18)
    
    for i, config in enumerate(RUN_CONFIGS):
        points = generate_attractor(MATRICES, VECTORS, config['num_points'], DROP_FIRST)
        dimension, x, y, r_squared = calc_box_counting_dimension(points, NUM_SCALES)
        
        ax_attractor = axes[i, 0]
        ax_dimension = axes[i, 1]
        
        ax_attractor.scatter(points[:, 0], points[:, 1], 
                             s=config['marker_size'], 
                             c=config['color'], 
                             alpha=config['alpha'], 
                             linewidths=0)
        ax_attractor.set_title(f"Точек: {config['num_points']:,}")
        ax_attractor.set_xlabel("X")
        ax_attractor.set_ylabel("Y")
        ax_attractor.axis('equal')
        ax_attractor.grid(True, linestyle='--', alpha=0.5)
        
        ax_dimension.scatter(x, y, color='blue', zorder=5)
        trend_line = dimension * x + (y.mean() - dimension * x.mean())
        ax_dimension.plot(x, trend_line, color='red', linestyle='--', 
                          label=f'Размерность D ≈ {dimension:.3f}\n$R^2$ = {r_squared:.4f}')
        ax_dimension.set_title(f"Box-Counting ({config['num_points']:,} точек)")
        ax_dimension.set_xlabel("log(1/ε)")
        ax_dimension.set_ylabel("log(N(ε))")
        ax_dimension.legend()
        ax_dimension.grid(True, linestyle='--', alpha=0.5)
        
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()