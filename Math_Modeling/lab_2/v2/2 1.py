import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

fig = plt.figure(figsize=(11, 8))
ax = fig.add_subplot(111, projection='3d')

p_grid = np.linspace(0, 6, 35)
y_grid = np.linspace(-4.5, 0.5, 35)
P, Y = np.meshgrid(p_grid, y_grid)

dY = Y**2 + 4*Y + 5 - P

surf = ax.plot_surface(P, Y, dY, cmap='coolwarm', alpha=0.7, 
                       rstride=1, cstride=1, edgecolor='none')

zero_plane = ax.plot_surface(P, Y, np.zeros_like(dY), color='gray', alpha=0.3)

p_vals = np.linspace(1, 6, 150)
y_stable = -2 - np.sqrt(p_vals - 1)
y_unstable = -2 + np.sqrt(p_vals - 1)

ax.plot(p_vals, y_stable, np.zeros_like(y_stable), color='blue', linewidth=3)
ax.plot(p_vals, y_unstable, np.zeros_like(y_unstable), color='red', linestyle='--', linewidth=3)

ax.scatter(1, -2, 0, color='black', s=80, zorder=5)

legend_elements = [
    Patch(facecolor='salmon', edgecolor='none', alpha=0.7, label='Поверхность фазовой скорости y\''),
    Patch(facecolor='gray', edgecolor='none', alpha=0.4, label='Плоскость нулевой скорости (y\'=0)'),
    Line2D([0], [0], color='blue', lw=3, label='Устойчивая ветвь'),
    Line2D([0], [0], color='red', lw=3, linestyle='--', label='Неустойчивая ветвь'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='black', markersize=8, label='Точка бифуркации')
]

ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)

ax.set_xlabel("Параметр p", fontsize=12)
ax.set_ylabel("Состояние y", fontsize=12)
ax.set_zlabel("Скорость y'", fontsize=12)

ax.view_init(elev=20, azim=-125)

plt.tight_layout()
plt.show()