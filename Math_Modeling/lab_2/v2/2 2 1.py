import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

a = 1.0

def system(t, state):
    x, y, z = state
    return [a*y + x*z, a*z + x*y, a*x + y*z]

points = [
    (0, 0, 0, "O(0,0,0)\nнеустойчивая"),
    (a, a, -a, "P1(a,a,-a)\nнеустойчивая"),
    (a, -a, a, "P2(a,-a,a)\nнеустойчивая"),
    (-a, a, a, "P3(-a,a,a)\nнеустойчивая"),
    (-a, -a, -a, "P4(-a,-a,-a)\nустойчивая")
]

fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('white')

for idx, (px, py, pz, title) in enumerate(points):
    ax = fig.add_subplot(2, 3, idx + 1, projection='3d')
    pt = np.array([px, py, pz])
    
    def get_out_of_bounds(center):
        def out_of_bounds(t, state):
            return 1.5 - np.max(np.abs(state - center))
        out_of_bounds.terminal = True
        return out_of_bounds
        
    event = get_out_of_bounds(pt)
    
    np.random.seed(42 + idx) 
    ics = []
    for _ in range(50):
        vec = np.random.randn(3)
        vec /= np.linalg.norm(vec)
        ics.append(pt + vec * np.random.uniform(0.1, 0.7))
        
    for ic in ics:
        sol_f = solve_ivp(system, (0, 5), ic, method='RK45', 
                          events=event, max_step=0.05)
        ax.plot(sol_f.y[0], sol_f.y[1], sol_f.y[2], 
                color='#1f77b4', alpha=0.5, lw=1.2)
        
        if (px, py, pz) != (-a, -a, -a):
            sol_b = solve_ivp(system, (0, -5), ic, method='RK45', 
                              events=event, max_step=0.05)
            ax.plot(sol_b.y[0], sol_b.y[1], sol_b.y[2], 
                    color='#ff7f0e', alpha=0.4, lw=1.2)
            
    is_stable = (px == -a and py == -a and pz == -a)
    color = 'lime' if is_stable else 'red'
    ax.scatter(*pt, color=color, s=120, edgecolor='black', zorder=10)
    
    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlim(px - 1.5, px + 1.5)
    ax.set_ylim(py - 1.5, py + 1.5)
    ax.set_zlim(pz - 1.5, pz + 1.5)
    
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    ax.view_init(elev=30, azim=45)

ax_legend = fig.add_subplot(2, 3, 6)
ax_legend.axis('off')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
           markeredgecolor='black', markersize=12, label='Неустойчивая точка'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', 
           markeredgecolor='black', markersize=12, label='Устойчивая точка'),
    Line2D([0], [0], color='#1f77b4', lw=3, label='Прямое время (t > 0)\nРеальное движение системы'),
    Line2D([0], [0], color='#ff7f0e', lw=3, label='Обратное время (t < 0)\nМатематический реверс')
]

ax_legend.legend(handles=legend_elements, loc='center', fontsize=14, 
                 framealpha=0.9, title="Легенда", title_fontproperties={'weight':'bold', 'size': 16},
                 labelspacing=1.5)

plt.tight_layout()
plt.show()