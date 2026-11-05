import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from matplotlib.lines import Line2D

a = 0.0

def system(t, state):
    x, y, z = state
    return [x*z, x*y, y*z]

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
fig.patch.set_facecolor('white')

def out_of_bounds(t, state):
    return 4.0 - np.max(np.abs(state))
out_of_bounds.terminal = True

np.random.seed(42)
ics = (np.random.rand(60, 3) - 0.5) * 3 

for ic in ics:
    sol_f = solve_ivp(system, (0, 15), ic, method='RK45', 
                      events=out_of_bounds, max_step=0.1)
    ax.plot(sol_f.y[0], sol_f.y[1], sol_f.y[2], 
            color='#1f77b4', alpha=0.4, lw=1.2)
    
    sol_b = solve_ivp(system, (0, -15), ic, method='RK45', 
                      events=out_of_bounds, max_step=0.1)
    ax.plot(sol_b.y[0], sol_b.y[1], sol_b.y[2], 
            color='#ff7f0e', alpha=0.4, lw=1.2)

pos_axis = np.linspace(0.05, 3, 50)
zeros = np.zeros(50)

ax.plot(pos_axis, zeros, zeros, color='red', lw=4, zorder=10)
ax.plot(zeros, pos_axis, zeros, color='red', lw=4, zorder=10)
ax.plot(zeros, zeros, pos_axis, color='red', lw=4, zorder=10)

ax.scatter(0, 0, 0, color='black', s=80, edgecolor='white', zorder=15)

legend_elements = [
    Line2D([0], [0], color='red', lw=3, label='Неустойчивые полуоси (+x, +y, +z)'),
    Line2D([0], [0], color='#1f77b4', lw=2, label='Прямое время (t > 0)'),
    Line2D([0], [0], color='#ff7f0e', lw=2, label='Обратное время (t < 0)')
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=11, 
          framealpha=0.9, title="Легенда", title_fontproperties={'weight':'bold'})

ax.set_title("Фазовый портрет при a = 0", fontweight='bold', fontsize=14)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_zlim(-3, 3)
ax.set_xlabel('X', fontweight='bold')
ax.set_ylabel('Y', fontweight='bold')
ax.set_zlabel('Z', fontweight='bold')

ax.view_init(elev=30, azim=45)
plt.tight_layout()
plt.show()