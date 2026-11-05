import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

a = 1.0

def system(t, state):
    x, y, z = state
    dxdt = x - a * y * z
    dydt = y - a * x * z
    dzdt = z - a * x * y
    return [dxdt, dydt, dzdt]

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

points = [
    (0, 0, 0),
    (1, 1, 1),
    (1, -1, -1),
    (-1, 1, -1),
    (-1, -1, 1)
]

for p in points:
    color = 'black' if p == (0,0,0) else 'red'
    ax.scatter(*p, color=color, s=50, edgecolors='k', zorder=5)

t_span = (0, 2.5)
np.random.seed(42)

for p in points:
    for _ in range(5):
        initial_state = np.array(p) + np.random.uniform(-0.1, 0.1, 3)
        sol = solve_ivp(system, t_span, initial_state, dense_output=True, max_step=0.01)
        ax.plot(sol.y[0], sol.y[1], sol.y[2], color='blue', alpha=0.3, linewidth=1)

ax.set_title('Фазовый портрет системы (a=1)')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_xlim([-2, 2])
ax.set_ylim([-2, 2])
ax.set_zlim([-2, 2])

plt.show()
