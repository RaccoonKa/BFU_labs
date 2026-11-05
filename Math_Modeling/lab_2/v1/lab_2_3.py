import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return 2 + x - 3 * x**2

x_vals = np.linspace(-1.2, 1.2, 400)
y_vals = f(x_vals)

fig, ax = plt.subplots(figsize=(8, 8))

ax.plot(x_vals, y_vals, label='$f(x) = 2 + x - 3x^2$', color='blue', linewidth=2)
ax.plot([-1.2, 1.2], [-1.2, 1.2], label='$y = x$', color='black', linestyle='--', linewidth=1.5)

x_star1 = np.sqrt(2/3)
x_star2 = -np.sqrt(2/3)
ax.plot(x_star1, x_star1, 'ro', markersize=8, zorder=5, label='Особые точки')
ax.plot(x_star2, x_star2, 'ro', markersize=8, zorder=5)

x0 = x_star1 + 0.02
x_traj, y_traj = [x0], [0]

for _ in range(3):
    y_next = f(x0)
    x_traj.extend([x0, x0])
    y_traj.extend([y_traj[-1], y_next])
    x_traj.extend([x0, y_next])
    y_traj.extend([y_next, y_next])
    x0 = y_next

ax.plot(x_traj, y_traj, color='green', linewidth=1.5, label='Траектория отталкивания')

ax.set_title('Паутинная диаграмма (Лестница Ламере)')
ax.set_xlabel('$x_n$')
ax.set_ylabel('$x_{n+1}$')
ax.grid(True)
ax.legend()
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-1.5, 2.5)

plt.show()
