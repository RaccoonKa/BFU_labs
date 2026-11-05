import numpy as np
import matplotlib.pyplot as plt


DELTA = 0.05
N_ITERS = 4

def f(x):
    return 1 + x - 2 * x**2

x_star1 = np.sqrt(2) / 2
x_star2 = -np.sqrt(2) / 2

x_vals = np.linspace(-1.5, 1.3, 500)
y_vals = f(x_vals)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

def plot_cobweb(ax, x0, color, label):
    ax.plot(x0, x0, marker='o', color=color, markersize=7, zorder=5)
    
    x, y = x0, x0
    xs, ys = [x], [y]
    
    for _ in range(N_ITERS):
        next_y = f(x)

        xs.append(x)
        ys.append(next_y)

        xs.append(next_y)
        ys.append(next_y)
        
        x = next_y
        
    ax.plot(xs, ys, color=color, alpha=0.8, linewidth=1.5, label=label)

ax1.plot(x_vals, y_vals, label='$f(x) = 1 + x - 2x^2$', color='#1f77b4', lw=2)
ax1.plot(x_vals, x_vals, label='$y = x$', color='gray', linestyle='--')

ax1.plot(x_star1, x_star1, marker='o', color='black', markersize=12, zorder=6, label=f'$x_1^* \\approx 0.707$')

plot_cobweb(ax1, x_star1 + DELTA, 'red', f'Старт: $x_1 + {DELTA}$')
plot_cobweb(ax1, x_star1 - DELTA, 'orange', f'Старт: $x_1 - {DELTA}$')

ax1.set_xlim(-1.2, 1.2)
ax1.set_ylim(-1.5, 1.5)
ax1.set_title('Поведение вблизи $x_1$')
ax1.set_xlabel('$x_n$')
ax1.set_ylabel('$x_{n+1}$')
ax1.legend()
ax1.grid(True)

ax2.plot(x_vals, y_vals, label='$f(x) = 1 + x - 2x^2$', color='#1f77b4', lw=2)
ax2.plot(x_vals, x_vals, label='$y = x$', color='gray', linestyle='--')

ax2.plot(x_star2, x_star2, marker='o', color='black', markersize=12, zorder=6, label=f'$x_2^* \\approx -0.707$')

plot_cobweb(ax2, x_star2 + DELTA, 'purple', f'Старт: $x_2 + {DELTA}$')
plot_cobweb(ax2, x_star2 - DELTA, 'magenta', f'Старт: $x_2 - {DELTA}$')

ax2.set_xlim(-1.5, 0.5)
ax2.set_ylim(-2, 1)
ax2.set_title('Поведение вблизи $x_2$')
ax2.set_xlabel('$x_n$')
ax2.set_ylabel('$x_{n+1}$')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()