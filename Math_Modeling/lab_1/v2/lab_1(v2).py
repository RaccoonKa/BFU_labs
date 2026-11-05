import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.patches import Patch


# Формулы
def U1(P, Delta, m1, m2):
    # U1 = (P + m2*Delta) / (m1 + m2)
    M = m1 + m2
    return (P + m2 * Delta) / M


def U2(P, Delta, m1, m2):
    # U2 = (P - m1*Delta) / (m1 + m2)
    M = m1 + m2
    return (P - m1 * Delta) / M


def E0(P, Delta, m1, m2):
    # E0 = P^2/(2M) + m1*m2*Delta^2/(2M)
    M = m1 + m2
    mu = m1 * m2 / M
    return P**2 / (2 * M) + 0.5 * mu * Delta**2


def E_after(P, m1, m2):
    # V1 = min(P,0)/m1, V2 = max(P,0)/m2
    # E = 1/2*m1*V1^2 + 1/2*m2*V2^2
    return np.where(P >= 0, P**2 / (2 * m2), P**2 / (2 * m1))


def valid_mask(P, Delta, m1, m2):
    # Допустимая область:
    # Delta > 0
    # P >= 0: 0 <= P <= m2*Delta
    # P < 0:  -min(m1,m2)*Delta < P < 0
    min_m = np.minimum(m1, m2)
    return (Delta > 0) & (
        ((P >= 0) & (P <= m2 * Delta)) |
        ((P < 0) & (P > -min_m * Delta))
    )


# Сетка для P и Delta
P_vals = np.linspace(-10, 10, 600)
Delta_vals = np.linspace(0.0, 5.0, 600)
PP, DD = np.meshgrid(P_vals, Delta_vals)


# Окно и ползунки
fig = plt.figure(figsize=(11, 7))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(bottom=0.25)

ax_m1 = fig.add_axes([0.25, 0.12, 0.5, 0.03])
ax_m2 = fig.add_axes([0.25, 0.07, 0.5, 0.03])

s_m1 = Slider(ax_m1, 'm1', 0.5, 5.0, valinit=1.0)
s_m2 = Slider(ax_m2, 'm2', 0.5, 5.0, valinit=1.0)


# Функция обновления графика
def update(val=None):
    m1 = s_m1.val
    m2 = s_m2.val

    mask = valid_mask(PP, DD, m1, m2)

    E0_plot = np.where(mask, E0(PP, DD, m1, m2), np.nan)
    E_plot  = np.where(mask, E_after(PP, m1, m2), np.nan)

    ax.clear()

    ax.plot_surface(PP, DD, E0_plot, cmap='viridis', alpha=0.85, linewidth=0)
    ax.plot_surface(PP, DD, E_plot,  cmap='plasma',  alpha=0.85, linewidth=0)

    ax.set_xlabel('P = m1 U1 + m2 U2')
    ax.set_ylabel('Delta = U1 - U2')
    ax.set_zlabel('Energy')
    ax.set_title(f'm1 = {m1:.2f},   m2 = {m2:.2f}')

    vals = np.concatenate([
        E0_plot[np.isfinite(E0_plot)],
        E_plot[np.isfinite(E_plot)]
    ])

    if vals.size > 0:
        ax.set_zlim(0, np.max(vals) * 1.1)

    ax.legend(
        handles=[
            Patch(color=plt.get_cmap('viridis')(0.5), label='E0 before'),
            Patch(color=plt.get_cmap('plasma')(0.5),  label='E after'),
        ],
        loc='upper left'
    )

    ax.view_init(elev=25, azim=-60)
    fig.canvas.draw_idle()

s_m1.on_changed(update)
s_m2.on_changed(update)

update()
plt.show()
