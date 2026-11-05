import numpy as np
import matplotlib.pyplot as plt

p_vals_left = np.linspace(-1, 2, 100)
p_vals_right = np.linspace(2, 5, 100)

y1_left = np.full_like(p_vals_left, -1)
y1_right = np.full_like(p_vals_right, -1)

y2_left = 1 - p_vals_left
y2_right = 1 - p_vals_right

plt.figure(figsize=(8, 6))

plt.plot(p_vals_left, y1_left, color='blue', linestyle='-', linewidth=2, label='Устойчивое равновесие')
plt.plot(p_vals_right, y1_right, color='blue', linestyle='--', linewidth=2, label='Неустойчивое равновесие')

plt.plot(p_vals_left, y2_left, color='red', linestyle='--', linewidth=2)
plt.plot(p_vals_right, y2_right, color='red', linestyle='-', linewidth=2)

plt.plot(2, -1, 'ko', markersize=8, label='Точка бифуркации (2, -1)')

plt.title('Транскритическая бифуркация')
plt.xlabel('Параметр p')
plt.ylabel('y')
plt.grid(True)
plt.legend()
plt.xlim(-1, 5)
plt.ylim(-4, 3)

plt.show()
