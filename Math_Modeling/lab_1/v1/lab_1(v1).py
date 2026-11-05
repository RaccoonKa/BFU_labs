import numpy as np
import matplotlib.pyplot as plt


def calculate_collision(m1, m2, v1, v2):
    P = m1 * v1 + m2 * v2
    Ek_initial = 0.5 * m1 * v1 ** 2 + 0.5 * m2 * v2 ** 2

    A = m1 * (m1 + m2)
    B = -2 * P * m1
    C = P ** 2 - 1.98 * Ek_initial * m2

    D = B ** 2 - 4 * A * C

    root1 = (-B + np.sqrt(D)) / (2 * A)
    root2 = (-B - np.sqrt(D)) / (2 * A)

    u1 = min(root1, root2)

    u2 = (P - m1 * u1) / m2

    E1_initial = 0.5 * m1 * v1 ** 2
    E2_initial = 0.5 * m2 * v2 ** 2
    E1_final = 0.5 * m1 * u1 ** 2
    E2_final = 0.5 * m2 * u2 ** 2

    return E1_initial, E2_initial, E1_final, E2_final


m2 = 5.0
v1 = 10.0
v2 = 2.0
m1_values = np.linspace(1, 20, 100)

E1_in_list = []
E2_in_list = []
E1_out_list = []
E2_out_list = []

for m1 in m1_values:
    e1_in, e2_in, e1_out, e2_out = calculate_collision(m1, m2, v1, v2)
    E1_in_list.append(e1_in)
    E2_in_list.append(e2_in)
    E1_out_list.append(e1_out)
    E2_out_list.append(e2_out)

plt.figure(figsize=(10, 6))

plt.plot(m1_values, E1_in_list, label="Энергия шара 1 (до)", linestyle="--", color="blue")
plt.plot(m1_values, E1_out_list, label="Энергия шара 1 (после)", linewidth=2, color="blue")

plt.plot(m1_values, E2_in_list, label="Энергия шара 2 (до)", linestyle="--", color="red")
plt.plot(m1_values, E2_out_list, label="Энергия шара 2 (после)", linewidth=2, color="red")

plt.title("Зависимость энергий шаров от массы первого шара (m1)")
plt.xlabel("Масса первого шара m1, кг")
plt.ylabel("Кинетическая энергия, Дж")
plt.legend()
plt.grid(True)
plt.show()