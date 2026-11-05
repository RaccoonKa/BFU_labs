# Вычисление точных значений площади и интеграла
import sympy as sp

x, y = sp.symbols('x y')

y_top = 4 - x**2 # верхняя граница фигуры
y_bottom = x**2 - 4 # нижняя граница фигуры

intersections = sp.solve(sp.Eq(y_top, y_bottom), x)
x_min, x_max = intersections[0], intersections[1]

exact_area = sp.integrate(1, (y, y_bottom, y_top), (x, x_min, x_max))

f_z = x + y**2 # искомая функция
exact_integral = sp.integrate(f_z, (y, y_bottom, y_top), (x, x_min, x_max))

print(f"Пределы интегрирования по x: от {x_min} до {x_max}")
print(f"Точная площадь S = {exact_area}")
print(f"Точное значение интеграла I = {exact_integral}")