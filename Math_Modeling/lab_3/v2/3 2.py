import sympy as sp

r, phi = sp.symbols('r phi')
pi = sp.pi

r_inner = phi
r_outer = phi + pi

area_integrand = r

area_inner = sp.integrate(area_integrand, (r, r_inner, r_outer))
exact_area = sp.integrate(area_inner, (phi, 0, 6*pi))
exact_area_simplified = sp.expand(exact_area)

print("ПЛОЩАДЬ")
# print("Точное аналитическое значение площади:")
# sp.pprint(exact_area_simplified)
print(f"Строчный формат: {exact_area_simplified}")
print(f"Численное значение: {exact_area_simplified.evalf():.4f}\n")

x = r * sp.cos(phi)
y = r * sp.sin(phi)

integral_integrand = (x**2 + y**3) * r

integral_inner = sp.integrate(integral_integrand, (r, r_inner, r_outer))

exact_integral = sp.integrate(integral_inner, (phi, 0, 6*pi))

exact_integral_simplified = sp.expand(exact_integral)

print("ИНТЕГРАЛ:")
# print("Точное аналитическое значение интеграла:")
# sp.pprint(exact_integral_simplified)
print(f"Строчный формат: {exact_integral_simplified}")
print(f"Численное значение: {exact_integral_simplified.evalf():.4f}")