import numpy as np
import matplotlib.pyplot as plt
import os

class Solver:
    def __init__(self, a, x_start=0, x_end=1, t_start=0, t_end=10, x_step=0.1):
        self.a = a
        self.x_start = x_start
        self.x_end = x_end
        self.t_start = t_start
        self.t_end = t_end
        self.x_step = x_step

        self.t_step = 0.95 * x_step / abs(a)
        self.lam = abs(a) * self.t_step / x_step

        self.initial = lambda x: 2*x**2 - 5*x + 5
        self.boundary_left = lambda t: t**2 - 5*t + 5
        self.boundary_right = lambda t: t**2 - 5*t + 2
        self.source = lambda x, t: 3*x

        self.nx = int((x_end - x_start) / x_step) + 1
        self.nt = int((t_end - t_start) / self.t_step) + 1

        print(f"a = {a}, шаг t = {self.t_step:.5f}, λ = {self.lam:.3f}")

    def get_grid(self):
        x = np.linspace(self.x_start, self.x_end, self.nx)
        t = np.linspace(self.t_start, self.t_end, self.nt)
        return x, t

    def initialize_rectangle(self):
        x, t = self.get_grid()
        U = np.zeros((self.nt, self.nx))
        U[0, :] = self.initial(x)

        if self.a > 0:
            for n in range(1, self.nt):
                U[n, 0] = self.boundary_left(t[n])
        else:
            for n in range(1, self.nt):
                U[n, -1] = self.boundary_right(t[n])
        return x, t, U

    def initialize_halfplane(self):
        if self.a > 0:
            x_start_ext = self.x_start - self.nt * self.x_step
            x_end_ext = self.x_end
        else:
            x_start_ext = self.x_start
            x_end_ext = self.x_end + self.nt * self.x_step

        nx_ext = int((x_end_ext - x_start_ext) / self.x_step) + 1
        x_ext = np.linspace(x_start_ext, x_end_ext, nx_ext)
        t = np.linspace(self.t_start, self.t_end, self.nt)

        U = np.zeros((self.nt, nx_ext))
        U[0, :] = self.initial(x_ext)

        return x_ext, t, U, nx_ext

    def scheme1_upwind_left(self, rectangle=True):
        if self.a <= 0:
            raise ValueError("scheme1_upwind_left предназначена только для a>0")

        if rectangle:
            x, t, U = self.initialize_rectangle()
            nx, nt = self.nx, self.nt
        else:
            x, t, U, nx = self.initialize_halfplane()
            nt = self.nt

        coef = self.a * self.t_step / self.x_step

        for n in range(nt - 1):
            for i in range(1, nx):
                U[n+1, i] = U[n, i] - coef * (U[n, i] - U[n, i-1]) + self.t_step * self.source(x[i], t[n])

        if rectangle:
            return x, t, U
        else:
            idx_start = np.argmax(x >= 0)
            idx_end = np.argmax(x > 1) - 1
            if idx_end < idx_start:
                idx_end = nx - 1
            return x[idx_start:idx_end+1], t, U[:, idx_start:idx_end+1]

    def scheme2_upwind_right(self, rectangle=True):
        if self.a >= 0:
            raise ValueError("scheme2_upwind_right предназначена только для a<0")

        if rectangle:
            x, t, U = self.initialize_rectangle()
            nx, nt = self.nx, self.nt
        else:
            x, t, U, nx = self.initialize_halfplane()
            nt = self.nt

        coef = self.a * self.t_step / self.x_step

        for n in range(nt - 1):
            for i in range(nx - 1):
                U[n+1, i] = U[n, i] - coef * (U[n, i+1] - U[n, i]) + self.t_step * self.source(x[i], t[n])

        if rectangle:
            return x, t, U
        else:
            idx_start = np.argmax(x >= 0)
            idx_end = np.argmax(x > 1) - 1
            if idx_end < idx_start:
                idx_end = nx - 1
            return x[idx_start:idx_end+1], t, U[:, idx_start:idx_end+1]

    def scheme3_implicit(self):
        if self.a <= 0:
            raise ValueError("Неявная схема реализована только для a>0")
        x, t, U = self.initialize_rectangle()
        lam = self.a * self.t_step / self.x_step

        for n in range(self.nt - 1):
            U[n+1, 0] = self.boundary_left(t[n+1])
            for i in range(1, self.nx):
                U[n+1, i] = (U[n, i] + lam * U[n+1, i-1] + self.t_step * self.source(x[i], t[n])) / (1 + lam)
        return x, t, U

    def scheme4(self):
        x, t, U = self.initialize_rectangle()
        lam = self.a * self.t_step / self.x_step
        lam2 = lam**2

        for n in range(self.nt - 1):
            for i in range(1, self.nx - 1):
                U[n+1, i] = (U[n, i]
                             - 0.5 * lam * (U[n, i+1] - U[n, i-1])
                             + 0.5 * lam2 * (U[n, i+1] - 2*U[n, i] + U[n, i-1])
                             + self.t_step * self.source(x[i], t[n]))
            if self.a > 0:
                U[n+1, 0] = self.boundary_left(t[n+1])
                U[n+1, -1] = U[n, -1] - lam * (U[n, -1] - U[n, -2]) + self.t_step * self.source(x[-1], t[n])
            else:
                U[n+1, -1] = self.boundary_right(t[n+1])
                U[n+1, 0] = U[n, 0] - lam * (U[n, 1] - U[n, 0]) + self.t_step * self.source(x[0], t[n])
        return x, t, U

    def visualize(self, x, t, U, title, filename, region, scheme, bc_info=""):
        X, T = np.meshgrid(x, t)
        fig = plt.figure(figsize = (10, 8))
        ax = fig.add_subplot(111, projection = '3d')
        surf = ax.plot_surface(X, T, U, cmap = 'plasma', edgecolor = 'none', alpha = 0.9)

        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('U(x,t)')

        full_title = f"a = {self.a}, {region}, схема {scheme}"
        if bc_info:
            full_title += f", {bc_info}"
        ax.set_title(full_title)

        ax.view_init(elev = 25, azim = -60)
        plt.tight_layout()
        plt.savefig(filename, dpi = 150)
        print(f"Сохранен: {filename}")
        plt.close()

def main():
    os.makedirs("Results_6_lab", exist_ok=True)
    graph_num = 1

    solver = Solver(a = 2)

    x, t, U = solver.scheme1_upwind_left(rectangle = False)
    solver.visualize(x, t, U, "", f"Results_6_lab/graph{graph_num}_half_scheme1_a2_var21.png",
                     "полуплоскость", 1, "")
    graph_num += 1

    x, t, U = solver.scheme1_upwind_left(rectangle = True)
    U2 = U
    solver.visualize(x, t, U, "", f"Results_6_lab/graph{graph_num}_rect_scheme1_a2_var21.png",
                     "прямоугольник", 1, "U(0,t)=t²-5t+5")
    graph_num += 1

    x, t, U = solver.scheme3_implicit()
    solver.visualize(x, t, U, "", f"Results_6_lab/graph{graph_num}_rect_scheme3_a2_var21.png",
                     "прямоугольник", 3, "U(0,t)=t²-5t+5")
    graph_num += 1

    x, t, U = solver.scheme4()
    U4 = U
    solver.visualize(x, t, U, "", f"Results_6_lab/graph{graph_num}_rect_scheme4_a2_var21.png",
                     "прямоугольник", 4, "U(0,t)=t²-5t+5")
    graph_num += 1

    solver = Solver(a=-2)

    x, t, U = solver.scheme2_upwind_right(rectangle = False)
    solver.visualize(x, t, U, "", f"Results_6_lab/graph{graph_num}_half_scheme2_a-2_var21.png",
                     "полуплоскость", 2, "")
    graph_num += 1

    x, t, U = solver.scheme2_upwind_right(rectangle = True)
    U6 = U
    solver.visualize(x, t, U, "", f"Results_6_lab/graph{graph_num}_rect_scheme2_a-2_var21.png",
                     "прямоугольник", 2, "U(1,t)=t²-5t+2")
    graph_num += 1

    x, t, U = solver.scheme4()
    U7 = U
    solver.visualize(x, t, U, "", f"Results_6_lab/graph{graph_num}_rect_scheme4_a-2_var21.png",
                     "прямоугольник", 4, "U(1,t)=t²-5t+2")

    diff = np.max(np.abs(U2 - U4))
    print(f"Максимальная разница 2 и 4: {diff:.2e}")
    diff = np.max(np.abs(U6 - U7))
    print(f"Максимальная разница 6 и 7: {diff:.2e}")

if __name__ == "__main__":
    main()